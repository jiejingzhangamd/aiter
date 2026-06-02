#!/usr/bin/env python3
"""
Read REAL kernel resources (vgpr/agpr/sgpr/lds/spill) from built .o files.
gen_instances.py reports static heuristic estimates that are systematically
high (vgpr +25-60%, lds 2x for double-buffered pipelines).

Usage:
    python kid_real_resources.py [<build_dir>] [--filter SUBSTR]

Defaults build_dir to aiter/jit/build/module_deepgemm_opus/build.#!/usr/bin/env python3
"""
Read REAL kernel resources (vgpr/agpr/sgpr/lds/spill) from built .o files.
gen_instances.py reports static heuristic estimates that are systematically
high (vgpr +25-60%, lds 2x for double-buffered pipelines).

Usage:
    python kid_real_resources.py [<build_dir>] [--filter SUBSTR]

Defaults build_dir to aiter/jit/build/module_deepgemm_opus/build.
Requires .so to be built first.

Reports per kid:
    vgpr, agpr, sgpr, lds_bytes, vgpr_spill, sgpr_spill, private_seg
    + computed wg/CU under launch_bounds hint H using gfx942 rules:
        vgpr_cap = floor(4 SIMD * floor(512 / (vgpr+agpr)) / wave_per_wg)
        lds_cap  = floor(65536 / lds_per_wg)
        real_wg_cu = min(vgpr_cap, lds_cap, hint, 8)

Caveat: hint H is parsed from kernel mangled name not available; pass via
--hint to override, default 2 (matches current splitk_atomic).
"""
import argparse, os, re, subprocess, sys, tempfile

ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
LLVM_OBJCOPY = f"{ROCM}/llvm/bin/llvm-objcopy"
LLVM_READELF = f"{ROCM}/llvm/bin/llvm-readelf"
BUNDLER = f"{ROCM}/llvm/bin/clang-offload-bundler"

VGPR_POOL_PER_SIMD = 512   # gfx942 unified vgpr+agpr
SIMD_PER_CU = 4
LDS_PER_CU = 64 * 1024
MAX_WG_PER_CU = 8


def extract_notes(o_path):
    """Return dict of fields from .note in the gfx942 device slice of an .o."""
    with tempfile.TemporaryDirectory() as td:
        fatbin = f"{td}/k.fatbin"
        dev = f"{td}/k.dev"
        try:
            subprocess.run(
                [LLVM_OBJCOPY, f"--dump-section=.hip_fatbin={fatbin}", o_path],
                check=True, capture_output=True,
            )
            subprocess.run(
                [BUNDLER, "--type=o", f"--input={fatbin}", f"--output={dev}",
                 "--unbundle", "--targets=hipv4-amdgcn-amd-amdhsa--gfx942"],
                check=True, capture_output=True,
            )
            out = subprocess.run(
                [LLVM_READELF, "--notes", dev],
                check=True, capture_output=True, text=True,
            ).stdout
        except subprocess.CalledProcessError:
            return None
    fields = {}
    for line in out.splitlines():
        m = re.match(r"\s*-?\s*\.(\w+):\s*(\d+)", line)
        if m and m.group(1) in ("vgpr_count", "sgpr_count", "agpr_count",
                                 "group_segment_fixed_size",
                                 "private_segment_fixed_size",
                                 "vgpr_spill_count", "sgpr_spill_count"):
            fields[m.group(1)] = int(m.group(2))
    return fields


def parse_wave_per_wg(filename):
    """Parse BLOCK_SIZE from filename (e.g., '..._256x64x64x64...' or
    '..._512x128x128x64...'). Returns 4 for BLOCK=256, 8 for BLOCK=512."""
    m = re.search(r"_(\d+)x\d+x\d+x\d+_", filename)
    if not m:
        return None
    bs = int(m.group(1))
    return bs // 64  # warp size


def real_wg_per_cu(vgpr, agpr, lds, wave_per_wg, hint):
    vgpr_total = vgpr + agpr
    waves_per_simd = VGPR_POOL_PER_SIMD // max(vgpr_total, 1)
    waves_per_cu = waves_per_simd * SIMD_PER_CU
    vgpr_cap = waves_per_cu // max(wave_per_wg, 1)
    lds_cap = LDS_PER_CU // max(lds, 1) if lds else 999
    return min(vgpr_cap, lds_cap, hint, MAX_WG_PER_CU), vgpr_cap, lds_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", nargs="?",
                    default=os.path.expanduser(
                        "~/aiter/aiter/jit/build/module_deepgemm_opus/build"))
    ap.add_argument("--filter", default="")
    ap.add_argument("--hint", type=int, default=2,
                    help="launch_bounds min wg/CU (default 2)")
    args = ap.parse_args()

    if not os.path.isdir(args.build_dir):
        # try relative to repo
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        args.build_dir = f"{repo}/aiter/jit/build/module_deepgemm_opus/build"
    if not os.path.isdir(args.build_dir):
        print(f"ERR: build dir not found: {args.build_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in sorted(os.listdir(args.build_dir)):
        if not f.endswith(".device.cuda.o"):
            continue
        if args.filter and args.filter not in f:
            continue
        kid_name = f.replace(".device.cuda.o", "").replace("opus_gemm_", "")
        path = os.path.join(args.build_dir, f)
        n = extract_notes(path)
        if not n:
            continue
        wpw = parse_wave_per_wg(f) or 4
        real, vcap, lcap = real_wg_per_cu(
            n["vgpr_count"], n.get("agpr_count", 0),
            n["group_segment_fixed_size"], wpw, args.hint)
        rows.append((kid_name, n, wpw, real, vcap, lcap))

    if not rows:
        print("no built .o found; build first via `python _tmp_perf_50402_bld.py`")
        sys.exit(1)

    print(f"{'kid':<70} {'vgpr':>5} {'agpr':>5} {'sgpr':>5} {'lds_kb':>7} "
          f"{'wpw':>4} {'vcap':>5} {'lcap':>5} {'real_wg/cu':>10} {'spill':>6}")
    print("-" * 140)
    for kid_name, n, wpw, real, vcap, lcap in rows:
        spill = f"{n.get('vgpr_spill_count',0)}/{n.get('sgpr_spill_count',0)}"
        print(f"{kid_name:<70} {n['vgpr_count']:>5} {n.get('agpr_count',0):>5} "
              f"{n['sgpr_count']:>5} {n['group_segment_fixed_size']/1024:>7.2f} "
              f"{wpw:>4} {vcap:>5} {lcap:>5} {real:>10} {spill:>6}")
    print()
    print(f"hint={args.hint}; real_wg/cu = min(vcap, lcap, hint, {MAX_WG_PER_CU})")
    print("NOTE: gen_instances.py reports STATIC ESTIMATES (vgpr +25-60% high, "
          "lds 2x high for double-buf). Use this tool for real values.")


if __name__ == "__main__":
    main()
