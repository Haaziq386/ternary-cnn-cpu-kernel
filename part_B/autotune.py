#!/usr/bin/env python3
"""
autotune.py — parameter sweep for part_B on i7-12650H.

Sweeps three dimensions in order:
  Phase 1  kSpatialTile × kChannelTile   (compile-time constants in layers.cpp)
  Phase 2  compiler-flag variants         (on best tile config)
  Phase 3  OMP schedule variants          (on best tile + best flag config)

Usage (from part_B/ directory):
  python3 autotune.py              # full sweep, 1000 iters per trial
  python3 autotune.py --quick      # 400 iters (faster, ~4–6 min total)
  python3 autotune.py --phase tiles
  python3 autotune.py --threads 8
  sudo python3 autotune.py         # higher-priority bench (lower p99 noise)

Results are printed as a table and saved to autotune_results.json.
The script always restores original source files on exit (even on Ctrl-C).
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
PART_B     = Path(__file__).parent.resolve()
LAYERS_CPP = PART_B / "src" / "layers.cpp"
CMAKE_FILE = PART_B / "CMakeLists.txt"
BINARY     = PART_B / "build" / "ternary_infer"
MODEL_BIN  = PART_B / "model.bin"

# ── search space ──────────────────────────────────────────────────────────────
# kSpatialTile: smaller → more work items (better OMP balance for Stage 3's 8×8)
#               larger  → fewer barriers but worse tail utilisation
# kChannelTile: used only in conv_fp32 and linear (~10% of runtime)
SPATIAL_TILES = [8, 16, 32, 64]   # current default: 64
CHANNEL_TILES = [16, 32, 64]      # current default: 32

# Extra compiler flags to test on top of the existing CMakeLists.txt set.
# "base" = no extra flags (current configuration).
COMPILER_VARIANTS: dict[str, list[str]] = {
    "base":                    [],
    "+ffast-math":             ["-ffast-math"],
    "+prefetch-loop-arrays":   ["-fprefetch-loop-arrays"],
    "+fvect-cost-model":       ["-fvect-cost-model=unlimited"],
}

# OMP schedule in the collapse(2) loop inside conv_ternary.
# "static" is the current default.
OMP_SCHEDULES: dict[str, str] = {
    "static":   "schedule(static)",
    "dynamic1": "schedule(dynamic,1)",
    "guided":   "schedule(guided)",
}

BASELINE = {"kSpatialTile": 64, "kChannelTile": 32}

# ── helpers ───────────────────────────────────────────────────────────────────

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def build(jobs: int = 4) -> bool:
    r = subprocess.run(
        ["cmake", "--build", "build", f"-j{jobs}"],
        cwd=PART_B, capture_output=True, text=True,
    )
    if r.returncode != 0:
        snippet = (r.stderr or r.stdout)[-600:]
        print(f"    !! BUILD FAILED:\n{snippet}")
    return r.returncode == 0

def bench_once(threads: int, iters: int, warmup: int) -> dict | None:
    """Run one benchmark trial; return {mean, median, p99} or None on failure."""
    cmd = (
        f"taskset -c 0-{threads - 1} "
        f"env OMP_NUM_THREADS={threads} "
        f"{BINARY} {MODEL_BIN} "
        f"--bench --iters {iters} --warmup {warmup}"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    !! BENCH FAILED: {r.stderr[:200]}")
        return None
    m = re.search(
        r"mean_us=([\d.]+)\s+median_us=([\d.]+)\s+p99_us=([\d.]+)", r.stdout
    )
    if not m:
        print(f"    !! PARSE FAILED: {r.stdout[:200]}")
        return None
    return {
        "mean":   float(m.group(1)),
        "median": float(m.group(2)),
        "p99":    float(m.group(3)),
    }

def bench_best(threads: int, iters: int, warmup: int, trials: int) -> dict | None:
    """Run `trials` bench trials, return the one with the lowest median."""
    results = []
    for _ in range(trials):
        r = bench_once(threads, iters, warmup)
        if r:
            results.append(r)
    if not results:
        return None
    return min(results, key=lambda x: x["median"])

# ── patch helpers ─────────────────────────────────────────────────────────────

def patch_tiles(src: str, spatial: int, channel: int) -> str:
    src = re.sub(
        r"constexpr int kSpatialTile = \d+;",
        f"constexpr int kSpatialTile = {spatial};",
        src,
    )
    src = re.sub(
        r"constexpr int kChannelTile = \d+;[^\n]*",
        f"constexpr int kChannelTile = {channel}; // Tile output channels for L2 cache blocking",
        src,
    )
    return src

def patch_omp_schedule(src: str, schedule_str: str) -> str:
    """Replace the schedule clause on the collapse(2) for-loop in conv_ternary."""
    return re.sub(
        r"schedule\([\w,]+\)\s+collapse\(2\)",
        f"{schedule_str} collapse(2)",
        src,
    )

def patch_cmake_add_flags(cmake_src: str, extra_flags: list[str]) -> str:
    """Insert extra_flags into target_compile_options, after -ffp-contract=fast."""
    if not extra_flags:
        return cmake_src
    addition = "".join(f"\n        {f}" for f in extra_flags)
    marker = "        -ffp-contract=fast\n    )"
    if marker not in cmake_src:
        print("    !! WARNING: could not locate flag insertion point in CMakeLists.txt")
        return cmake_src
    return cmake_src.replace(marker, f"        -ffp-contract=fast{addition}\n    )")

def fmt(m: dict) -> str:
    return f"mean={m['mean']:8.1f}  median={m['median']:8.1f}  p99={m['p99']:8.1f}"

# ── per-config runner ─────────────────────────────────────────────────────────

def run_config(
    label: str,
    layers_src: str,
    cmake_src: str,
    threads: int,
    iters: int,
    warmup: int,
    trials: int,
) -> dict | None:
    _write(LAYERS_CPP, layers_src)
    _write(CMAKE_FILE, cmake_src)
    if not build():
        return None
    return bench_best(threads, iters, warmup, trials)

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick",   action="store_true",
                    help="400 iters / 15 warmup / 1 trial per config (≈4-6 min)")
    ap.add_argument("--threads", type=int, default=6,
                    help="OMP_NUM_THREADS / taskset core count (default 6)")
    ap.add_argument("--phase",   default="all",
                    choices=["tiles", "flags", "schedule", "all"],
                    help="Which phase(s) to run")
    args = ap.parse_args()

    iters  = 400  if args.quick else 1000
    warmup = 15   if args.quick else 30
    trials = 1    if args.quick else 2
    threads = args.threads

    print(f"\nautotune.py  [{datetime.now():%Y-%m-%d %H:%M}]")
    print(f"  iters={iters}  warmup={warmup}  trials={trials}  threads={threads}")
    print(f"  binary: {BINARY}\n")

    orig_layers = _read(LAYERS_CPP)
    orig_cmake  = _read(CMAKE_FILE)

    all_results: list[dict] = []
    best_tile   = dict(BASELINE)   # updated after phase 1
    best_flags: list[str] = []     # updated after phase 2

    HEADER = f"{'Config':<42}  {'mean':>8}  {'median':>8}  {'p99':>8}"
    DIV    = "─" * 72

    try:
        # ── Phase 1: kSpatialTile × kChannelTile ─────────────────────────────
        if args.phase in ("tiles", "all"):
            print("═" * 72)
            print("Phase 1 — kSpatialTile × kChannelTile sweep")
            print("  kChannelTile matters mostly for conv_fp32/linear (~10% of runtime).")
            print("  kSpatialTile controls OMP work-item granularity in conv_ternary.")
            print(HEADER)
            print(DIV)

            tile_results: list[dict] = []

            for spatial in SPATIAL_TILES:
                for channel in CHANNEL_TILES:
                    label = f"kST={spatial:3d}  kCT={channel:3d}"
                    mark  = " ← baseline" if spatial == BASELINE["kSpatialTile"] and channel == BASELINE["kChannelTile"] else ""
                    layers_patched = patch_tiles(orig_layers, spatial, channel)
                    m = run_config(label, layers_patched, orig_cmake,
                                   threads, iters, warmup, trials)
                    if m:
                        print(f"  {label}{mark:<15}  {fmt(m)}")
                        tile_results.append({
                            "phase": "tiles",
                            "label": label,
                            "kSpatialTile": spatial,
                            "kChannelTile": channel,
                            **m,
                        })
                    else:
                        print(f"  {label:<42}  FAILED")

            if tile_results:
                best_tr = min(tile_results, key=lambda x: x["median"])
                best_tile = {"kSpatialTile": best_tr["kSpatialTile"],
                             "kChannelTile": best_tr["kChannelTile"]}
                print(DIV)
                print(f"  ★ Best tile:  kST={best_tr['kSpatialTile']}  kCT={best_tr['kChannelTile']}  {fmt(best_tr)}")
            all_results.extend(tile_results)

        # ── Phase 2: compiler-flag variants on best tile ──────────────────────
        if args.phase in ("flags", "all"):
            st = best_tile["kSpatialTile"]
            ct = best_tile["kChannelTile"]
            print()
            print("═" * 72)
            print(f"Phase 2 — compiler-flag variants  (kST={st}  kCT={ct})")
            print(HEADER)
            print(DIV)

            layers_best = patch_tiles(orig_layers, st, ct)
            flag_results: list[dict] = []

            for flag_name, extra_flags in COMPILER_VARIANTS.items():
                cmake_patched = patch_cmake_add_flags(orig_cmake, extra_flags)
                mark = " ← current" if flag_name == "base" else ""
                m = run_config(flag_name, layers_best, cmake_patched,
                               threads, iters, warmup, trials)
                if m:
                    print(f"  {flag_name + mark:<42}  {fmt(m)}")
                    flag_results.append({
                        "phase": "flags",
                        "label": flag_name,
                        "extra_flags": extra_flags,
                        **m,
                    })
                else:
                    print(f"  {flag_name:<42}  FAILED")

            if flag_results:
                best_fr = min(flag_results, key=lambda x: x["median"])
                best_flags = best_fr["extra_flags"]
                print(DIV)
                print(f"  ★ Best flags: '{best_fr['label']}'  {fmt(best_fr)}")
            all_results.extend(flag_results)

        # ── Phase 3: OMP schedule variants ───────────────────────────────────
        if args.phase in ("schedule", "all"):
            st = best_tile["kSpatialTile"]
            ct = best_tile["kChannelTile"]
            print()
            print("═" * 72)
            print(f"Phase 3 — OMP schedule variants  (kST={st}  kCT={ct}  flags={best_flags or 'base'})")
            print(HEADER)
            print(DIV)

            layers_best_tiles = patch_tiles(orig_layers, st, ct)
            cmake_best_flags  = patch_cmake_add_flags(orig_cmake, best_flags)
            sched_results: list[dict] = []

            for sched_name, sched_str in OMP_SCHEDULES.items():
                layers_src = patch_omp_schedule(layers_best_tiles, sched_str)
                mark = " ← current" if sched_name == "static" else ""
                m = run_config(sched_name, layers_src, cmake_best_flags,
                               threads, iters, warmup, trials)
                if m:
                    print(f"  {sched_name + mark:<42}  {fmt(m)}")
                    sched_results.append({
                        "phase": "schedule",
                        "label": sched_name,
                        "schedule_str": sched_str,
                        **m,
                    })
                else:
                    print(f"  {sched_name:<42}  FAILED")

            if sched_results:
                best_sr = min(sched_results, key=lambda x: x["median"])
                print(DIV)
                print(f"  ★ Best schedule: '{best_sr['label']}'  {fmt(best_sr)}")
            all_results.extend(sched_results)

    finally:
        # Always restore originals and rebuild so the binary stays consistent.
        print("\n  Restoring original source files and rebuilding …")
        _write(LAYERS_CPP, orig_layers)
        _write(CMAKE_FILE, orig_cmake)
        build()

    # ── full summary ─────────────────────────────────────────────────────────
    if all_results:
        print()
        print("═" * 72)
        print("Summary — all configs sorted by median (µs)")
        print(f"{'Phase':<10} {'Label':<35}  {'mean':>8}  {'median':>8}  {'p99':>8}")
        print("─" * 72)
        for r in sorted(all_results, key=lambda x: x["median"]):
            print(f"  {r['phase']:<8} {r['label']:<35}  {fmt(r)}")

        # Save JSON
        out = PART_B / "autotune_results.json"
        out.write_text(
            json.dumps(
                {"timestamp": datetime.now().isoformat(),
                 "settings": {"iters": iters, "warmup": warmup, "trials": trials,
                              "threads": threads},
                 "results": all_results},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  Full results saved → {out}")
        print("  Paste the table above into RESULTS.md as section X.")
    else:
        print("  No results collected.")


if __name__ == "__main__":
    main()
