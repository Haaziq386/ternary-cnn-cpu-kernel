#!/usr/bin/env python3
"""
Part C: Benchmarking — PyTorch vs C++ ternary inference.

Usage:
    python benchmark_pytorch.py \
        --npz  ../part_A/ternary_weights.npz \
        --pth  ../part_A/ternary.pth \
        --baseline-pth ../part_A/baseline.pth \
        --model-bin ../part_B/model.bin \
        --cpp-mean-us 5714.55 \
        --cpp-median-us 5656.07 \
        --cpp-p99-us 6843.96 \
        --iters 1000 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
PART_A = REPO_ROOT / "part_A"
if str(PART_A) not in sys.path:
    sys.path.insert(0, str(PART_A))

from resnet20 import ResNet20          # noqa: E402
from ternary_layer import TernaryConv2d  # noqa: E402


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_inference(model: nn.Module, sample: torch.Tensor, warmup: int, iters: int) -> list[float]:
    """Return per-iteration latency in microseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)

        timings: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(sample)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1e6)  # convert to microseconds
    return timings


def percentile(data: list[float], p: float) -> float:
    idx = int(p / 100.0 * (len(data) - 1))
    return sorted(data)[idx]


def summarise(timings: list[float]) -> dict[str, float]:
    s = sorted(timings)
    return {
        "mean_us":   round(sum(s) / len(s), 2),
        "median_us": round(s[len(s) // 2], 2),
        "p99_us":    round(percentile(s, 99), 2),
        "min_us":    round(s[0], 2),
        "max_us":    round(s[-1], 2),
    }


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def file_size_kb(path: str | Path) -> float:
    return round(os.path.getsize(path) / 1024, 1)


def peak_rss_mb() -> float:
    """Read peak resident set size from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return -1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Part C benchmark: PyTorch vs C++")
    parser.add_argument("--npz",          type=Path, default=PART_A / "ternary_weights.npz")
    parser.add_argument("--pth",          type=Path, default=PART_A / "ternary.pth")
    parser.add_argument("--baseline-pth", type=Path, default=PART_A / "baseline.pth")
    parser.add_argument("--model-bin",    type=Path, default=REPO_ROOT / "part_B" / "model.bin")
    parser.add_argument("--cpp-mean-us",   type=float, required=True)
    parser.add_argument("--cpp-median-us", type=float, required=True)
    parser.add_argument("--cpp-p99-us",    type=float, required=True)
    parser.add_argument("--iters",  type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load sample input
    # -----------------------------------------------------------------------
    print("Loading data...")
    npz = np.load(args.npz, allow_pickle=True)
    sample_np = np.asarray(npz["sample_input"], dtype=np.float32)[:1]   # single image
    sample = torch.from_numpy(sample_np)

    # -----------------------------------------------------------------------
    # 2. Benchmark ternary PyTorch model
    # -----------------------------------------------------------------------
    print(f"Loading ternary model from {args.pth} ...")
    try:
        ternary_state = torch.load(args.pth, map_location="cpu", weights_only=True)
    except TypeError:
        ternary_state = torch.load(args.pth, map_location="cpu")

    ternary_model = ResNet20(conv_cls=TernaryConv2d)
    ternary_model.load_state_dict(ternary_state)

    print(f"Benchmarking ternary PyTorch ({args.warmup} warmup, {args.iters} iters)...")
    ternary_timings = time_inference(ternary_model, sample, args.warmup, args.iters)
    py_ternary = summarise(ternary_timings)

    # -----------------------------------------------------------------------
    # 3. Benchmark baseline full-precision PyTorch model (if available)
    # -----------------------------------------------------------------------
    py_baseline = None
    if args.baseline_pth.exists():
        print(f"Loading baseline model from {args.baseline_pth} ...")
        try:
            baseline_state = torch.load(args.baseline_pth, map_location="cpu", weights_only=True)
        except TypeError:
            baseline_state = torch.load(args.baseline_pth, map_location="cpu")

        baseline_model = ResNet20(conv_cls=nn.Conv2d)
        baseline_model.load_state_dict(baseline_state)

        print(f"Benchmarking baseline PyTorch ({args.warmup} warmup, {args.iters} iters)...")
        baseline_timings = time_inference(baseline_model, sample, args.warmup, args.iters)
        py_baseline = summarise(baseline_timings)

    # -----------------------------------------------------------------------
    # 4. C++ numbers (from --bench run)
    # -----------------------------------------------------------------------
    cpp_ternary = {
        "mean_us":   round(args.cpp_mean_us, 2),
        "median_us": round(args.cpp_median_us, 2),
        "p99_us":    round(args.cpp_p99_us, 2),
    }

    # -----------------------------------------------------------------------
    # 5. Speedup
    # -----------------------------------------------------------------------
    speedup_vs_py_ternary_mean   = round(py_ternary["mean_us"]   / cpp_ternary["mean_us"],   2)
    speedup_vs_py_ternary_median = round(py_ternary["median_us"] / cpp_ternary["median_us"], 2)

    speedup_vs_py_baseline_mean   = None
    speedup_vs_py_baseline_median = None
    if py_baseline:
        speedup_vs_py_baseline_mean   = round(py_baseline["mean_us"]   / cpp_ternary["mean_us"],   2)
        speedup_vs_py_baseline_median = round(py_baseline["median_us"] / cpp_ternary["median_us"], 2)

    # -----------------------------------------------------------------------
    # 6. Memory analysis
    # -----------------------------------------------------------------------
    peak_rss = peak_rss_mb()
    memory = {
        "baseline_pth_kb":  file_size_kb(args.baseline_pth) if args.baseline_pth.exists() else None,
        "ternary_pth_kb":   file_size_kb(args.pth) if args.pth.exists() else None,
        "model_bin_kb":     file_size_kb(args.model_bin) if args.model_bin.exists() else None,
        "python_peak_rss_mb": peak_rss,
    }

    # Compression ratios
    if memory["ternary_pth_kb"] and memory["model_bin_kb"]:
        memory["bin_vs_ternary_pth_ratio"] = round(memory["model_bin_kb"] / memory["ternary_pth_kb"], 3)
    if memory["baseline_pth_kb"] and memory["model_bin_kb"]:
        memory["bin_vs_baseline_pth_ratio"] = round(memory["model_bin_kb"] / memory["baseline_pth_kb"], 3)

    # -----------------------------------------------------------------------
    # 7. Print report
    # -----------------------------------------------------------------------
    sep = "=" * 60

    print(f"\n{sep}")
    print("PART C BENCHMARK RESULTS")
    print(sep)

    print("\n--- Latency (single image, CPU, single-threaded) ---")
    print(f"{'Impl':<35} {'mean_us':>10} {'median_us':>10} {'p99_us':>10}")
    print("-" * 68)
    if py_baseline:
        print(f"{'PyTorch baseline (FP32)':<35} {py_baseline['mean_us']:>10.1f} {py_baseline['median_us']:>10.1f} {py_baseline['p99_us']:>10.1f}")
    print(f"{'PyTorch ternary (TernaryConv2d)':<35} {py_ternary['mean_us']:>10.1f} {py_ternary['median_us']:>10.1f} {py_ternary['p99_us']:>10.1f}")
    print(f"{'C++ ternary (AVX2 kernel)':<35} {cpp_ternary['mean_us']:>10.1f} {cpp_ternary['median_us']:>10.1f} {cpp_ternary['p99_us']:>10.1f}")

    print(f"\n--- Speedup (C++ ternary vs PyTorch) ---")
    print(f"  vs PyTorch ternary  — mean: {speedup_vs_py_ternary_mean}×  median: {speedup_vs_py_ternary_median}×")
    if py_baseline:
        print(f"  vs PyTorch baseline — mean: {speedup_vs_py_baseline_mean}×  median: {speedup_vs_py_baseline_median}×")

    print(f"\n--- Memory / Model Size ---")
    if memory["baseline_pth_kb"]:
        print(f"  baseline.pth (FP32 weights):  {memory['baseline_pth_kb']:>8.1f} KB")
    if memory["ternary_pth_kb"]:
        print(f"  ternary.pth  (FP32 storage):  {memory['ternary_pth_kb']:>8.1f} KB")
    if memory["model_bin_kb"]:
        print(f"  model.bin    (packed ternary): {memory['model_bin_kb']:>8.1f} KB")
    if memory.get("bin_vs_ternary_pth_ratio"):
        print(f"  model.bin is {memory['bin_vs_ternary_pth_ratio']:.2f}× the size of ternary.pth  "
              f"({(1 - memory['bin_vs_ternary_pth_ratio']) * 100:.1f}% smaller)")
    if memory.get("bin_vs_baseline_pth_ratio"):
        print(f"  model.bin is {memory['bin_vs_baseline_pth_ratio']:.2f}× the size of baseline.pth "
              f"({(1 - memory['bin_vs_baseline_pth_ratio']) * 100:.1f}% smaller)")
    if peak_rss > 0:
        print(f"  Python process peak RSS: {peak_rss} MB")
    print(f"\n  Note: C++ inference peak RSS can be measured with:")
    print(f"        /usr/bin/time -v ./build/ternary_infer model.bin --bench")

    print(f"\n{sep}")

    # -----------------------------------------------------------------------
    # 8. Save results.json
    # -----------------------------------------------------------------------
    results = {
        "latency": {
            "pytorch_ternary":   py_ternary,
            "pytorch_baseline":  py_baseline,
            "cpp_ternary":       cpp_ternary,
        },
        "speedup": {
            "cpp_vs_pytorch_ternary_mean":    speedup_vs_py_ternary_mean,
            "cpp_vs_pytorch_ternary_median":  speedup_vs_py_ternary_median,
            "cpp_vs_pytorch_baseline_mean":   speedup_vs_py_baseline_mean,
            "cpp_vs_pytorch_baseline_median": speedup_vs_py_baseline_median,
        },
        "memory": memory,
        "config": {
            "iters": args.iters,
            "warmup": args.warmup,
            "device": "cpu",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
