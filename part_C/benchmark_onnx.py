#!/usr/bin/env python3
"""Benchmark ONNX Runtime (CPUExecutionProvider) vs existing results."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


# --- helpers (mirror of benchmark_pytorch.py) ---
def percentile(data: list[float], p: float) -> float:
    idx = int(p / 100.0 * (len(data) - 1))
    return sorted(data)[idx]


def summarise(timings: list[float]) -> dict[str, float]:
    s = sorted(timings)
    return {
        "mean_us": round(sum(s) / len(s), 2),
        "median_us": round(s[len(s) // 2], 2),
        "p99_us": round(percentile(s, 99), 2),
        "min_us": round(s[0], 2),
        "max_us": round(s[-1], 2),
    }


def file_size_kb(path: str | Path) -> float:
    return round(os.path.getsize(path) / 1024, 1)


# --- ORT helpers ---
def make_session(path: Path, num_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def time_session(
    session: ort.InferenceSession,
    dummy_np: np.ndarray,
    warmup: int,
    iters: int,
) -> list[float]:
    name = session.get_inputs()[0].name
    feed = {name: dummy_np}

    for _ in range(warmup):
        session.run(None, feed)

    timings: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        timings.append((time.perf_counter() - t0) * 1e6)
    return timings


# --- main ---
def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    onnx_dir = Path(__file__).parent / "onnx_models"

    parser.add_argument("--baseline-onnx", type=Path, default=onnx_dir / "baseline_fp32.onnx")
    parser.add_argument("--ternary-onnx", type=Path, default=onnx_dir / "ternary_fp32.onnx")
    parser.add_argument("--npz", type=Path, default=repo_root / "part_A" / "ternary_weights.npz")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results_onnx.json")
    args = parser.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    dummy = np.asarray(npz["sample_input"], dtype=np.float32)[:1]

    print(
        f"Benchmarking ORT baseline ({args.warmup} warmup, {args.iters} iters, "
        f"{args.threads} thread(s))..."
    )
    ort_baseline = summarise(
        time_session(
            make_session(args.baseline_onnx, args.threads),
            dummy,
            args.warmup,
            args.iters,
        )
    )

    print("Benchmarking ORT ternary ...")
    ort_ternary = summarise(
        time_session(
            make_session(args.ternary_onnx, args.threads),
            dummy,
            args.warmup,
            args.iters,
        )
    )

    sep = "=" * 60
    print(f"\n{sep}\nONNX RUNTIME BENCHMARK RESULTS\n{sep}")
    print(f"{'Impl':<40} {'mean_us':>10} {'median_us':>10} {'p99_us':>10}")
    print("-" * 72)
    print(
        f"{'ORT FP32 baseline':<40} {ort_baseline['mean_us']:>10.1f} "
        f"{ort_baseline['median_us']:>10.1f} {ort_baseline['p99_us']:>10.1f}"
    )
    print(
        f"{'ORT FP32 ternary (frozen weights)':<40} {ort_ternary['mean_us']:>10.1f} "
        f"{ort_ternary['median_us']:>10.1f} {ort_ternary['p99_us']:>10.1f}"
    )

    results = {
        "latency": {
            "ort_baseline_fp32": ort_baseline,
            "ort_ternary_fp32": ort_ternary,
        },
        "model_sizes": {
            "baseline_onnx_kb": file_size_kb(args.baseline_onnx),
            "ternary_onnx_kb": file_size_kb(args.ternary_onnx),
        },
        "config": {
            "iters": args.iters,
            "warmup": args.warmup,
            "intra_op_num_threads": args.threads,
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "execution_mode": "ORT_SEQUENTIAL",
            "provider": "CPUExecutionProvider",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
