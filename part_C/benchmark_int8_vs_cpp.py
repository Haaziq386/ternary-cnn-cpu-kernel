#!/usr/bin/env python3
"""Compare Part B C++ ternary inference vs ONNX Runtime dynamic INT8.

This script avoids retraining and focuses on deploy-time changes:
1) Ensure FP32 ONNX exports exist (optionally generate them).
2) Quantize FP32 ONNX models to dynamic INT8.
3) Benchmark C++ ternary and ORT INT8 under matched CPU policies:
   - multi-core pinned (e.g. cores 0-5)
   - single-core pinned (e.g. core 0)
4) Print a terminal table and save a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


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


def file_size_kb(path: Path) -> float:
    return round(path.stat().st_size / 1024, 1)


def parse_core_spec(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid CPU range '{token}' in '{spec}'")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    if not cpus:
        raise ValueError(f"No CPUs parsed from --multi-cores='{spec}'")
    return cpus


@contextmanager
def pinned_affinity(cpus: set[int]):
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        yield
        return

    prev = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, prev)


def make_session(path: Path, num_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


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


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed (exit={}):\n{}\nOutput:\n{}".format(
                proc.returncode,
                " ".join(cmd),
                proc.stdout,
            )
        )
    return proc.stdout


def parse_cpp_bench_metrics(output: str) -> dict[str, float]:
    m = re.search(r"mean_us=([\d.]+)\s+median_us=([\d.]+)\s+p99_us=([\d.]+)", output)
    if not m:
        raise ValueError(
            "Could not parse C++ benchmark line. Expected 'mean_us=... median_us=... p99_us=...'.\n"
            f"Full output:\n{output}"
        )
    return {
        "mean_us": round(float(m.group(1)), 2),
        "median_us": round(float(m.group(2)), 2),
        "p99_us": round(float(m.group(3)), 2),
    }


def maybe_export_fp32(args: argparse.Namespace) -> None:
    if args.skip_export:
        return
    if args.baseline_fp32_onnx.exists() and args.ternary_fp32_onnx.exists():
        return

    export_script = Path(__file__).parent / "export_onnx.py"
    cmd = [
        sys.executable,
        str(export_script),
        "--opset",
        str(args.opset),
        "--baseline-pth",
        str(args.baseline_pth),
        "--pth",
        str(args.ternary_pth),
        "--npz",
        str(args.npz),
        "--out-dir",
        str(args.baseline_fp32_onnx.parent),
    ]
    print("[1/5] FP32 ONNX not found. Exporting with export_onnx.py ...")
    run_command(cmd)


def quantize_to_dynamic_int8(fp32_model: Path, int8_model: Path) -> None:
    int8_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_dynamic(
            model_input=str(fp32_model),
            model_output=str(int8_model),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
        )
        return
    except Exception as err:
        # Some exported graphs carry stale/contradictory value_info entries that break
        # ORT quantizer's shape inference pass. Retry with stripped value_info.
        print(f"  quantize_dynamic retry for {fp32_model.name}: {err}")

    sanitized = int8_model.parent / f"{fp32_model.stem}.sanitized.onnx"
    model = onnx.load(str(fp32_model))
    del model.graph.value_info[:]
    onnx.save(model, str(sanitized))
    try:
        quantize_dynamic(
            model_input=str(sanitized),
            model_output=str(int8_model),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
        )
    finally:
        if sanitized.exists():
            sanitized.unlink()


def build_cpp_cmd(
    args: argparse.Namespace,
    core_spec: str,
    omp_threads: int,
) -> list[str]:
    cmd: list[str] = []
    if args.use_sudo:
        cmd.append("sudo")
    # Set OMP_NUM_THREADS in the executed command itself so it survives sudo env filtering.
    cmd += ["env", f"OMP_NUM_THREADS={omp_threads}"]
    if args.nice is not None:
        cmd += ["nice", "-n", str(args.nice)]
    cmd += [
        "taskset",
        "-c",
        core_spec,
        str(args.cpp_bin),
        str(args.model_bin),
        "--bench",
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]
    return cmd


def speedup(a_us: float, b_us: float) -> float:
    return round(a_us / b_us, 2)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_onnx_dir = Path(__file__).parent / "onnx_models"

    parser = argparse.ArgumentParser(description="Benchmark C++ ternary vs ONNX dynamic INT8")
    parser.add_argument("--npz", type=Path, default=repo_root / "part_A" / "ternary_weights.npz")
    parser.add_argument("--baseline-pth", type=Path, default=repo_root / "part_A" / "baseline.pth")
    parser.add_argument("--ternary-pth", type=Path, default=repo_root / "part_A" / "ternary.pth")
    parser.add_argument("--cpp-bin", type=Path, default=repo_root / "part_B" / "build" / "ternary_infer")
    parser.add_argument("--model-bin", type=Path, default=repo_root / "part_B" / "model.bin")

    parser.add_argument("--baseline-fp32-onnx", type=Path, default=default_onnx_dir / "baseline_fp32.onnx")
    parser.add_argument("--ternary-fp32-onnx", type=Path, default=default_onnx_dir / "ternary_fp32.onnx")
    parser.add_argument("--baseline-int8-onnx", type=Path, default=default_onnx_dir / "baseline_dynamic_int8.onnx")
    parser.add_argument("--ternary-int8-onnx", type=Path, default=default_onnx_dir / "ternary_dynamic_int8.onnx")

    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--multi-cores", type=str, default="0-5")
    parser.add_argument("--single-core", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--opset", type=int, default=18)

    parser.add_argument("--skip-export", action="store_true", help="Do not auto-export FP32 ONNX files")
    parser.add_argument("--use-sudo", action="store_true", help="Prefix C++ runs with sudo")
    parser.add_argument("--nice", type=int, default=None, help="Optional nice level for C++ runs, e.g. -20")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results_int8_vs_cpp.json",
    )
    args = parser.parse_args()

    if args.use_sudo and hasattr(os, "geteuid") and os.geteuid() == 0:
        print("[warn] Script is already running as root; --use-sudo is redundant.")

    for required in [args.npz, args.cpp_bin, args.model_bin, args.baseline_pth, args.ternary_pth]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    print("[1/5] Preparing ONNX models ...")
    maybe_export_fp32(args)
    if not args.baseline_fp32_onnx.exists() or not args.ternary_fp32_onnx.exists():
        raise FileNotFoundError(
            "FP32 ONNX models are missing. Run export_onnx.py first or remove --skip-export."
        )

    print("[2/5] Quantizing FP32 ONNX -> dynamic INT8 ...")
    quantize_to_dynamic_int8(args.baseline_fp32_onnx, args.baseline_int8_onnx)
    quantize_to_dynamic_int8(args.ternary_fp32_onnx, args.ternary_int8_onnx)

    npz = np.load(args.npz, allow_pickle=True)
    dummy = np.asarray(npz["sample_input"], dtype=np.float32)[:1]

    multi_affinity = parse_core_spec(args.multi_cores)
    single_affinity = {int(args.single_core)}

    print("[3/5] Running C++ benchmarks (multi-core + single-core) ...")
    cpp_multi_cmd = build_cpp_cmd(args, args.multi_cores, args.threads)
    cpp_multi_out = run_command(cpp_multi_cmd)
    cpp_multi = parse_cpp_bench_metrics(cpp_multi_out)

    cpp_single_cmd = build_cpp_cmd(args, str(args.single_core), 1)
    cpp_single_out = run_command(cpp_single_cmd)
    cpp_single = parse_cpp_bench_metrics(cpp_single_out)

    print("[4/5] Running ONNX dynamic INT8 benchmarks (multi-core + single-core) ...")
    with pinned_affinity(multi_affinity):
        ort_multi_baseline = summarise(
            time_session(make_session(args.baseline_int8_onnx, args.threads), dummy, args.warmup, args.iters)
        )
        ort_multi_ternary = summarise(
            time_session(make_session(args.ternary_int8_onnx, args.threads), dummy, args.warmup, args.iters)
        )

    with pinned_affinity(single_affinity):
        ort_single_baseline = summarise(
            time_session(make_session(args.baseline_int8_onnx, 1), dummy, args.warmup, args.iters)
        )
        ort_single_ternary = summarise(
            time_session(make_session(args.ternary_int8_onnx, 1), dummy, args.warmup, args.iters)
        )

    print("[5/5] Reporting ...")
    sep = "=" * 88
    print(f"\n{sep}")
    print("C++ TERNARY VS ONNX DYNAMIC INT8 (lower is better, microseconds)")
    print(sep)
    print(f"{'Policy':<16} {'Impl':<30} {'mean_us':>10} {'median_us':>10} {'p99_us':>10}")
    print("-" * 88)
    print(f"{'multi-core':<16} {'C++ ternary':<30} {cpp_multi['mean_us']:>10.2f} {cpp_multi['median_us']:>10.2f} {cpp_multi['p99_us']:>10.2f}")
    print(f"{'multi-core':<16} {'ORT baseline dynamic INT8':<30} {ort_multi_baseline['mean_us']:>10.2f} {ort_multi_baseline['median_us']:>10.2f} {ort_multi_baseline['p99_us']:>10.2f}")
    print(f"{'multi-core':<16} {'ORT ternary dynamic INT8':<30} {ort_multi_ternary['mean_us']:>10.2f} {ort_multi_ternary['median_us']:>10.2f} {ort_multi_ternary['p99_us']:>10.2f}")
    print(f"{'single-core':<16} {'C++ ternary':<30} {cpp_single['mean_us']:>10.2f} {cpp_single['median_us']:>10.2f} {cpp_single['p99_us']:>10.2f}")
    print(f"{'single-core':<16} {'ORT baseline dynamic INT8':<30} {ort_single_baseline['mean_us']:>10.2f} {ort_single_baseline['median_us']:>10.2f} {ort_single_baseline['p99_us']:>10.2f}")
    print(f"{'single-core':<16} {'ORT ternary dynamic INT8':<30} {ort_single_ternary['mean_us']:>10.2f} {ort_single_ternary['median_us']:>10.2f} {ort_single_ternary['p99_us']:>10.2f}")

    multi_speedup_vs_ort_ternary = {
        "mean_x": speedup(ort_multi_ternary["mean_us"], cpp_multi["mean_us"]),
        "median_x": speedup(ort_multi_ternary["median_us"], cpp_multi["median_us"]),
        "p99_x": speedup(ort_multi_ternary["p99_us"], cpp_multi["p99_us"]),
    }
    single_speedup_vs_ort_ternary = {
        "mean_x": speedup(ort_single_ternary["mean_us"], cpp_single["mean_us"]),
        "median_x": speedup(ort_single_ternary["median_us"], cpp_single["median_us"]),
        "p99_x": speedup(ort_single_ternary["p99_us"], cpp_single["p99_us"]),
    }

    print("\nSpeedup (ORT ternary INT8 latency / C++ ternary latency):")
    print(
        "  multi-core  mean={}x  median={}x  p99={}x".format(
            multi_speedup_vs_ort_ternary["mean_x"],
            multi_speedup_vs_ort_ternary["median_x"],
            multi_speedup_vs_ort_ternary["p99_x"],
        )
    )
    print(
        "  single-core mean={}x  median={}x  p99={}x".format(
            single_speedup_vs_ort_ternary["mean_x"],
            single_speedup_vs_ort_ternary["median_x"],
            single_speedup_vs_ort_ternary["p99_x"],
        )
    )

    model_sizes = {
        "model_bin_kb": file_size_kb(args.model_bin),
        "baseline_fp32_onnx_kb": file_size_kb(args.baseline_fp32_onnx),
        "ternary_fp32_onnx_kb": file_size_kb(args.ternary_fp32_onnx),
        "baseline_dynamic_int8_onnx_kb": file_size_kb(args.baseline_int8_onnx),
        "ternary_dynamic_int8_onnx_kb": file_size_kb(args.ternary_int8_onnx),
    }

    results = {
        "latency_us": {
            "multi_core": {
                "cpp_ternary": cpp_multi,
                "ort_baseline_dynamic_int8": ort_multi_baseline,
                "ort_ternary_dynamic_int8": ort_multi_ternary,
            },
            "single_core": {
                "cpp_ternary": cpp_single,
                "ort_baseline_dynamic_int8": ort_single_baseline,
                "ort_ternary_dynamic_int8": ort_single_ternary,
            },
        },
        "speedup": {
            "multi_core_ort_ternary_int8_vs_cpp_ternary": multi_speedup_vs_ort_ternary,
            "single_core_ort_ternary_int8_vs_cpp_ternary": single_speedup_vs_ort_ternary,
        },
        "model_sizes_kb": model_sizes,
        "config": {
            "iters": args.iters,
            "warmup": args.warmup,
            "multi_cores": args.multi_cores,
            "single_core": args.single_core,
            "threads": args.threads,
            "quantization": "dynamic_int8",
            "op_types_to_quantize": ["Conv", "MatMul", "Gemm"],
            "cpp_use_sudo": args.use_sudo,
            "cpp_nice": args.nice,
        },
        "raw_outputs": {
            "cpp_multi": cpp_multi_out,
            "cpp_single": cpp_single_out,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON report: {args.out}")


if __name__ == "__main__":
    main()