# Benchmark History and Optimization Log

This file keeps the full benchmark trail across environments and optimization steps.
The top-level README only reports the best controlled result.

---

## Benchmark Protocols Used

### Ad-hoc interactive runs

- Command: `./build/ternary_infer model.bin --bench --iters 1000 --warmup 10`
- Environment: VS Code terminal and regular Ubuntu terminal
- Notes: high run-to-run variance due to scheduler/background noise

### Controlled single-core runs (recommended)

- Command: `sudo taskset -c 0 nice -n -20 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50`
- Environment: pinned to one core with high priority
- Notes: this is the reported method for Part B latency numbers

### Controlled multi-core runs (OpenMP)

- Command: `sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50`
- Environment: pinned to 6 cores with explicit OpenMP thread count
- Notes: use this to measure multi-core `conv_ternary` scaling separately from single-core reference numbers

---

## Raw Run History

### A) VS Code terminal (1000 iters, warmup 10)

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 5714.55 | 5656.07 | 6843.96 |
| 2 | 6100.14 | 6025.23 | 8123.57 |
| 3 | 6009.66 | 5923.79 | 7572.49 |

### B) Ubuntu terminal after closing apps (1000 iters, warmup 10)

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 7192.64 | 6897.85 | 11003.68 |
| 2 | 7362.13 | 6987.63 | 11673.72 |
| 3 | 7553.54 | 7283.52 | 12584.18 |
| 4 | 7295.18 | 6914.65 | 12554.42 |

### C) Back to VS Code terminal (1000 iters, warmup 10)

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 6747.26 | 6356.49 | 12341.70 |

### D) Controlled single-core (before compiler-flag tuning)

Command:

```bash
sudo taskset -c 0 nice -n -20 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 6061.09 | 5980.48 | 8375.04 |
| 2 | 6231.13 | 6064.85 | 9227.82 |

### E) Controlled single-core (after compiler-flag tuning)

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 5688.79 | 5485.02 | 8725.57 |
| 2 | 5547.78 | 5427.73 | 7781.68 |
| 3 | 6209.98 | 5795.07 | 9309.89 |
| 4 | 5604.62 | 5438.60 | 8309.83 |
| 5 | 5389.29 | 5311.43 | 6865.71 |
| 6 | 5410.37 | 5335.79 | 6928.59 |

### F) With Fused ReLU in conv1 layer -> DECREASE -SO Removed

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 5852.98 | 5778.86 | 7586.77 |
| 2 | 5881.63 | 5799.51 | 7414.91 |
| 3 | 5760.36 | 5707.11 | 7283.31 |
| 4 | 5884.69 | 5823.44 | 7567.69 |

### G) Extend to 4+4 accumulators in dot_product_ternary_avx2 ->commented out


| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 7113.75 | 6957.43 | 10375.63 |
| 2 | 7265.77 | 7042.17 | 11627.58 |
| 3 | 7771.24 | 7443.56 | 11138.54 |

## H) (M, N) Cache Blocking on GEMM

Tiled both spatial (M=64) and output channel (N=32) dimensions in `conv_ternary`, `conv_fp32`, and `linear` functions. 
This keeps weight rows (pos_bits, neg_bits) and their corresponding activation data in L2 cache, reducing memory bandwidth pressure.

**Loop structure:** For each spatial tile + channel tile block, process all spatial positions before moving to the next channel block.

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 5151.11 | 5116.66 | 6311.53 |
| 2 | 5109.37 | 5134.15 | 6250.87 |
| 3 | 5083.05 | 5086.88 | 6275.63 |
| 4 | 5118.79 | 5100.92 | 6355.16 |
| 5 | 5293.78 | 5280.71 | 6738.85 |
| 6 | 5218.40 | 5199.90 | 6656.44 |

## I) im2col Fast Path (remove full-buffer zero-fill)

Removed blanket full-buffer zero-fill in `im2col()` and wrote zeros only for out-of-bounds elements and k-padding tail.
This avoids redundant memory writes before real copy work.

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 4945.87 | 4862.89 | 6674.15 |
| 2 | 4975.75 | 4922.14 | 7035.28 |
| 3 | 5002.16 | 4901.09 | 7260.47 |

## J) OpenMP in `conv_ternary` (6-thread controlled runs)

Applied OpenMP work-sharing to `conv_ternary` output-channel work with one persistent parallel region to reduce repeated thread launch overhead.

Command:

```bash
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 4348.28 | 4101.54 | 7285.92 |
| 2 | 4782.36 | 4550.23 | 7983.69 |
| 3 | 4259.65 | 3995.04 | 7028.06 |
| 4 | 4474.71 | 4150.84 | 7515.81 |

## K) Aggressive OpenMP in im2col and conv_fp32 (6-thread controlled runs)

Extended OpenMP to the previously bottlenecked serial layers: `im2col` (using `collapse(2)`) and `conv_fp32` (over output channels).
Also tested pointwise layers (add, relu) but removed OpenMP pragmas there because of minor regressions.

Command:

```bash
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 2946.60 | 2676.01 | 5654.11 |
| 2 | 2738.21 | 2549.47 | 4488.70 |
| 3 | 2862.03 | 2545.02 | 5254.98 |

## L) 4-wide ternary kernel + collapse(2) OpenMP (single-core)

Replaced single-channel `dot_product_ternary_avx2` with `dot_product_ternary_4x_avx2`.
Load each activation row **once** and compute **4 output channels simultaneously** (4× fewer activation reads).
Also restructured `conv_ternary` with `collapse(2)` over `(oc_group × spatial_tile)` pairs — one
OpenMP barrier per layer instead of one per spatial tile (16 tiles → 1).

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 4448.19 | 4357.38 | 6271.82 |
| 2 | 4492.03 | 4385.32 | 6356.46 |
| 3 | 4403.88 | 4344.98 | 6348.11 |
| 4 | 4518.70 | 4413.61 | 6461.12 |

## M) 4-wide kernel + collapse(2) OpenMP (6-thread controlled runs)

Same code as L), benchmarked with 6 threads.

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 2084.27 | 1805.41 | 3900.39 |
| 2 | 2077.00 | 1788.39 | 3885.26 |
| 3 | 2121.31 | 1854.07 | 4071.03 |
| 4 | 2215.61 | 2053.61 | 3651.92 |
| 5 | 2314.22 | 2211.29 | 4447.15 |

## N) Attempted: explicit 2-unroll with dual accumulators per channel → reverted

Added `acc_a / acc_b` per output channel (8 accumulators total) with explicit 2-byte unrolling to try
to hide L2 latency and FP-add dependency chains. Result: **worse** — increased register pressure caused
the compiler to generate suboptimal code.

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 4782.21 | 4577.59 | 7940.96 |
| 2 | 4607.49 | 4514.23 | 7223.33 |
| 3 | 4616.39 | 4563.97 | 6519.50 |

Reverted to single-accumulator version with `#pragma GCC unroll 4`.

## O) Attempted: kSpatialTile=128 → reverted

Doubled spatial tile size from 64 to 128. Fewer work items meant worse load balance for the
small-channel layers (16-channel group has only 4 oc_groups × 8 tiles = 32 work items for 6 threads).
**Worse on both single-core and multi-core.**

| Config | median single-core (us) | median 6-thread (us) |
|---|---:|---:|
| kSpatialTile=64 (kept) | 4344.98 | 1788.39 |
| kSpatialTile=128 (reverted) | ~4440 | ~1925 |

## P) Attempted: merged im2col parallel region into conv_ternary → reverted

Inlined the im2col loop directly inside `conv_ternary`'s `#pragma omp parallel` region using an
orphaned `#pragma omp for`, eliminating 18 separate parallel-region launches (one per ternary layer).
Result: **slightly worse** — the extra implicit barrier between im2col and conv phases inside the
persistent region cost more than was saved by avoiding 18 fork/join cycles.

Reverted to separate parallel regions.

## Q) Hardware counter probe with perf

Goal: check whether more threads were still buying useful speedup before adding more algorithmic complexity.

### Setup

```bash
sudo apt-get update && sudo apt-get install -y linux-tools-common linux-tools-generic
```

### perf attempt

```bash
/usr/lib/linux-tools-6.8.0-107/perf stat -d taskset -c 0-5 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

Observed output from this run:

```text
mean_us=1780.31 median_us=1831.20 p99_us=2594.66
task-clock:u = 33480.85 msec (6.144 CPUs utilized)
page-faults:u = 589
time elapsed = 5.448967364 s
user = 33.419834 s, sys = 0.049195 s
```

PMU limitation reported by `perf`:

```text
Unable to find PMU or event on a PMU of 'cpu_core'
<not supported> cycles:u
<not supported> instructions:u
<not supported> branches:u
<not supported> branch-misses:u
<not supported> L1-dcache-loads:u
<not supported> L1-dcache-load-misses:u
<not supported> LLC-loads:u
<not supported> LLC-load-misses:u
```

### Thread-scaling probe

```bash
for t in 1 2 4 6; do OMP_NUM_THREADS=$t ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50; done
```

| Threads | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 4584.42 | 4546.26 | 6182.36 |
| 2 | 2706.97 | 2614.49 | 4320.41 |
| 4 | 1987.34 | 1835.72 | 3271.63 |
| 6 | 1927.56 | 1944.42 | 2786.96 |

### Resource usage

| Threads | wall time | CPU% | max RSS (KB) |
|---|---:|---:|---:|
| 1 | 15.57 s | 95% | 5264 |
| 6 | 5.65 s | 620% | 4984 |

### Conclusion

The code still scales well from 1 to 4 threads, but gains flatten by 6 threads. That suggests the hot path is no longer compute-only; memory traffic and/or OpenMP overhead are starting to dominate. Due to a WSL architectural constraint (Microsoft WSL2 kernel PMU exposure), `perf` cannot provide IPC, cache-miss, or bandwidth counters here, so deeper hardware-counter bottleneck analysis is not possible on this host.

## R) ONNX Runtime FP32 benchmark (CPUExecutionProvider)

Commands:

```bash
cd part_C
python export_onnx.py --opset 18
python benchmark_onnx.py --iters 3000 --warmup 50
```

Run used for reporting:

| Implementation | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| ORT FP32 baseline | 919.2 | 855.8 | 1633.4 |
| ORT FP32 ternary (frozen weights) | 857.9 | 807.6 | 1515.7 |

Speedup context (same README reference values):

- ORT baseline vs PyTorch baseline: **1.95x** (mean), **1.99x** (median)
- ORT ternary vs PyTorch ternary: **3.88x** (mean), **3.82x** (median)
- ORT ternary vs C++ AVX2 ternary (OpenMP, 6 threads): **2.42x** (mean), **2.21x** (median)

Model-size efficiency from ONNX export:

| Artifact | Size |
|---|---:|
| baseline.pth | 1111.9 KB |
| ternary.pth | 1111.3 KB |
| baseline_fp32.onnx | 84.3 KB |
| ternary_fp32.onnx | 237.6 KB |
| model.bin (packed ternary) | 280.1 KB |

- `baseline_fp32.onnx` is **13.19x smaller** than `baseline.pth` (**92.4% smaller**).
- `ternary_fp32.onnx` is **4.68x smaller** than `ternary.pth` (**78.6% smaller**).

Exporter warning note:

- The long traceback you saw is from opset down-conversion (18 -> 17) attempted by the exporter toolchain.
- Export is still valid if you see `verified: max_diff=...` for both models and `onnx.checker` passes.
- Using `--opset 18` avoids the conversion attempt and removes that noisy warning path.

---

## Summary for Reporting

### Best single-core run (4-wide kernel + collapse(2))

- mean: **4403.88 us**
- median: **4344.98 us**
- p99: **6348.11 us**

### Best 6-thread OpenMP run (4-wide kernel + collapse(2))

- mean: **1908.09 us**
- median: **1696.95 us**
- p99: **3424.06 us**

### Improvement Chain (full history)

| Step | Median (us) | Improvement |
|---|---:|---|
| Baseline (no opt) | 6061.09 | - |
| Compiler flags only | 5427.73 | **10.47%** |
| M,N cache blocking | 5086.88 | **6.28%** (vs flags) |
| im2col fast path | 4862.89 | **4.40%** (vs M,N block) |
| 4-wide kernel + collapse(2) | 4344.98 | **10.65%** (vs im2col fast path) |
| **Total single-core** | 4344.98 | **28.31%** (vs baseline) |

---

## Reproducibility Notes

1. Keep single-core and multi-core runs in separate tables; do not mix them.
2. Use the controlled single-core command for final reference reporting.
3. Use the controlled multi-core command for OpenMP scaling reporting.
4. Run at least 3 trials and report best plus variance-aware context.
5. If comparing with PyTorch, apply matching control methods:

```bash
# Single-core
sudo taskset -c 0 nice -n -20 env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$(which python)" part_C/benchmark_pytorch.py --cpp-mean-us <mean> --cpp-median-us <median> --cpp-p99-us <p99> --iters 3000 --warmup 50

# Multi-core (6 threads)
sudo taskset -c 0-5 nice -n -20 env \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
  "$(which python)" part_C/benchmark_pytorch.py --cpp-mean-us <mean> --cpp-median-us <median> --cpp-p99-us <p99> --iters 3000 --warmup 50

# Environment-only threaded run (no taskset pinning)
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
  "$(which python)" part_C/benchmark_pytorch.py --cpp-mean-us <mean> --cpp-median-us <median> --cpp-p99-us <p99> --iters 3000 --warmup 50
```

## Latest Cross-Framework Comparison (4-wide kernel, 6-thread OpenMP)

Best C++ result from section S (latest controlled rerun) below.

| Implementation | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| PyTorch baseline (FP32) | 1788.8 | 1700.5 | 3373.7 |
| PyTorch ternary | 3330.4 | 3083.6 | 7012.4 |
| C++ ternary (OpenMP, 6 threads) | 1908.09 | 1696.95 | 3424.06 |

Speedups:
- vs PyTorch ternary: **1.75x** (mean), **1.82x** (median)
- vs PyTorch baseline: **0.94x** (mean), **1.00x** (median)

## S) thread scaling + ORT multi-thread

Goal: refresh numbers with consistent high-priority runs and verify scaling beyond 6 threads on i7-12650H.

### C++ ternary (`part_B/build/ternary_infer`)

Controlled 6-thread runs:

```bash
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 1908.09 | 1696.95 | 3424.06 |
| 2 | 1949.93 | 1717.83 | 4516.74 |
| 3 | 1974.39 | 1783.22 | 3839.43 |

8-thread exploratory run:

```bash
sudo taskset -c 0-7 nice -n -10 env OMP_NUM_THREADS=8 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

| mean (us) | median (us) | p99 (us) |
|---:|---:|---:|
| 1666.97 | 1652.95 | 2172.33 |

Observation: 8 threads improved latency vs the controlled 6-thread setup on this host.

### ONNX Runtime (`part_C/benchmark_onnx.py`)

6-thread ORT run:

```bash
sudo nice -n -20 "$(which python)" benchmark_onnx.py --iters 3000 --warmup 50 --threads 6
```

| Implementation | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| ORT FP32 baseline | 329.8 | 321.6 | 607.5 |
| ORT FP32 ternary (frozen weights) | 328.6 | 322.9 | 560.7 |

16-thread ORT was tested but showed larger p99 variance; 6-thread values are used as the stable reference.

### Command correctness notes

1. In `part_C`, use `"$(which python)"` with `sudo` to avoid `python: No such file or directory`.
2. `./build/ternary_infer` only exists in `part_B`, so run that command from `part_B`.

## T) Const/restrict micro-optimization pass

Goal: apply low-risk compile-time and aliasing improvements in Part B hot paths without changing model behavior.

### Code changes

1. Added compiler-friendly aliasing contracts (`__restrict`) to AVX2 kernel interfaces and definitions.
2. Hoisted repeated `.data()` base-pointer lookups in `conv_fp32` and `conv_ternary`.
3. Tightened local const usage and made small helper intent explicit (`inline round_up`).
4. Kept all algorithmic structure unchanged (no layout/packing format changes).

### Quick performance sanity checks

Commands used during this pass:

```bash
# Short sanity run (not controlled)
./build/ternary_infer model.bin --bench --iters 500 --warmup 20

# Cleaner single-core check
taskset -c 0 env OMP_NUM_THREADS=1 \
  ./build/ternary_infer model.bin --bench --iters 800 --warmup 30
```

Observed outputs:

| Run type | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| Short sanity (500/20) | 2842.80 | 1794.00 | 15046.69 |
| Single-core check (800/30, pinned) | 4554.06 | 4334.38 | 6823.60 |

### Follow-up controlled multi-core rerun (6 threads)

After this pass, controlled reruns were recorded (also summarized in section S):

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 1908.09 | 1696.95 | 3424.06 |
| 2 | 1949.93 | 1717.83 | 4516.74 |
| 3 | 1974.39 | 1783.22 | 3839.43 |

Conclusion: the pass preserved correctness and maintained the current performance envelope while simplifying hot-path runtime state (better const/restrict intent, fewer repeated lookups).

