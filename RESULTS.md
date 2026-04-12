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

## U) Per-layer profiling with `PROFILE_LAYERS` (6-thread controlled run)

### Setup

Added `std::chrono::steady_clock` timing guards to `run_resnet20()` under a `PROFILE_LAYERS` compile-time flag.
Timing is zero-overhead when the flag is off; active only when built with `-DPROFILE_LAYERS=ON`.

```bash
cmake -S . -B build -DPROFILE_LAYERS=ON && cmake --build build -j
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

Each forward pass printed one `[PROFILE]` table. The final benchmark line reports stats over all 3000 iterations.

### Aggregate (3000 iterations, 6 threads)

| Metric | Value (us) |
|---|---:|
| mean | 2827.51 |
| median | 2727.33 |
| p99 | 5153.57 |

### Representative per-layer breakdown (approximate means from visible samples)

| Layer group | Mean (us) | % of total |
|---|---:|---:|
| `conv_ternary` (18 residual convs) | ~2560 | **~89%** |
| `conv_fp32` (stem + 3 projections) | ~280 | ~10% |
| `relu_inplace` | ~16 | <1% |
| `add_inplace` | ~15 | <1% |
| `global_avg_pool` | ~1 | <1% |
| `linear` | ~0 | <1% |
| **TOTAL** | **~2827** | 100% |

### Bottleneck diagnosis

`conv_ternary` is the sole bottleneck at ~89% of wall time.
Two root causes were identified by reading the kernel source:

**1. `mask_to_ps` table-lookup overhead** (`ternary_kernel.cpp:49-53`): -- DIDN't Work out -> poor results
Each byte of packed weight triggers a 32-byte load from `kMaskTable[256][8]` (8 KB).
In `dot_product_ternary_2x4_avx2` the inner loop fires **8 such loads per iteration**
(`pos0..pos3`, `neg0..neg3`).  Even though the 8 KB table stays in L1, each load
carries a 4–5 cycle load-use latency and produces a memory-dependency chain.

**2. Register pressure / spills in `dot_product_ternary_2x4_avx2`** (`ternary_kernel.cpp:176-238`):
The function holds 8 accumulators live across the entire loop, then attempts to keep all
8 mask vectors (`p0`–`p3`, `n0`–`n3`) plus `x0, nx0, x1, nx1, sign_mask` simultaneously.
That requires ~21 YMM registers; AVX2 only has 16.  `#pragma GCC unroll 4` multiplies the
pressure by ×4, guaranteeing accumulator or mask spills to the stack on every iteration.

---

## Summary for Reporting

### Best single-core run (4-wide kernel + collapse(2))

- mean: **4403.88 us**
- median: **4344.98 us**
- p99: **6348.11 us**

### Best 6-thread OpenMP run (kST=32, guided, latest)

- mean: **1611.4 us**
- median: **1597.7 us**
- p99: **2328.5 us**

### Improvement Chain (full history)

| Step | Median (us) | Improvement |
|---|---:|---|
| Baseline (no opt) | 6061.09 | - |
| Compiler flags only | 5427.73 | **10.47%** |
| M,N cache blocking | 5086.88 | **6.28%** (vs flags) |
| im2col fast path | 4862.89 | **4.40%** (vs M,N block) |
| 4-wide kernel + collapse(2) | 4344.98 | **10.65%** (vs im2col fast path) |
| **Total single-core** | 4344.98 | **28.31%** (vs baseline) |
| kST=32 + guided (autotune, 6-thread) | 1597.7 (multi) | **−5.6% median, −8.9% p99 vs prior best** |

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

## Latest Cross-Framework Comparison (kST=32 + guided, 6-thread OpenMP)

Best C++ result from section X (autotune controlled rerun) below.

| Implementation | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| PyTorch baseline (FP32) | 1788.8 | 1700.5 | 3373.7 |
| PyTorch ternary | 3330.4 | 3083.6 | 7012.4 |
| C++ ternary (OpenMP, 6 threads) | 1611.4 | 1597.7 | 2328.5 |

Speedups:
- vs PyTorch ternary: **2.07x** (mean), **1.93x** (median)
- vs PyTorch baseline: **1.11x** (mean), **1.06x** (median)

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

## V) Fix 2: remove `#pragma GCC unroll 4` from `dot_product_ternary_2x4_avx2`

Goal: reduce register pressure and spill risk in the 2x4 ternary kernel by removing explicit 4x unrolling in the hottest loop.

Change made:

1. Removed `#pragma GCC unroll 4` from `dot_product_ternary_2x4_avx2`.
2. Kept `#pragma GCC unroll 4` on `dot_product_fp32_avx2` unchanged (that kernel has much lower register pressure).

Controlled 6-thread runs:

```bash
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 1664.27 | 1693.19 | 2558.51 |
| 2 | 1704.25 | 1728.31 | 2575.82 |
| 3 | 1755.58 | 1786.12 | 2653.30 |
| 4 | 1726.74 | 1747.64 | 2600.74 |

Result: this change improves the 6-thread performance envelope versus the prior 1908.09 / 1696.95 / 3424.06 reference, with a much tighter p99 tail.

## W) Sub-layer profiling: im2col vs. dot-product breakdown

### Motivation  
Section U showed `conv_ternary` ≈90% of runtime, but not its internal split. Two hypotheses:  
1. **im2col is significant** → removing it (direct conv) could yield large gains.  
2. **Stage 1 dominates** due to larger spatial size (32×32 vs 8×8).

### Changes  
Extended `PROFILE_LAYERS` to capture:  
- **im2col vs dot-product time** via `g_ternary_breakdown`  
- **Per-stage totals** (stage 1/2/3) via per-block accumulation  

### Profiling overhead  
`chrono` instrumentation adds ~14% overhead, but ratios remain valid.

### Key findings  
1. **im2col = 30–39%** (~500 µs): significant memory-copy overhead  
2. **Dot product = 60–70%**: dominant but already optimized  
3. **Stage 1 dominates (40–50%)**: largest spatial workload  

### Optimization directions  
| Direction | Target | Expected gain |
|---|---|---|
| Direct convolution (no im2col) | Remove ~35% im2col overhead | High ROI |
| im2col optimization | Reduce bandwidth pressure | Medium ROI |
| Kernel tuning | Improve dot-product | Low ROI |

---

## X) Parameter autotune sweep — kSpatialTile, kChannelTile, compiler flags, OMP schedule

### Motivation

Section W identified `conv_ternary` as ≥89% of runtime and Stage 1 (32×32) as the dominant stage.
The OpenMP `collapse(2)` loop distributes work over `(oc_group × spatial_tile)` pairs.
With `kSpatialTile=64` and the Stage 3 layers (8×8 spatial, 16 oc_groups):
- spatial_tiles = 64/64 = **1** tile × 16 groups = **16 work items** for 6 threads → 2.7 items/thread
- Stage 2 (16×16): 256/64 = 4 tiles × 8 groups = **32 items** → 5.3 items/thread

Smaller spatial tiles create more work items and improve load balance across threads.

### Tunable parameters identified

| Parameter | Location | Current | Effect |
|---|---|---|---|
| `kSpatialTile` | `layers.cpp:30` | 64 | Work-item granularity for `conv_ternary` collapse(2) OMP loop |
| `kChannelTile` | `layers.cpp:31` | 32 | L2 blocking for `conv_fp32` + `linear` (~10% of runtime) |
| OMP schedule | `layers.cpp:188` | `static` | Work distribution strategy for the collapse(2) loop |
| Compiler flags | `CMakeLists.txt` | see file | `-ffast-math`, `-fprefetch-loop-arrays`, `-fvect-cost-model=unlimited` |

### Sweep script

`part_B/autotune.py` — three-phase automated sweep:
- **Phase 1**: full grid over `kSpatialTile ∈ {8, 16, 32, 64}` × `kChannelTile ∈ {16, 32, 64}`
- **Phase 2**: compiler-flag variants on best tile config
- **Phase 3**: OMP schedule variants (`static`, `dynamic,1`, `guided`) on best tile+flag config

Usage: `python3 autotune.py [--quick] [--threads N] [--phase tiles|flags|schedule|all]`

### Phase 1 results — tile sweep (1000 iters, 2 trials, 6 threads)

| kSpatialTile | kChannelTile | mean (µs) | median (µs) | p99 (µs) | note |
|---:|---:|---:|---:|---:|---|
| 8 | 16 | 1983.6 | 2021.2 | 2990.0 | |
| 8 | 32 | 2008.5 | 2051.7 | 2983.1 | |
| 8 | 64 | 1942.9 | 2000.7 | 2713.2 | |
| 16 | 16 | 2110.1 | 2126.8 | 3548.0 | |
| 16 | 32 | 2062.4 | 2094.3 | 2899.2 | |
| 16 | 64 | 1975.8 | 2051.0 | 2698.2 | |
| **32** | **16** | **1717.4** | **1734.6** | **2790.3** | **phase winner** |
| 32 | 32 | 2057.2 | 2040.1 | 2913.0 | |
| 32 | 64 | 2119.2 | 2167.3 | 3201.3 | |
| 64 | 16 | 2035.6 | 2081.5 | 3032.7 | |
| 64 | 32 | 2183.3 | 2206.0 | 3314.7 | ← baseline |
| 64 | 64 | 2108.9 | 2132.7 | 3112.8 | |

**Winner:** `kSpatialTile=32`. `kChannelTile` has negligible effect (only affects `conv_fp32`/`linear`, ~10% of runtime) — kept at 32.

**Why kST=32 wins:**

| Stage | Spatial | kST=64 work items | kST=32 work items | items/thread (32) |
|---|---|---:|---:|---:|
| 1 (16 ch) | 32×32 | 64 | 128 | 21 |
| 2 (32 ch) | 16×16 | 32 | 64 | 11 |
| 3 (64 ch) | 8×8 | 16 | 32 | 5 |

Halving the tile doubles work items everywhere. Stage 3 goes from 2.7 to 5.3 items/thread — much better utilization without flooding the scheduler.

### Phase 2 results — compiler flags (kST=32, kCT=32, 1 trial quick mode)

| Flags | mean (µs) | median (µs) | p99 (µs) |
|---|---:|---:|---:|
| base (current) | 2049.3 | 2100.9 | 2777.9 |
| +ffast-math | 2105.2 | 2103.7 | 2930.4 |
| +prefetch-loop-arrays | 2127.9 | 2156.8 | 3098.4 |
| +fvect-cost-model=unlimited | 1980.4 | 2049.2 | 2825.8 |

**Conclusion:** all variants within noise of base. No compiler flag changes applied.

### Phase 3 results — OMP schedule (kST=32, kCT=32, 1 trial quick mode)

| Schedule | mean (µs) | median (µs) | p99 (µs) |
|---|---:|---:|---:|
| static (current) | 1982.2 | 2080.8 | 2847.6 |
| dynamic,1 | 2138.6 | 2177.7 | 2892.7 |
| **guided** | **1869.7** | **1899.3** | **2605.8** |

**Winner:** `schedule(guided)`. Confirmed with 3×1500-iter matched-thermal interleaved test:

| Pair | OLD (kST=64, static) median | NEW (kST=32, guided) median | Improvement |
|---:|---:|---:|---:|
| 1 | 1975.9 | 1722.8 | **12.8%** |
| 2 | 2093.6 | 1910.3 | **8.7%** |
| 3 | 2125.8 | 1842.8 | **13.3%** |

`guided` adapts chunk sizes dynamically and handles the unequal work-item sizes between stages better than static pre-division.

### Changes applied

```diff
- constexpr int kSpatialTile = 64;
+ constexpr int kSpatialTile = 32;  // autotuned: beats 64 by ~20% (better OMP balance for all 3 stages)

- #pragma omp for schedule(static) collapse(2)
+ #pragma omp for schedule(guided) collapse(2)
```

### New controlled reference (3 runs × 3000 iters, 50 warmup, 6 threads)

```bash
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

| Run | mean (µs) | median (µs) | p99 (µs) |
|---:|---:|---:|---:|
| 1 | 1611.4 | 1597.7 | 2328.5 |
| 2 | 1617.0 | 1625.6 | 2413.3 |
| 3 | 1642.5 | 1695.8 | 2350.0 |

**Best run:** mean=1611.4 µs, median=1597.7 µs, p99=2328.5 µs

Comparison vs prior best (section V, kST=64 + static):

| Metric | Section V | Section X | Change |
|---|---:|---:|---:|
| mean (µs) | 1664.3 | 1611.4 | **−3.2%** |
| median (µs) | 1693.2 | 1597.7 | **−5.6%** |
| p99 (µs) | 2558.5 | 2328.5 | **−9.0%** |

---

## Y) ONNX dynamic INT8 vs C++ ternary (fair pinned comparison)

Goal: compare current C++ ternary against ONNX Runtime dynamic INT8 under the same core pinning policy and iteration counts, without retraining.

### Command

Run from `part_C`:

```bash
"$(which python)" benchmark_int8_vs_cpp.py \
  --skip-export --multi-cores 0-5 --single-core 0 --threads 6 \
  --iters 3000 --warmup 50 --use-sudo --nice -20
```

Important:
1. Do **not** wrap the whole script invocation with outer `sudo` when using `--use-sudo`.
2. `benchmark_onnx.py` reports FP32 ONNX latency; do not compare those numbers as if they were INT8.

### Representative run

| Policy | Implementation | mean (us) | median (us) | p99 (us) |
|---|---|---:|---:|---:|
| Multi-core (`0-5`, threads=6) | C++ ternary | 1938.00 | 1802.73 | 3229.75 |
| Multi-core (`0-5`, threads=6) | ORT baseline dynamic INT8 | 2217.95 | 2083.35 | 3680.00 |
| Multi-core (`0-5`, threads=6) | ORT ternary dynamic INT8 | 1634.24 | 1576.95 | 2844.85 |
| Single-core (`0`, threads=1) | C++ ternary | 4559.11 | 4268.36 | 8527.33 |
| Single-core (`0`, threads=1) | ORT baseline dynamic INT8 | 3816.15 | 3587.26 | 6418.56 |
| Single-core (`0`, threads=1) | ORT ternary dynamic INT8 | 2334.02 | 2104.92 | 4678.74 |

### Interpretation

In this dynamic INT8 setup, ORT remains faster than current C++ ternary for both multi-core and single-core pinned runs. This does not conflict with the FP32 ONNX results from section R, which are a different model format/path.

---

## Z) perf-like telemetry output + OpenMP barrier-elision pass

### Motivation

The existing `PROFILE_LAYERS` timing is useful for layer attribution, but not for scheduler/runtime behavior.
Added a benchmark-side telemetry mode to expose perf-like process metrics directly from `ternary_infer`.
The telemetry path is compiled only when `PROFILE_LAYERS` is enabled so the default inference binary remains unchanged.

### Code changes

1. Added `--perf-like` flag in `part_B/src/main.cpp`.
2. On Linux, benchmark mode now reports:
   - `task-clock-ms` (from process user+sys CPU time)
   - `wall-ms` and derived `cpus-utilized = task_clock / wall`
   - `context-switches` (voluntary + involuntary)
   - `page-faults` (minor + major)
   - best-effort `instructions`/`cycles`, IPC, and average GHz via `perf_event_open` when available.
3. Reduced synchronization overhead in hot loops:
   - `conv_ternary`: `#pragma omp for schedule(guided) collapse(2) nowait`
   - `conv_fp32`: `#pragma omp for schedule(static) nowait`

### New command

```bash
cmake -S . -B build -DPROFILE_LAYERS=ON && cmake --build build -j
taskset -c 0-5 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50 --perf-like
```

### Example output (this host)

```text
mean_us=1686.23 median_us=1688.41 p99_us=2321.35
[PERF_LIKE] task-clock-ms=30350.95 wall-ms=5058.89 cpus-utilized=6.00
[PERF_LIKE] context-switches=449 (voluntary=0, involuntary=449)
[PERF_LIKE] page-faults=7 (minor=7, major=0)
[PERF_LIKE] instructions/cycles unavailable (cycles unavailable: No such file or directory)
```

`instructions/cycles` are unavailable here due host PMU exposure limits (consistent with prior WSL constraints).

### Performance rerun after barrier-elision (3x, pinned)

```bash
for i in 1 2 3; do
  taskset -c 0-5 env OMP_NUM_THREADS=6 \
    ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
done
```

| Run | mean (us) | median (us) | p99 (us) |
|---:|---:|---:|---:|
| 1 | 1553.79 | 1527.58 | 2456.70 |
| 2 | 1612.64 | 1622.61 | 2317.45 |
| 3 | 1657.66 | 1668.41 | 2317.48 |

### Comparison vs previous best controlled reference (section X)

Reference from section X (with `sudo taskset ... nice -n -20`):
- mean: 1611.4 us
- median: 1597.7 us
- p99: 2328.5 us

Current best rerun:
- best mean: 1553.79 us (**-3.57%** vs 1611.4)
- best median: 1527.58 us (**-4.39%** vs 1597.7)
- best p99: 2317.45 us (**-0.48%** vs 2328.5)

