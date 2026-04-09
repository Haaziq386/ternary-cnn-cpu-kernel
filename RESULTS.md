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

---

## Summary for Reporting

### Best controlled run (with M,N cache blocking + im2col fast path)

- mean: **4945.87 us**
- median: **4862.89 us**
- p99: **6674.15 us**

### Best controlled OpenMP run (6 threads)

- mean: **4259.65 us**
- median: **3995.04 us**
- p99: **7028.06 us**
- vs current single-core reference (5106.85 us median): **21.78% lower median latency**

### Improvement Chain (no opt → compiler flags → M,N blocking → im2col fast path)

| Step | Median (us) | Improvement |
|---|---:|---|
| Baseline (no opt) | 6061.09 | - |
| Compiler flags only | 5427.73 | **10.47%** |
| M,N cache blocking | 5086.88 | **6.28%** (vs flags) |
| im2col fast path | 4862.89 | **4.40%** (vs M,N block) |
| **Total** | 4862.89 | **19.77%** (vs baseline) |

---

## Latest Cross-Framework Comparison (OpenMP run)

Source: `part_C/results.json` generated from the latest command below.

Command used:

```bash
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
  "$(which python)" part_C/benchmark_pytorch.py \
  --cpp-mean-us 4259.65 --cpp-median-us 3995.04 --cpp-p99-us 7028.06 \
  --iters 3000 --warmup 50
```

Latency results:

| Implementation | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| PyTorch baseline (FP32) | 1596.9 | 1480.2 | 3711.5 |
| PyTorch ternary | 2838.4 | 2589.6 | 5699.9 |
| C++ ternary (OpenMP, 6 threads) | 4259.65 | 3995.04 | 7028.06 |

Speedups (C++ vs PyTorch):

- vs PyTorch ternary: 0.67x (mean), 0.65x (median)
- vs PyTorch baseline: 0.37x (mean), 0.37x (median)

Memory outputs from `part_C/results.json`:

- baseline.pth: 1111.9 KB
- ternary.pth: 1111.3 KB
- model.bin: 280.1 KB
- Python peak RSS during benchmark: 3649.1 MB

Note: C++ inference peak RSS can be measured with:

```bash
/usr/bin/time -v ./build/ternary_infer model.bin --bench
```

This prints the process memory high-water mark (`Maximum resident set size`) for the C++ binary.

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
