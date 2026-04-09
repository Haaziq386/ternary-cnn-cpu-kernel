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

### G) Extend to 4+4 accumulators in dot_product_ternary_avx2


| Run | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| 1 | 7113.75 | 6957.43 | 10375.63 |
| 2 | 7265.77 | 7042.17 | 11627.58 |
| 3 | 7771.24 | 7443.56 | 11138.54 |



---

## Summary for Reporting

### Best controlled run (post-optimization)

- mean: **5547.78 us**
- median: **5427.73 us**
- p99: **7781.68 us**

### Pre-vs-post controlled comparison (best-to-best)

- median improvement: 5980.48 us -> 5427.73 us (**9.24% faster**)
- mean improvement: 6061.09 us -> 5547.78 us (**8.47% faster**)

---

## F) Latest Controlled Cross-Framework Comparison

Source: `part_C/results.json` generated via pinned single-core command.

Command used:

```bash
sudo taskset -c 0 nice -n -20 env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$(which python)" part_C/benchmark_pytorch.py \
  --cpp-mean-us 5547.78 --cpp-median-us 5427.73 --cpp-p99-us 7781.68 \
  --iters 3000 --warmup 50
```

Latency results:

| Implementation | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| PyTorch baseline (FP32) | 1620.58 | 1535.87 | 3070.63 |
| PyTorch ternary | 2252.14 | 2132.05 | 3584.30 |
| C++ ternary | 5547.78 | 5427.73 | 7781.68 |

Speedups (C++ vs PyTorch):

- vs PyTorch ternary: 0.41x (mean), 0.39x (median)
- vs PyTorch baseline: 0.29x (mean), 0.28x (median)

Memory outputs from `part_C/results.json`:

- baseline.pth: 1111.9 KB
- ternary.pth: 1111.3 KB
- model.bin: 280.1 KB
- Python peak RSS during benchmark: 3087.4 MB

Note: C++ inference peak RSS can be measured with:

```bash
/usr/bin/time -v ./build/ternary_infer model.bin --bench
```

This prints the process memory high-water mark (`Maximum resident set size`) for the C++ binary.

---

## Reproducibility Notes

1. Use the controlled single-core command for any final number.
2. Run at least 3 trials and report best plus variance-aware context.
3. If comparing with PyTorch, apply the same control method:

```bash
sudo taskset -c 0 nice -n -20 env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$(which python)" part_C/benchmark_pytorch.py --cpp-mean-us <mean> --cpp-median-us <median> --cpp-p99-us <p99> --iters 3000 --warmup 50
```