# Ternary CNN CPU Kernel

Implementation and benchmarking of a ternary-weight ResNet-20 on CIFAR-10, with a hand-written AVX2 CPU inference kernel.

---

## Repository Structure

```
part_A/   — PyTorch training (ResNet-20, TernaryConv2d, weight export)
part_B/   — C++ inference engine (AVX2 SIMD ternary kernel)
part_C/   — Benchmarking script and results
Question/ — Assignment brief
```

---

## End-to-End Workflow

```bash
# Part A — train the ternary model
cd part_A
python train.py --model ternary --epochs 30 --batch-size 128 --data-root ./data

# Part B — convert weights and build the C++ engine
cd ../part_B
cmake -S . -B build && cmake --build build -j
python convert_weights.py ../part_A/ternary_weights.npz ../part_A/ternary.pth model.bin

# Validate: C++ output must match PyTorch reference
./build/ternary_infer model.bin --validate

# Benchmark: single-image latency
./build/ternary_infer model.bin --bench --iters 1000 --warmup 10

# Part B benchmark (controlled single-core)
sudo taskset -c 0 nice -n -20 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50

# Part B benchmark (OpenMP multi-core, 6 threads)
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50

# Part C — benchmark PyTorch and compare (latest C++ OpenMP best example)
cd ../part_C
python benchmark_pytorch.py \
  --cpp-mean-us 1611.4 --cpp-median-us 1597.7 --cpp-p99-us 2328.5 \
  --iters 3000 --warmup 50

# Part C — export ONNX models and benchmark ONNX Runtime (FP32)
python export_onnx.py --opset 18
sudo nice -n -20 "$(which python)" benchmark_onnx.py --threads 6 --iters 3000 --warmup 50

# Part C — compare C++ ternary vs ONNX dynamic INT8 (single + multi-core in one run)
"$(which python)" benchmark_int8_vs_cpp.py \
  --multi-cores 0-5 --single-core 0 --threads 6 \
  --iters 3000 --warmup 50 --use-sudo --nice -20

# If FP32 ONNX models already exist and you want to skip export:
"$(which python)" benchmark_int8_vs_cpp.py \
  --skip-export --multi-cores 0-5 --single-core 0 --threads 6 \
  --iters 3000 --warmup 50 --use-sudo --nice -20

# Part C comparison (controlled single-core)
cd ../part_C
sudo taskset -c 0 nice -n -20 env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$(which python)" benchmark_pytorch.py \
  --cpp-mean-us <YOUR_CPP_MEAN> --cpp-median-us <YOUR_CPP_MEDIAN> --cpp-p99-us <YOUR_CPP_P99> \
  --iters 3000 --warmup 50

# Part C comparison (controlled multi-core, 6 threads)
cd ../part_C
taskset -c 0-5 nice -n -20 env \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
  "$(which python)" benchmark_pytorch.py \
  --cpp-mean-us <YOUR_CPP_MEAN> --cpp-median-us <YOUR_CPP_MEDIAN> --cpp-p99-us <YOUR_CPP_P99> \
  --iters 3000 --warmup 50
```

Google Colab (Part A training):
https://colab.research.google.com/drive/1PWD7w80TcWyLHg2Q7tilTP_pUpgmUVzb?usp=sharing

---

## Part A — Model

ResNet-20 for CIFAR-10 with a custom `TernaryConv2d` layer. Weights are constrained to `{-1, 0, +1}` using a threshold quantizer with per-tensor scale `alpha = mean(|W|)`. Gradients flow through a Straight-Through Estimator.

- Stem conv and projection shortcuts remain full-precision.
- 18 ternary 3×3 convolutions handle the bulk of compute.
- Exports `ternary_weights.npz` (int8 weights, alphas, layer shapes, 16-sample reference batch).

**Part A results** (`part_A/results.json`):

| Metric | Value |
|---|---|
| Ternary test accuracy (30 epochs) | 88.2% |
| Parameter count | 272,474 |
| Model file size | 1.14 MB |
| Ternary weight sparsity | 33.8% (fraction of weights quantized to 0) |

Trained for 30 epochs (reduced from 200 for time). Full 200-epoch runs yield ~91% per published ResNet-20 numbers. Accuracy does not affect the inference kernel or benchmark validity.

---

## Part B — C++ Inference Engine

A C++17 inference engine that loads the trained weights and runs ResNet-20 forward passes using AVX2 SIMD.

**Key design decisions:**

- **Dual-bitmap weight packing** — each ternary weight stored as two bits (`pos_bits`, `neg_bits`). The dot product reduces to masked float adds with zero floating-point multiplies in the inner loop.
- **Lookup-table mask expansion** — a compile-time 256-entry table converts each byte of weight bits into an AVX2 mask in one L1 load, no arithmetic decode needed.
- **BatchNorm folding** — BN parameters are folded into per-channel `scale` and `bias` at conversion time. Zero BN math at runtime.
- **im2col → GEMM** — input patches are linearized so the inner loop walks contiguous memory. 4 parallel FP accumulators per dot product to hide add latency.
- **Autotuned tile size** — `kSpatialTile=32` found via grid sweep over `{8,16,32,64}`. Doubles OpenMP work items vs the default 64, fixing Stage 3 (8×8) underutilization. Combined with `schedule(guided)` for adaptive chunk sizing.
- **Pre-allocated scratch buffers** — zero heap allocation during inference.

Build flags: `-O3 -march=native -mavx2 -mfma -mbmi2 -funroll-loops`

**Validation output:**
```
OK: 16/16 top-1 matches, max probability diff = 0.000003
```

---

## Part C — Benchmark Results

README shows the best controlled multi-core C++ result. Full benchmark history is in [RESULTS.md](RESULTS.md).

Controlled C++ methods:

```bash
# Single-core reference (used for reported table below)
sudo taskset -c 0 nice -n -20 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50

# Multi-core OpenMP (6 threads)
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

### Latency

Latest controlled rerun on this machine (April 2026):

- C++ run: pinned to 6 cores (`0-5`), `OMP_NUM_THREADS=6`, `nice -n -20`
- ORT run: `--threads 6`, `nice -n -20`

| Implementation | mean (us) | median (us) | p99 (us) |
|---|---|---|---|
| PyTorch baseline (FP32) | 1788.8 | 1700.5 | 3373.7 |
| PyTorch ternary (TernaryConv2d) | 3330.4 | 3083.6 | 7012.4 |
| **C++ AVX2 ternary (OpenMP, 6 threads, latest)** | **1611.4** | **1597.7** | **2328.5** |
| ORT FP32 baseline (`--threads 6`) | 329.8 | 321.6 | 607.5 |
| ORT FP32 ternary (frozen wts, `--threads 6`) | 328.6 | 322.9 | 560.7 |

Exploratory C++ run at 8 threads (`taskset -c 0-7`, `OMP_NUM_THREADS=8`, `nice -n -10`) reached:

- mean: 1666.97 us
- median: 1652.95 us
- p99: 2172.33 us

ORT ternary uses `do_constant_folding=True` during export, so ternary quantization arithmetic is absorbed into Conv weight constants. ONNX Runtime executes standard FP32 convolutions and has no runtime awareness of ternary structure.

### Speedup For 6 threads

The C++ kernel is now **faster than PyTorch ternary** and **roughly tied with PyTorch FP32 baseline on median latency**.

- C++ OpenMP vs C++ single-core: **2.70x** (mean), **2.72x** (median)
- C++ OpenMP vs PyTorch ternary: **2.07x** (mean), **1.93x** (median)
- C++ OpenMP vs PyTorch baseline: **1.11x** (mean), **1.06x** (median)
- ORT baseline (`--threads 6`) vs PyTorch baseline: **5.42x** (mean), **5.29x** (median)
- ORT ternary (`--threads 6`) vs PyTorch ternary: **10.14x** (mean), **9.55x** (median)
- ORT ternary (`--threads 6`) vs C++ AVX2 ternary (OpenMP, 6 threads): **4.90x** (mean), **4.96x** (median)

### Memory / Model Size

| File | Size |
|---|---|
| `baseline.pth` (FP32 weights) | 1111.9 KB |
| `ternary.pth` (FP32 storage, ternary values) | 1111.3 KB |
| `model.bin` (packed ternary, 2 bits/weight) | 280.1 KB |
| `baseline_fp32.onnx` | 84.3 KB |
| `ternary_fp32.onnx` | 237.6 KB |
| `python` peak RSS during benchmark | 3900.0 MB |

`model.bin` is **4× smaller** than `ternary.pth` — the ternary packing (dual bitmaps + folded BN) compresses conv weights from 32 bits/weight to 2 bits/weight.

ONNX export also improves artifact size vs `.pth` checkpoints:
- `baseline_fp32.onnx` is **13.19x smaller** than `baseline.pth` (**92.4% smaller**).
- `ternary_fp32.onnx` is **4.68x smaller** than `ternary.pth` (**78.6% smaller**).

### Why C++ is still slower than purely FP32 PyTorch

PyTorch dispatches `conv2d` on CPU to **oneDNN** (Intel's Deep Neural Network Library), which is a production-grade BLAS with:
- Multi-threaded execution across CPU cores and automatic work distribution
- Decades of cache-blocking optimisation (Goto GEMM)
- Hardware-specific codepaths tuned per microarchitecture

The C++ kernel uses (M,N) cache blocking, a 4-wide ternary dot product, and aggressive OpenMP parallelism. The table above reports the **best multi-core OpenMP (6-thread)** result; the controlled single-core reference path is tracked in the optimization history below.

Single-core structure:
- Spatial tile (M): 32 elements (autotuned)
- Output channel group (N): 4 channels per dot-product call (`dot_product_ternary_4x_avx2`)
- Activation row loaded **once** and applied to 4 output channels simultaneously → 4× fewer L2 reads

OpenMP multi-core structure:
- `collapse(2)` over `(oc_group × spatial_tile)` work items → **one barrier per layer** instead of one per spatial tile
- For 32×32 output with 32 spatial tiles: reduces from 32 barriers to 1 per ternary layer

Optimization history (single-core median, controlled runs):

| Step | Median (us) | Improvement |
|---|---:|---|
| Baseline (no opt) | 6061.09 | — |
| Compiler flags | 5427.73 | **10.5%** |
| M,N cache blocking | 5086.88 | **6.3%** |
| im2col fast path | 4862.89 | **4.4%** |
| 4-wide kernel + collapse(2) | 4344.98 | **10.7%** |
| **Total** | **4344.98** | **28.3%** (vs baseline) |

Full run history and failed experiments are in [RESULTS.md](RESULTS.md).

Dynamic INT8 comparison note (same pinned policy):
- `benchmark_int8_vs_cpp.py` compares C++ ternary against **ONNX Runtime dynamic INT8** in both multi-core and single-core modes.
- Latest run (`results_int8_vs_cpp.json`):
  - Multi-core (0-5, 6 threads): C++ **1938.00 / 1802.73 / 3229.75 us** (mean/median/p99), ORT baseline INT8 **2052.76 / 1879.66 / 3421.90 us**, ORT ternary INT8 **1661.21 / 1586.97 / 2813.80 us**.
  - Single-core (0): C++ **4448.19 / 4270.72 / 7221.33 us**, ORT baseline INT8 **3544.37 / 3447.45 / 5151.37 us**, ORT ternary INT8 **2107.36 / 2063.26 / 3264.19 us**.
- Interpretation: C++ beats ORT baseline dynamic INT8 in multi-core, but ORT ternary dynamic INT8 remains faster; in single-core, ORT dynamic INT8 is faster.
- FP32 ONNX values from `benchmark_onnx.py` are a different baseline and should not be mixed with INT8 conclusions.

Further gains would require INT8 quantisation to use AVX-VNNI `vpdpbusd` (4× throughput) or a fused streaming im2col to keep activation data in L1 rather than L2.

The **memory story is strong**: 4× smaller model file with the same accuracy, zero floating-point multiplies in the hot path, and reduced memory bandwidth — the structural efficiency of ternary is real even if single-threaded latency doesn't match production libraries.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy
pip install onnx onnxscript "onnxruntime>=1.19,<1.25"
```

C++ build requires CMake 3.16+ and a GCC/Clang with AVX2 support.

---

## System Configurations:
- Benchmarked on Intel i7-12650H (Alder Lake, AVX2 + AVX-VNNI) under WSL2.
- Median latency reported; WSL scheduling introduces ~2-5% run-to-run variance.
- Latest benchmark config: 3000 iterations, 50 warmup, CPU mode.

---

## Notes

- CIFAR-10 is downloaded automatically by torchvision on first run.
- `conv1` (stem) and projection shortcuts in residual blocks stay full-precision; only the 18 residual 3×3 convs are ternary.
- Part B correctness is verified by comparing C++ softmax outputs against the 16-sample reference batch exported from PyTorch in Part A (max diff = 3e-6).
- Model trained for 30 epochs on google collab (reduced from 200 for time constraints). Full 200-epoch training would yield ~92% baseline / ~91% ternary per published ResNet-20 results. Kernel correctness and benchmarking are independent of training duration.