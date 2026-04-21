# Ternary CNN CPU Kernel

Implementation and benchmarking of a ternary-weight ResNet-20 on CIFAR-10, with a hand-written AVX2/AVX-VNNI CPU inference kernel.

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
cmake -S . -B build -DPROFILE_LAYERS=ON && cmake --build build -j
python convert_weights.py ../part_A/ternary_weights.npz ../part_A/ternary.pth model.bin \
  --sample-input-bin ../part_A/sample_input.bin \
  --sample-output-bin ../part_A/sample_output.bin

# Validate: C++ output must match PyTorch reference
./build/ternary_infer model.bin --validate \
  --sample-input ../part_A/sample_input.bin \
  --expected-output ../part_A/sample_output.bin

# Benchmark: single-image latency
./build/ternary_infer model.bin --bench --iters 1000 --warmup 10

# Part B benchmark (controlled single-core)
sudo taskset -c 0 nice -n -20 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50

# Part B benchmark (OpenMP multi-core, 6 threads)
sudo taskset -c 0-5 nice -n -20 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50

# Part B benchmark + perf-like runtime counters (PROFILE_LAYERS build only)
taskset -c 0-5 env OMP_NUM_THREADS=6 \
  ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50 --perf-like

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

# Part C — compare C++ ternary vs ONNX static INT8 (single + multi-core in one run)
# Run inside the project virtual env (.venv):
"$(which python)" benchmark_static_int8_vs_cpp.py \
  --skip-export --multi-cores 0-5 --single-core 0 --threads 6 \
  --iters 3000 --warmup 50 --nice -5

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
- **Lower OpenMP barrier overhead** — added `nowait` on key `omp for` hot loops where no cross-thread dependency exists, reducing unnecessary synchronization.
- **Pre-allocated scratch buffers** — zero heap allocation during inference.

Build flags: `-O3 -march=native -mavx2 -mfma -mbmi2 -funroll-loops`

**Validation output:**
```
OK: 16/16 top-1 matches, max probability diff = 0.020357
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
| **C++ INT8/VNNI ternary (OpenMP, 6 threads, latest)** | **1066.93** | **1105.48** | **1605.87** |
| ORT FP32 baseline (`--threads 6`) | 329.8 | 321.6 | 607.5 |
| ORT FP32 ternary (frozen wts, `--threads 6`) | 328.6 | 322.9 | 560.7 |

Exploratory C++ run at 8 threads (`taskset -c 0-7`, `OMP_NUM_THREADS=8`, `nice -n -10`) reached:

- mean: 1666.97 us
- median: 1652.95 us
- p99: 2172.33 us

ORT ternary uses `do_constant_folding=True` during export, so ternary quantization arithmetic is absorbed into Conv weight constants. ONNX Runtime executes standard FP32 convolutions and has no runtime awareness of ternary structure.

### Speedup For 6 threads

The C++ INT8/VNNI path is now **faster than PyTorch ternary** and **faster than the PyTorch FP32 baseline** on this machine, while ONNX Runtime FP32 still leads on absolute latency.

- C++ INT8/VNNI OpenMP vs C++ single-core: **2.83x** (mean), **2.84x** (median)
- C++ INT8/VNNI OpenMP vs PyTorch ternary: **3.12x** (mean), **2.79x** (median)
- C++ INT8/VNNI OpenMP vs PyTorch baseline: **1.68x** (mean), **1.54x** (median)
- ORT baseline (`--threads 6`) vs PyTorch baseline: **5.42x** (mean), **5.29x** (median)
- ORT ternary (`--threads 6`) vs PyTorch ternary: **10.14x** (mean), **9.55x** (median)
- ORT ternary (`--threads 6`) vs C++ INT8/VNNI ternary (OpenMP, 6 threads): **3.26x** (mean), **3.42x** (median)

### Memory / Model Size

| File | Size |
|---|---|
| `baseline.pth` (FP32 weights) | 1111.9 KB |
| `ternary.pth` (FP32 storage, ternary values) | 1111.3 KB |
| `model.bin` (version-2 int8 ternary weights + calibrated activation scales; no embedded validation tensors) | 284.6 KB |
| `sample_input.bin` (validation input, float32 NCHW) | 192.0 KB |
| `sample_output.bin` (validation expected output, float32 NxC) | 0.6 KB |
| `baseline_fp32.onnx` | 84.3 KB |
| `ternary_fp32.onnx` | 237.6 KB |
| `python` peak RSS during benchmark | 3900.0 MB |

`model.bin` is **~3.9× smaller** than `ternary.pth` — version-2 int8 ternary storage keeps the model compact while trading size for a much faster AVX-VNNI runtime path, and it no longer stores validation samples in the model artifact.

ONNX export also improves artifact size vs `.pth` checkpoints:
- `baseline_fp32.onnx` is **13.19x smaller** than `baseline.pth` (**92.4% smaller**).
- `ternary_fp32.onnx` is **4.68x smaller** than `ternary.pth` (**78.6% smaller**).

### Why C++ is still slower than purely FP32 PyTorch
- **Version-2 INT8 ternary weights** — the exporter now calibrates the 18 ternary conv inputs from the 16-sample batch and writes padded int8 weights plus per-layer activation scales.
- **AVX-VNNI inner loop** — ternary convs accumulate with `_mm256_dpbusd_epi32` (`vpdpbusd`) over uint8 activations and signed int8 weights.
- **BatchNorm folding** — BN parameters are folded into per-channel `scale` and `bias` at conversion time. Zero BN math at runtime.
- **Streaming quantized im2col tiles** — input patches are quantized directly into a small per-thread tile buffer (`kSpatialTile x k_pad`) and consumed immediately by the VNNI kernel, avoiding full-feature-map im2col write/read traffic.
- **Autotuned tile size** — `kSpatialTile=32` found via grid sweep over `{8,16,32,64}`. Doubles OpenMP work items vs the default 64, fixing Stage 3 (8×8) underutilization. Combined with `schedule(guided)` for adaptive chunk sizing.
- **Lower OpenMP barrier overhead** — added `nowait` on key `omp for` hot loops where no cross-thread dependency exists, reducing unnecessary synchronization.
- **Pre-allocated scratch buffers** — zero heap allocation during inference.

Single-core structure:
- Spatial tile (M): 32 elements (autotuned)
- Activation row loaded **once** and applied to 4 output channels simultaneously → 4× fewer L2 reads

OpenMP multi-core structure:
- `collapse(2)` over `(oc_group × spatial_tile)` work items → **one barrier per layer** instead of one per spatial tile
- For 32×32 output with 32 spatial tiles: reduces from 32 barriers to 1 per ternary layer

Optimization history (single-core median, controlled runs):

- Baseline (no opt): 6061.09 us
- Compiler flags: 5427.73 us
- M,N cache blocking: 5086.88 us
- im2col fast path: 4862.89 us
- 4-wide kernel + collapse(2): 4344.98 us

Full run history and failed experiments are in [RESULTS.md](RESULTS.md).

Dynamic INT8 comparison note (same pinned policy):
`model.bin` is **~3.9× smaller** than `ternary.pth` — version-2 int8 ternary storage keeps the model compact while trading size for a much faster AVX-VNNI runtime path, and it no longer stores validation samples in the model artifact.
- Latest run (`results_int8_vs_cpp.json`):
  - Multi-core (0-5, 6 threads): C++ **1794.70 / 1788.51 / 2778.65 us** (mean/median/p99), ORT baseline INT8 **2013.82 / 1992.00 / 3113.35 us**, ORT ternary INT8 **1294.21 / 1282.21 / 1981.35 us**.
  - Single-core (0): C++ **4498.38 / 4469.25 / 6065.87 us**, ORT baseline INT8 **3644.35 / 3563.22 / 5537.66 us**, ORT ternary INT8 **2102.85 / 2033.68 / 3055.82 us**.
- Interpretation: C++ beats ORT baseline dynamic INT8 in multi-core, but ORT ternary dynamic INT8 remains faster; in single-core, ORT dynamic INT8 is faster.
- FP32 ONNX values from `benchmark_onnx.py` are a different baseline and should not be mixed with INT8 conclusions.

Static INT8 comparison note:
- `benchmark_static_int8_vs_cpp.py` compares C++ ternary against **ONNX Runtime static INT8 (QDQ, MinMax calibration with 16 samples from `sample_input`)**.
- Latest run (`results_static_int8_vs_cpp.json`):
  - Multi-core (0-5, 6 threads): C++ **1786.07 / 1782.85 / 2747.49 us**, ORT baseline static INT8 **250.43 / 237.23 / 503.99 us**, ORT ternary static INT8 **450.68 / 418.38 / 957.47 us**.
  - Single-core (0): C++ **4653.74 / 4652.73 / 6338.38 us**, ORT baseline static INT8 **408.25 / 396.07 / 812.18 us**, ORT ternary static INT8 **439.36 / 420.49 / 829.28 us**.
- Matched FP32 ONNX reference from `.venv` (`results_onnx.json`, 6 threads pinned to cores 0-5):
  - ORT baseline FP32 **309.91 / 298.46 / 555.91 us**
  - ORT ternary FP32 **309.08 / 290.23 / 564.94 us**
- Interpretation: on this machine and policy, **static INT8 baseline is faster than both dynamic INT8 and FP32**. Static INT8 ternary is much faster than dynamic INT8 ternary, but still slower than static INT8 baseline.

Further gains likely require reducing scalar FP32 work around the ternary conv path (residual add/ReLU/writeback traffic) and/or deeper kernel fusion beyond the current streaming VNNI conv implementation.

The **memory story is strong**: ~12.6× smaller deploy-time model file with the same accuracy, zero floating-point multiplies in the hot path, and reduced memory bandwidth — the structural efficiency of ternary is real even if single-threaded latency doesn't match production libraries.

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
- Part B correctness is verified by comparing C++ softmax outputs against the 16-sample reference batch exported from PyTorch in Part A (`sample_input.bin` + `sample_output.bin`, max diff = 3e-6).
- Model trained for 30 epochs on google collab (reduced from 200 for time constraints). Full 200-epoch training would yield ~92% baseline / ~91% ternary per published ResNet-20 results. Kernel correctness and benchmarking are independent of training duration.