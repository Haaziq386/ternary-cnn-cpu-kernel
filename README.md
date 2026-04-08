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

# Part C — benchmark PyTorch and compare
cd ../part_C
python benchmark_pytorch.py \
  --cpp-mean-us 5714.55 --cpp-median-us 5656.07 --cpp-p99-us 6843.96 \
  --iters 1000 --warmup 10
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
- **Pre-allocated scratch buffers** — zero heap allocation during inference.

Build flags: `-O3 -march=native -mavx2 -mfma -mbmi2 -funroll-loops`

**Validation output:**
```
OK: 16/16 top-1 matches, max probability diff = 0.000003
```

---

## Part C — Benchmark Results

Single-image CPU inference, single-threaded, 1000 iterations with 10 warmup. Full results in `part_C/results.json`.

### Latency

| Implementation | mean (µs) | median (µs) | p99 (µs) |
|---|---|---|---|
| PyTorch baseline (FP32) | 1114 | 1065 | 1925 |
| PyTorch ternary (TernaryConv2d) | 1975 | 1931 | 3313 |
| **C++ AVX2 ternary kernel** | **5715** | **5656** | **6844** |

### Speedup

The C++ kernel is currently **slower** than PyTorch (~0.35× vs PyTorch ternary, ~0.19× vs baseline). See the note below.

### Memory / Model Size

| File | Size |
|---|---|
| `baseline.pth` (FP32 weights) | 1111.9 KB |
| `ternary.pth` (FP32 storage, ternary values) | 1111.3 KB |
| `model.bin` (packed ternary, 2 bits/weight) | 280.1 KB |

`model.bin` is **4× smaller** than `ternary.pth` — the ternary packing (dual bitmaps + folded BN) compresses conv weights from 32 bits/weight to 2 bits/weight.

### Why C++ is slower here

PyTorch dispatches `conv2d` on CPU to **oneDNN** (Intel's Deep Neural Network Library), which is a production-grade BLAS with:
- Multi-threaded execution across all CPU cores by default
- Decades of cache-blocking optimisation (Goto GEMM)
- Hardware-specific codepaths tuned per microarchitecture

The C++ kernel here is **single-threaded** and uses a naive `for oc { for spatial { dot_product() } }` loop with no GEMM-level cache tiling. This is a research/demonstration kernel, not a full inference engine.

To close the gap the next steps would be: OpenMP parallelism across output channels, cache-blocked GEMM tiling (M×N×K tiles), and INT8 activation quantisation to use AVX-VNNI `vpdpbusd` for 4× throughput.

The **memory story is strong**: 4× smaller model file with the same accuracy, and zero floating-point multiplies in the hot path — the structural efficiency of ternary is real even if the single-threaded latency benchmark doesn't capture it here.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy
```

C++ build requires CMake 3.16+ and a GCC/Clang with AVX2 support.

---

## Notes

- CIFAR-10 is downloaded automatically by torchvision on first run.
- `conv1` (stem) and projection shortcuts in residual blocks stay full-precision; only the 18 residual 3×3 convs are ternary.
- Part B correctness is verified by comparing C++ softmax outputs against the 16-sample reference batch exported from PyTorch in Part A (max diff = 3e-6).
- Model trained for 30 epochs on google collab (reduced from 200 for time constraints). Full 200-epoch training would yield ~92% baseline / ~91% ternary per published ResNet-20 results. Kernel correctness and benchmarking are independent of training duration.