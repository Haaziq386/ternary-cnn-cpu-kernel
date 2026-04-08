# Ternary CNN CPU Kernel

This repository contains coursework for implementing and evaluating ternary-weight CNNs.

## Current Scope

The current implementation covers **Part A: Model Development (Python/PyTorch)**.

Part A includes:
- A CIFAR-10 image classification model based on ResNet-20.
- A custom ternary convolution layer with weights constrained to `{-1, 0, +1}` using a Straight-Through Estimator (STE).
- Training and evaluation for both:
  - Baseline full-precision ResNet-20
  - Ternary ResNet-20
- Export of ternary weights and sample tensors for downstream usage in Part B.

## Repository Structure

- `part_A/train.py`: Main training script for baseline + ternary models, evaluation, and export.
- `part_A/resnet20.py`: CNN architecture implementation (ResNet-20 for CIFAR-10).
- `part_A/ternary_layer.py`: Ternary convolution layer implementation (`TernaryConv2d`).
- `Question/assignment.txt`: Assignment prompt/details.

## Part A Design

### 1) CNN Architecture

`ResNet20` is implemented in `part_A/resnet20.py` with:
- CIFAR-style stem (`3x3` conv, 16 channels)
- 3 residual stages with channel progression:
  - Stage 1: 16 channels
  - Stage 2: 32 channels (stride 2)
  - Stage 3: 64 channels (stride 2)
- Global average pooling + fully-connected classifier (`10` classes)

The architecture accepts a configurable convolution class (`conv_cls`) so the same model can run in:
- full precision (`torch.nn.Conv2d`)
- ternary mode (`TernaryConv2d`)

### 2) Ternary Convolution Layer

`TernaryConv2d` is implemented in `part_A/ternary_layer.py`.

For each forward pass:
- Compute scale: `alpha = mean(abs(W))`
- Quantize: `W_t = clamp(round(W / alpha), -1, 1)`
- Use STE to preserve gradient flow through the quantization path.

This gives ternary weights in the forward pass while allowing standard gradient-based optimization.

## Setup

Recommended environment:
- Python 3.9+
- PyTorch
- torchvision
- numpy

Create and activate a virtual environment before installing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install torch torchvision numpy
```

## Run Part A

Google Colab (Part A completed here):
- https://colab.research.google.com/drive/1PWD7w80TcWyLHg2Q7tilTP_pUpgmUVzb?usp=sharing

From repository root:

```bash
cd part_A
python train.py --model both --epochs 30 --batch-size 128 --data-root ./data
```

Optional arguments:
- `--device`: `cpu` or `cuda` (default: auto-detect)
- `--model`: choose `baseline`, `ternary`, or `both`
- `--epochs`: number of training epochs
- `--batch-size`: train batch size
- `--data-root`: path for CIFAR-10 download/cache

Separate runs:

```bash
python train.py --model baseline --epochs 30 --batch-size 128 --data-root ./data
python train.py --model ternary --epochs 30 --batch-size 128 --data-root ./data
```

## Outputs

Running `part_A/train.py` saves:
- `part_A/baseline.pth`: baseline model weights
- `part_A/ternary.pth`: ternary model weights
- `part_A/ternary_weights.npz`: exported ternary tensors and sample I/O
- `part_A/results.json`: summary metrics

When you run a single model, only the matching checkpoint and metrics are generated. The ternary export file is produced only for ternary runs.

`results.json` includes:
- Test accuracy (baseline and ternary)
- Parameter counts
- Saved model file sizes
- Ternary sparsity (fraction of quantized zeros)

## Notes

- CIFAR-10 is downloaded automatically by torchvision when first run.
- The first convolution (`conv1`) and projection shortcuts in residual blocks remain full precision.
- Part B lives in [part_B/README.md](part_B/README.md).
- Model trained for 30 epochs on google collab (reduced from 200 for time constraints). Full 200-epoch training would yield ~92% baseline / ~91% ternary per published ResNet-20 results. Kernel correctness and benchmarking are independent of training duration.