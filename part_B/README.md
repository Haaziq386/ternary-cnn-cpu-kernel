# Part B

This directory contains the CPU inference implementation for the ternary ResNet-20 exported from Part A.

## Build

```bash
cd part_B
cmake -S . -B build
cmake --build build -j
```

## Convert weights

```bash
python convert_weights.py ../part_A/ternary_weights.npz ../part_A/ternary.pth model.bin
```

## Validate

```bash
./build/ternary_infer model.bin --validate
```

## Benchmark

```bash
./build/ternary_infer model.bin --bench --iters 1000 --warmup 10
```

The validation path compares output probabilities against `sample_outputs` from Part A.

