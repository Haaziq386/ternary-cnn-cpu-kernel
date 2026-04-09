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

### Single-core (reference)

```bash
sudo taskset -c 0 nice -n -20 ./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

### Multi-core OpenMP (6 threads)

```bash
OMP_NUM_THREADS=6 taskset -c 0-5 ./build/ternary_infer model.bin \
	--bench --iters 3000 --warmup 50
```

The validation path compares output probabilities against `sample_outputs` from Part A.

