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
python convert_weights.py ../part_A/ternary_weights.npz ../part_A/ternary.pth model.bin \
	--sample-input-bin ../part_A/sample_input.bin \
	--sample-output-bin ../part_A/sample_output.bin
```

`convert_weights.py` now calibrates the 18 ternary conv inputs from the 16 sample images and emits a version-2 `model.bin` with padded int8 ternary weights plus per-layer activation scales.
The runtime checks for AVX-VNNI at startup and uses the `vpdpbusd` path on supported Intel CPUs.
`conv_ternary` now uses a streaming tile-stationary uint8 microkernel: each thread builds only a small spatial tile (`kSpatialTile x k_pad`) instead of materializing a full im2col tensor for the whole feature map.
`model.bin` still stores only network weights/metadata (no embedded validation tensors), which keeps the file small.

## Validate

```bash
./build/ternary_infer model.bin --validate \
	--sample-input ../part_A/sample_input.bin \
	--expected-output ../part_A/sample_output.bin
```

Backward compatibility: if `model.bin` contains embedded samples from an older export, `--validate` without extra paths still works.

## Benchmark

### Single-core (reference)

```bash
sudo taskset -c 0 nice -n -20 env OMP_NUM_THREADS=1 \
	./build/ternary_infer model.bin --bench --iters 3000 --warmup 50
```

### Multi-core OpenMP (6 threads)

```bash
OMP_NUM_THREADS=6 taskset -c 0-5 ./build/ternary_infer model.bin \
	--bench --iters 3000 --warmup 50
```

Latest controlled output on this host:

| Policy | mean (us) | median (us) | p99 (us) |
|---|---:|---:|---:|
| Single-core (`taskset -c 0`, `OMP_NUM_THREADS=1`) | 2439.15 | 2399.91 | 3363.80 |
| Multi-core (`taskset -c 0-5`, `OMP_NUM_THREADS=6`) | 1013.43 | 1050.18 | 1397.70 |

Multi-core vs single-core speedup: **2.41x** (mean), **2.29x** (median)

The validation path compares output probabilities against the Part A `sample_outputs` tensor (stored in `sample_output.bin`).
