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
	--sample-output-bin ../part_A/sample_output.bin \
	--ternary-format tl --tl-group-size 2
```

`convert_weights.py` now calibrates the 18 ternary conv inputs from the 16 sample images and emits a version-2 `model.bin` with padded TL weights (default) plus per-layer activation scales.
The runtime supports both `KIND_TERNARY_TL` (TL path) and `KIND_TERNARY_INT8` (legacy `vpdpbusd` path), preserving backward compatibility.
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

### Multi-core OpenMP (6 threads)

```bash
OMP_NUM_THREADS=6 taskset -c 0-5 ./build/ternary_infer model.bin \
	--bench --iters 3000 --warmup 50
```

The validation path compares output probabilities against the Part A `sample_outputs` tensor (stored in `sample_output.bin`).

