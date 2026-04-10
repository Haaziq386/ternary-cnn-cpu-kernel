#!/usr/bin/env python3
"""Export baseline and ternary ResNet-20 to ONNX format.
Produces: part_C/onnx_models/baseline_fp32.onnx
          part_C/onnx_models/ternary_fp32.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
PART_A = REPO_ROOT / "part_A"
if str(PART_A) not in sys.path:
    sys.path.insert(0, str(PART_A))

from resnet20 import ResNet20  # noqa: E402
from ternary_layer import TernaryConv2d  # noqa: E402


def export(model: nn.Module, dummy: torch.Tensor, path: Path, opset: int) -> float:
    torch.onnx.export(
        model,
        dummy,
        str(path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return round(path.stat().st_size / 1024, 1)


def verify(onnx_path: Path, dummy_np: np.ndarray, model: nn.Module) -> None:
    """Assert ORT output matches PyTorch within 1e-4."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy_np})[0]
    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy_np)).numpy()
    max_diff = abs(ort_out - pt_out).max()
    assert max_diff < 1e-4, f"max_diff={max_diff}"
    print(f"  verified: max_diff={max_diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=REPO_ROOT / "part_A" / "ternary_weights.npz")
    parser.add_argument("--pth", type=Path, default=REPO_ROOT / "part_A" / "ternary.pth")
    parser.add_argument("--baseline-pth", type=Path, default=REPO_ROOT / "part_A" / "baseline.pth")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "onnx_models")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = np.load(args.npz, allow_pickle=True)
    dummy_np = np.asarray(npz["sample_input"], dtype=np.float32)[:1]
    dummy = torch.from_numpy(dummy_np)

    print("Exporting baseline FP32...")
    try:
        baseline_state = torch.load(args.baseline_pth, map_location="cpu", weights_only=True)
    except TypeError:
        baseline_state = torch.load(args.baseline_pth, map_location="cpu")

    baseline_model = ResNet20(conv_cls=nn.Conv2d)
    baseline_model.load_state_dict(baseline_state)
    baseline_model.eval()

    baseline_path = args.out_dir / "baseline_fp32.onnx"
    baseline_kb = export(baseline_model, dummy, baseline_path, args.opset)
    verify(baseline_path, dummy_np, baseline_model)
    print(f"  baseline_fp32.onnx: {baseline_kb} KB")

    print("Exporting ternary FP32 (constant-folded)...")
    try:
        ternary_state = torch.load(args.pth, map_location="cpu", weights_only=True)
    except TypeError:
        ternary_state = torch.load(args.pth, map_location="cpu")

    ternary_model = ResNet20(conv_cls=TernaryConv2d)
    ternary_model.load_state_dict(ternary_state)
    ternary_model.eval()

    ternary_path = args.out_dir / "ternary_fp32.onnx"
    ternary_kb = export(ternary_model, dummy, ternary_path, args.opset)
    verify(ternary_path, dummy_np, ternary_model)
    print(f"  ternary_fp32.onnx:  {ternary_kb} KB")
    print("Done.")


if __name__ == "__main__":
    main()
