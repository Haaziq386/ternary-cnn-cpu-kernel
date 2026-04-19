#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PART_A_DIR = REPO_ROOT / "part_A"
if str(PART_A_DIR) not in sys.path:
    sys.path.insert(0, str(PART_A_DIR))

from resnet20 import ResNet20  # type: ignore  # noqa: E402
from ternary_layer import TernaryConv2d  # type: ignore  # noqa: E402


KIND_FP32 = 0
KIND_TERNARY = 1
KIND_LINEAR = 2


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected a state_dict in {path}")
    return state


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def named_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def get_hook_shapes(model: torch.nn.Module, sample_input: torch.Tensor) -> dict[str, dict[str, list[int]]]:
    shapes: dict[str, dict[str, list[int]]] = {}
    hooks = []

    def register(name: str, module: torch.nn.Module) -> None:
        if name == "":
            return

        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            shapes[name] = {
                "input_shape": list(inputs[0].shape),
                "output_shape": list(output.shape),
            }

        hooks.append(module.register_forward_hook(hook))

    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            register(module_name, module)

    model.eval()
    with torch.no_grad():
        model(sample_input)

    for hook in hooks:
        hook.remove()
    return shapes


def fold_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    weight = conv.weight.detach().cpu().numpy().astype(np.float32)
    gamma = bn.weight.detach().cpu().numpy().astype(np.float32)
    beta = bn.bias.detach().cpu().numpy().astype(np.float32)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    var = bn.running_var.detach().cpu().numpy().astype(np.float32)
    scale = gamma / np.sqrt(var + float(bn.eps))
    folded_weight = weight * scale[:, None, None, None]
    folded_bias = beta - scale * mean
    return folded_weight, folded_bias


def quantize_ternary(weight: np.ndarray, alpha: float) -> np.ndarray:
    ternary = np.rint(weight / (alpha + 1e-8)).clip(-1, 1).astype(np.int8)
    return ternary


def pack_ternary(ternary: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    flat = ternary.reshape(ternary.shape[0], -1)
    k_pad = round_up(flat.shape[1], 32)
    if k_pad != flat.shape[1]:
        pad = np.zeros((flat.shape[0], k_pad - flat.shape[1]), dtype=np.int8)
        flat = np.concatenate([flat, pad], axis=1)

    pos = np.packbits((flat > 0).astype(np.uint8), axis=1, bitorder="little")
    neg = np.packbits((flat < 0).astype(np.uint8), axis=1, bitorder="little")
    return pos.astype(np.uint8), neg.astype(np.uint8), k_pad


def write_u32_layer_header(
    out,
    kind: int,
    in_channels: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    output_h: int,
    output_w: int,
    k_pad: int,
    has_bias: int,
    name: str,
) -> None:
    encoded = name.encode("utf-8")
    out.write(
        struct.pack(
            "<15I",
            kind,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_h,
            output_w,
            k_pad,
            has_bias,
            len(encoded),
            0,
        )
    )
    out.write(encoded)


def write_fp32_conv(out, name: str, conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d | None, shape: dict[str, list[int]]) -> None:
    if bn is not None:
        weight, bias = fold_conv_bn(conv, bn)
    else:
        weight = conv.weight.detach().cpu().numpy().astype(np.float32)
        bias = conv.bias.detach().cpu().numpy().astype(np.float32) if conv.bias is not None else np.zeros(weight.shape[0], dtype=np.float32)
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    output_h = shape["output_shape"][2]
    output_w = shape["output_shape"][3]
    write_u32_layer_header(
        out,
        KIND_FP32,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        conv.stride[0],
        conv.stride[1],
        conv.padding[0],
        conv.padding[1],
        output_h,
        output_w,
        0,
        1,
        name,
    )
    out.write(weight.astype(np.float32).tobytes(order="C"))
    out.write(bias.astype(np.float32).tobytes(order="C"))


def write_ternary_conv(out, name: str, module: torch.nn.Module, bn: torch.nn.BatchNorm2d, npz: dict[str, np.ndarray] | None, shape: dict[str, list[int]]) -> None:
    safe_name = name.replace(".", "_")
    weight_np: np.ndarray
    alpha: float
    if npz is not None and f"weights_{safe_name}" in npz and f"alpha_{safe_name}" in npz:
        weight_np = np.asarray(npz[f"weights_{safe_name}"], dtype=np.int8)
        alpha = float(np.asarray(npz[f"alpha_{safe_name}"], dtype=np.float32).item())
    else:
        conv_weight = module.weight.detach().cpu().numpy().astype(np.float32)
        alpha = float(np.abs(conv_weight).mean())
        weight_np = quantize_ternary(conv_weight, alpha)

    ternary = weight_np.astype(np.int8)
    pos, neg, k_pad = pack_ternary(ternary)
    gamma = bn.weight.detach().cpu().numpy().astype(np.float32)
    beta = bn.bias.detach().cpu().numpy().astype(np.float32)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    var = bn.running_var.detach().cpu().numpy().astype(np.float32)
    scale = (gamma / np.sqrt(var + float(bn.eps))) * alpha
    bias = beta - (gamma / np.sqrt(var + float(bn.eps))) * mean

    out_channels, in_channels, kernel_h, kernel_w = ternary.shape
    output_h = shape["output_shape"][2]
    output_w = shape["output_shape"][3]
    write_u32_layer_header(
        out,
        KIND_TERNARY,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        module.stride[0],
        module.stride[1],
        module.padding[0],
        module.padding[1],
        output_h,
        output_w,
        k_pad,
        1,
        name,
    )
    out.write(pos.astype(np.uint8, copy=False).tobytes(order="C"))
    out.write(neg.astype(np.uint8, copy=False).tobytes(order="C"))
    out.write(scale.astype(np.float32).tobytes(order="C"))
    out.write(bias.astype(np.float32).tobytes(order="C"))


def write_linear(out, name: str, fc: torch.nn.Linear, shape: dict[str, list[int]]) -> None:
    weight = fc.weight.detach().cpu().numpy().astype(np.float32)
    bias = fc.bias.detach().cpu().numpy().astype(np.float32)
    out_features, in_features = weight.shape
    write_u32_layer_header(
        out,
        KIND_LINEAR,
        in_features,
        out_features,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        name,
    )
    out.write(weight.tobytes(order="C"))
    out.write(bias.tobytes(order="C"))


def build_model(model: torch.nn.Module, sample_input: torch.Tensor, npz: dict[str, np.ndarray] | None, out_path: Path) -> None:
    modules = named_modules(model)
    shapes = get_hook_shapes(model, sample_input)

    ordered_blocks = [
        ("layer1.0", False),
        ("layer1.1", False),
        ("layer1.2", False),
        ("layer2.0", True),
        ("layer2.1", False),
        ("layer2.2", False),
        ("layer3.0", True),
        ("layer3.1", False),
        ("layer3.2", False),
    ]

    _, input_channels, input_h, input_w = sample_input.shape
    num_classes = model.fc.out_features
    layer_count = 1 + 2 * len(ordered_blocks) + sum(1 for _, has_proj in ordered_blocks if has_proj) + 1

    header = struct.pack(
        "<8s12I",
        b"TRNCNNB1",
        1,
        0,
        input_channels,
        input_h,
        input_w,
        num_classes,
        layer_count,
        0,
        0,
        0,
        0,
        0,
    )

    with out_path.open("wb") as out:
        out.write(header)

        stem_conv = modules["conv1"]
        stem_bn = modules["bn1"]
        write_fp32_conv(out, "conv1", stem_conv, stem_bn, shapes["conv1"])

        for block_name, has_proj in ordered_blocks:
            conv1 = modules[f"{block_name}.conv1"]
            bn1 = modules[f"{block_name}.bn1"]
            conv2 = modules[f"{block_name}.conv2"]
            bn2 = modules[f"{block_name}.bn2"]
            write_ternary_conv(out, f"{block_name}.conv1", conv1, bn1, npz, shapes[f"{block_name}.conv1"])
            write_ternary_conv(out, f"{block_name}.conv2", conv2, bn2, npz, shapes[f"{block_name}.conv2"])
            out.write(struct.pack("<I", 1 if has_proj else 0))
            if has_proj:
                proj_conv = modules[f"{block_name}.shortcut.0"]
                proj_bn = modules[f"{block_name}.shortcut.1"]
                write_fp32_conv(out, f"{block_name}.shortcut.0", proj_conv, proj_bn, shapes[f"{block_name}.shortcut.0"])

        write_linear(out, "fc", model.fc, shapes["fc"])


def export_validation_tensors(npz: dict[str, np.ndarray], sample_input_bin: Path, sample_output_bin: Path) -> None:
    sample_input = np.asarray(npz["sample_input"], dtype=np.float32)
    sample_output = np.asarray(npz["sample_outputs"], dtype=np.float32)
    sample_input_bin.parent.mkdir(parents=True, exist_ok=True)
    sample_output_bin.parent.mkdir(parents=True, exist_ok=True)
    sample_input_bin.write_bytes(sample_input.tobytes(order="C"))
    sample_output_bin.write_bytes(sample_output.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Part A checkpoint into Part B model.bin")
    parser.add_argument("npz_path", type=Path, help="Path to ternary_weights.npz")
    parser.add_argument("pth_path", type=Path, help="Path to ternary.pth")
    parser.add_argument("out_path", type=Path, help="Output model.bin path")
    parser.add_argument("--sample-input-bin", type=Path, default=None, help="Optional output path for raw float32 NCHW sample input")
    parser.add_argument("--sample-output-bin", type=Path, default=None, help="Optional output path for raw float32 NxC expected probabilities")
    args = parser.parse_args()

    npz = load_npz(args.npz_path) if args.npz_path.exists() else None
    state_dict = load_state_dict(args.pth_path)

    model = ResNet20(conv_cls=TernaryConv2d)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if npz is None:
        raise FileNotFoundError(f"missing NPZ export: {args.npz_path}")

    sample_input = torch.from_numpy(np.asarray(npz["sample_input"], dtype=np.float32))
    build_model(model, sample_input, npz, args.out_path)
    if args.sample_input_bin is not None or args.sample_output_bin is not None:
        if args.sample_input_bin is None or args.sample_output_bin is None:
            raise ValueError("provide both --sample-input-bin and --sample-output-bin")
        export_validation_tensors(npz, args.sample_input_bin, args.sample_output_bin)
        print(f"Wrote {args.sample_input_bin}")
        print(f"Wrote {args.sample_output_bin}")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()