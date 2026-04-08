"""
Part A: Ternary CNN on CIFAR-10
- Baseline: full-precision ResNet-20
- Ternary:  ResNet-20 with TernaryConv2d (weights in {-1, 0, +1})
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet20 import ResNet20
from ternary_layer import TernaryConv2d


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_loaders(batch_size: int = 128, num_workers: int = 4,
                data_root: str = "./data"):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total


def train_model(model, train_loader, test_loader, device, epochs,
                label: str):
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        best_acc = max(best_acc, test_acc)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[{label}] epoch {epoch:3d}/{epochs} | "
                  f"loss {loss:.4f} | train {train_acc*100:.2f}% | "
                  f"test {test_acc*100:.2f}% | "
                  f"lr {scheduler.get_last_lr()[0]:.5f} | "
                  f"{time.time()-t0:.1f}s")
    return best_acc


# ---------------------------------------------------------------------------
# Model stats
# ---------------------------------------------------------------------------

def param_count(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(path):
    return os.path.getsize(path) / 1e6


@torch.no_grad()
def ternary_sparsity(model):
    """Fraction of ternary conv weights that are exactly 0 after quantization."""
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, TernaryConv2d):
            alpha = m.weight.abs().mean()
            W_t = (m.weight / (alpha + 1e-8)).round().clamp(-1.0, 1.0)
            zeros += (W_t == 0).sum().item()
            total += W_t.numel()
    return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Export ternary weights for Part B
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_ternary_weights(model, sample_loader, device, out_path: str):
    """Save ternary weights, alphas, layer shapes, and a sample batch."""
    model.eval()

    arrays = {}
    layer_shapes = {}  # keyed by layer name
    hooks = []

    # --- collect per-layer info via forward hook ---
    def make_hook(name):
        def hook(module, inp, out):
            layer_shapes[name] = {
                "input_shape":  list(inp[0].shape),   # (N, C_in, H, W)
                "output_shape": list(out.shape),       # (N, C_out, H', W')
                "kernel_size":  list(module.weight.shape),  # (C_out, C_in, kH, kW)
                "stride":       list(module.stride),
                "padding":      list(module.padding),
            }
        return hook

    for name, module in model.named_modules():
        if isinstance(module, TernaryConv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # grab one sample batch
    sample_imgs, sample_labels = next(iter(sample_loader))
    sample_imgs  = sample_imgs[:16].to(device)
    sample_labels = sample_labels[:16].to(device)

    # run forward to trigger hooks and get outputs
    logits = model(sample_imgs)
    probs  = F.softmax(logits, dim=1)

    for h in hooks:
        h.remove()

    # --- quantize and store weights ---
    for name, module in model.named_modules():
        if isinstance(module, TernaryConv2d):
            alpha = module.weight.abs().mean().item()
            W_t   = (module.weight / (alpha + 1e-8)).round().clamp(-1.0, 1.0)
            safe_name = name.replace(".", "_")
            arrays[f"weights_{safe_name}"] = W_t.cpu().numpy().astype(np.int8)
            arrays[f"alpha_{safe_name}"]   = np.float32(alpha)

    # --- store layer shapes ---
    shapes_list = []
    for name, shapes in layer_shapes.items():
        safe_name = name.replace(".", "_")
        shapes["layer_name"] = name
        shapes_list.append(shapes)
    arrays["layer_shapes"] = np.array(json.dumps(shapes_list), dtype=object)

    # --- store sample batch + expected outputs ---
    arrays["sample_input"]   = sample_imgs.cpu().numpy()    # float32 (16, 3, 32, 32)
    arrays["sample_labels"]  = sample_labels.cpu().numpy()  # int64   (16,)
    arrays["sample_outputs"] = probs.cpu().numpy()          # float32 (16, 10)

    np.savez(out_path, **arrays)
    print(f"Saved ternary weights to {out_path}  ({len(arrays)} arrays)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train baseline & ternary ResNet-20 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--model",
        choices=("both", "baseline", "ternary"),
        default="both",
        help="Choose which model(s) to train: full baseline, ternary, or both.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size, data_root=args.data_root
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    run_baseline = args.model in ("both", "baseline")
    run_ternary = args.model in ("both", "ternary")

    # -----------------------------------------------------------------------
    # 1. Baseline (full-precision)
    # -----------------------------------------------------------------------
    baseline_path = os.path.join(out_dir, "baseline.pth")
    if run_baseline:
        print("\n" + "="*60)
        print("Training BASELINE (full-precision ResNet-20)")
        print("="*60)
        baseline = ResNet20(conv_cls=nn.Conv2d).to(device)
        train_model(baseline, train_loader, test_loader, device,
                    args.epochs, label="baseline")
        final_baseline_acc = evaluate(baseline, test_loader, device)
        torch.save(baseline.state_dict(), baseline_path)
        results["baseline"] = {
            "accuracy":      round(final_baseline_acc * 100, 2),
            "param_count":   param_count(baseline),
            "model_size_mb": round(model_size_mb(baseline_path), 4),
        }
        print(f"Baseline final test accuracy: {final_baseline_acc*100:.2f}%")

    # -----------------------------------------------------------------------
    # 2. Ternary
    # -----------------------------------------------------------------------
    npz_path = os.path.join(out_dir, "ternary_weights.npz")
    ternary_path = os.path.join(out_dir, "ternary.pth")
    if run_ternary:
        print("\n" + "="*60)
        print("Training TERNARY ResNet-20")
        print("="*60)
        ternary = ResNet20(conv_cls=TernaryConv2d).to(device)
        train_model(ternary, train_loader, test_loader, device,
                    args.epochs, label="ternary")
        final_ternary_acc = evaluate(ternary, test_loader, device)
        torch.save(ternary.state_dict(), ternary_path)
        export_ternary_weights(ternary, test_loader, device, npz_path)
        results["ternary"] = {
            "accuracy":      round(final_ternary_acc * 100, 2),
            "param_count":   param_count(ternary),
            "model_size_mb": round(model_size_mb(ternary_path), 4),
            "sparsity":      round(ternary_sparsity(ternary), 4),
        }
        print(f"Ternary final test accuracy: {final_ternary_acc*100:.2f}%")

    # -----------------------------------------------------------------------
    # 3. Save results.json
    # -----------------------------------------------------------------------
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    saved_paths = [results_path]
    if run_baseline:
        saved_paths.insert(0, baseline_path)
    if run_ternary:
        saved_paths.insert(0, npz_path)
        saved_paths.insert(0, ternary_path)
    print(f"\nSaved: {', '.join(saved_paths)}")


if __name__ == "__main__":
    main()
