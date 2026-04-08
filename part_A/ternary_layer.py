"""Ternary convolution layer for Part A."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryConv2d(nn.Conv2d):
    """Drop-in replacement for nn.Conv2d with ternary weights.

    Forward: uses W_t * alpha where W_t in {-1, 0, +1} and alpha = mean(|W|).
    Backward: Straight-Through Estimator where gradient flows as identity.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.weight.abs().mean()
        # Quantize to ternary values for forward pass.
        w_t = (self.weight / (alpha + 1e-8)).round().clamp(-1.0, 1.0)
        # STE: keep quantized value in forward, identity gradient in backward.
        w_ternary = self.weight + (w_t * alpha - self.weight).detach()
        return F.conv2d(
            x,
            w_ternary,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
