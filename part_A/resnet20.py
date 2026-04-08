"""ResNet-20 architecture for CIFAR-10 used in Part A."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ternary_layer import TernaryConv2d


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, conv_cls=nn.Conv2d):
        super().__init__()
        self.conv1 = conv_cls(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_cls(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Projection shortcut stays full precision.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10 (He et al. 2016)."""

    def __init__(self, conv_cls=nn.Conv2d, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(conv_cls, 16, 16, blocks=3, stride=1)
        self.layer2 = self._make_layer(conv_cls, 16, 32, blocks=3, stride=2)
        self.layer3 = self._make_layer(conv_cls, 32, 64, blocks=3, stride=2)

        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(conv_cls, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride, conv_cls=conv_cls)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1, conv_cls=conv_cls))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, TernaryConv2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)
