from __future__ import annotations

import torch
from torch import nn


class SaliencyHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or max(in_channels // 2, 16)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError(f"Expected feature map [B,C,H,W], got shape {tuple(feature_map.shape)}")
        return self.layers(feature_map)
