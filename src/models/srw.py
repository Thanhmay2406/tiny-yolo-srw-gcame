from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SRWModule(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        alpha_init: float = 0.1,
        learnable_alpha: bool = True,
        spatial_expand_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_expand_mode = spatial_expand_mode
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(
        self,
        feature_map: torch.Tensor,
        saliency_map: torch.Tensor,
        return_gates: bool = False,
    ):
        if feature_map.ndim != 4:
            raise ValueError(f"Expected feature map [B,C,H,W], got shape {tuple(feature_map.shape)}")
        if saliency_map.ndim == 3:
            saliency_map = saliency_map.unsqueeze(1)
        if saliency_map.ndim != 4 or saliency_map.shape[1] != 1:
            raise ValueError(f"Expected saliency map [B,1,H,W], got shape {tuple(saliency_map.shape)}")

        if tuple(saliency_map.shape[-2:]) != tuple(feature_map.shape[-2:]):
            saliency_map = F.interpolate(
                saliency_map,
                size=feature_map.shape[-2:],
                mode=self.spatial_expand_mode,
                align_corners=False if self.spatial_expand_mode in {"bilinear", "bicubic"} else None,
            )

        gate_s = self.spatial_gate(saliency_map)
        pooled_feature = feature_map.mean(dim=(2, 3))
        pooled_saliency_feature = (feature_map * saliency_map).mean(dim=(2, 3))
        gate_c = self.channel_mlp(torch.cat([pooled_feature, pooled_saliency_feature], dim=1)).unsqueeze(-1).unsqueeze(-1)

        modulated = feature_map * gate_s * gate_c
        output = feature_map + self.alpha * modulated

        if return_gates:
            return output, gate_s, gate_c, self.alpha
        return output
