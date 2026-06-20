from __future__ import annotations

import torch
from torch import nn

from src.xai.saliency_normalization import normalize_saliency_tensor


class GradCAMLikeDetector(nn.Module):
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError(f"Expected feature map [B,C,H,W], got shape {tuple(feature_map.shape)}")

        working = feature_map
        if not working.requires_grad:
            working = working.clone().detach().requires_grad_(True)

        objective = working.square().mean()
        gradients = torch.autograd.grad(objective, working, retain_graph=True, create_graph=False)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        saliency = torch.relu((weights * working).sum(dim=1, keepdim=True))
        return normalize_saliency_tensor(saliency)
