from __future__ import annotations

import numpy as np
import torch


def normalize_saliency_tensor(saliency: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if saliency.ndim != 4:
        raise ValueError(f"Expected saliency tensor [B,C,H,W], got shape {tuple(saliency.shape)}")

    flat = saliency.flatten(2)
    min_value = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    max_value = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    scale = (max_value - min_value).clamp_min(eps)
    return (saliency - min_value) / scale


def normalize_saliency_array(saliency: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    saliency = saliency.astype(np.float32)
    min_value = float(saliency.min(initial=0.0))
    max_value = float(saliency.max(initial=0.0))
    scale = max(max_value - min_value, eps)
    return np.clip((saliency - min_value) / scale, 0.0, 1.0)
