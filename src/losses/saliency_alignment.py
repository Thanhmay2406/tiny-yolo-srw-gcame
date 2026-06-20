from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_saliency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def bce_saliency_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(min=eps, max=1.0 - eps)
    return F.binary_cross_entropy(pred, target)


def dice_saliency_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return (1.0 - dice).mean()


def get_saliency_loss(loss_type: str):
    normalized = loss_type.lower()
    if normalized == "mse":
        return mse_saliency_loss
    if normalized == "bce":
        return bce_saliency_loss
    if normalized == "dice":
        return dice_saliency_loss
    raise ValueError(f"Unsupported saliency loss type: {loss_type}")
