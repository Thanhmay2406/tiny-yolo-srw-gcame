from __future__ import annotations

import torch


def energy_in_box_loss(
    saliency: torch.Tensor,
    bbox_mask: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
    image_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if saliency.ndim != 4 or bbox_mask.ndim != 4:
        raise ValueError("Expected saliency and bbox_mask to have shape [B,1,H,W].")
    if tuple(saliency.shape) != tuple(bbox_mask.shape):
        raise ValueError("saliency and bbox_mask must have identical shapes.")

    saliency = saliency.float()
    bbox_mask = bbox_mask.float()
    total_energy = saliency.sum(dim=(1, 2, 3))
    in_box_energy = (saliency * bbox_mask).sum(dim=(1, 2, 3))
    has_box = bbox_mask.sum(dim=(1, 2, 3)) > 0
    per_image = 1.0 - (in_box_energy / (total_energy + eps))
    per_image = torch.where(has_box, per_image, torch.zeros_like(per_image))
    if image_weights is not None:
        per_image = per_image * image_weights.to(device=per_image.device, dtype=per_image.dtype)
    if reduction == "none":
        return per_image
    if reduction == "mean":
        return per_image.mean()
    if reduction == "sum":
        return per_image.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")
