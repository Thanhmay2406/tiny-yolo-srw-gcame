from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.energy_in_box import energy_in_box_loss


def create_ignore_mask_from_bbox_mask(bbox_mask: torch.Tensor, dilation_radius: int = 3) -> torch.Tensor:
    if bbox_mask.ndim != 4:
        raise ValueError("Expected bbox_mask to have shape [B,1,H,W].")
    if dilation_radius <= 0:
        return bbox_mask.float().clamp(0.0, 1.0)
    kernel = dilation_radius * 2 + 1
    dilated = F.max_pool2d(bbox_mask.float(), kernel_size=kernel, stride=1, padding=dilation_radius)
    return dilated.clamp(0.0, 1.0)


def background_suppression_loss(
    saliency: torch.Tensor,
    bbox_mask: torch.Tensor,
    dilation_radius: int = 3,
    eps: float = 1e-6,
    reduction: str = "mean",
    image_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if tuple(saliency.shape) != tuple(bbox_mask.shape):
        raise ValueError("saliency and bbox_mask must have identical shapes.")
    ignore_mask = create_ignore_mask_from_bbox_mask(bbox_mask, dilation_radius=dilation_radius)
    total_energy = saliency.float().sum(dim=(1, 2, 3))
    background_energy = (saliency.float() * (1.0 - ignore_mask)).sum(dim=(1, 2, 3))
    has_box = bbox_mask.float().sum(dim=(1, 2, 3)) > 0
    per_image = background_energy / (total_energy + eps)
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


def combined_energy_bg_loss(
    saliency: torch.Tensor,
    bbox_mask: torch.Tensor,
    beta_bg: float = 0.5,
    dilation_radius: int = 3,
    eps: float = 1e-6,
    reduction: str = "mean",
    image_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    energy = energy_in_box_loss(
        saliency,
        bbox_mask,
        eps=eps,
        reduction="none",
        image_weights=image_weights,
    )
    background = background_suppression_loss(
        saliency,
        bbox_mask,
        dilation_radius=dilation_radius,
        eps=eps,
        reduction="none",
        image_weights=image_weights,
    )
    per_image = energy + float(beta_bg) * background
    if reduction == "none":
        return per_image
    if reduction == "mean":
        return per_image.mean()
    if reduction == "sum":
        return per_image.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")
