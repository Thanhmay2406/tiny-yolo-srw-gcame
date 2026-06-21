from __future__ import annotations

import torch


def bbox_area_weights(
    bboxes: torch.Tensor,
    mode: str = "log_inverse",
    eps: float = 1e-6,
    max_weight: float | None = None,
) -> torch.Tensor:
    if bboxes.numel() == 0:
        return bboxes.new_zeros((0,))
    if bboxes.ndim != 2 or bboxes.shape[1] < 4:
        raise ValueError("Expected bboxes with shape [N,4] or [N,>=4].")
    wh = bboxes[:, -2:].float().clamp_min(eps)
    area = (wh[:, 0] * wh[:, 1]).clamp_min(eps)
    if mode == "log_inverse":
        weights = torch.log1p(1.0 / area)
    elif mode == "inverse_sqrt":
        weights = torch.rsqrt(area)
    else:
        raise ValueError(f"Unsupported size weight mode: {mode}")
    if max_weight is not None:
        weights = weights.clamp(max=float(max_weight))
    return weights


def image_level_size_weight(
    batch_idx: torch.Tensor,
    bboxes: torch.Tensor,
    batch_size: int,
    mode: str = "log_inverse",
    eps: float = 1e-6,
    max_weight: float | None = None,
) -> torch.Tensor:
    device = bboxes.device if isinstance(bboxes, torch.Tensor) else None
    weights = torch.ones((batch_size,), dtype=torch.float32, device=device)
    if bboxes.numel() == 0:
        return weights
    box_weights = bbox_area_weights(bboxes, mode=mode, eps=eps, max_weight=max_weight)
    flat_batch_idx = batch_idx.view(-1)
    for image_index in range(batch_size):
        selected = box_weights[flat_batch_idx == image_index]
        if selected.numel() > 0:
            weights[image_index] = selected.mean()
    return weights
