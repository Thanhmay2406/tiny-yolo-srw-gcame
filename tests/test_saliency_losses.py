from __future__ import annotations

import torch

from src.losses.saliency_alignment import bce_saliency_loss, dice_saliency_loss, mse_saliency_loss


def test_saliency_losses_return_finite_scalars() -> None:
    pred = torch.full((2, 1, 8, 8), 0.5)
    target = torch.zeros((2, 1, 8, 8))
    for fn in (mse_saliency_loss, bce_saliency_loss, dice_saliency_loss):
        value = fn(pred, target)
        assert value.ndim == 0
        assert torch.isfinite(value)
