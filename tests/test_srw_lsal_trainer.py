from __future__ import annotations

import torch

from src.trainers.srw_lsal_trainer import SRWLSalDetectionLoss


def test_srw_lsal_loss_adds_teacher_term_only_when_enabled() -> None:
    loss = object.__new__(SRWLSalDetectionLoss)
    loss.saliency_loss_fn = lambda pred, target: torch.mean((pred - target) ** 2)
    loss.lambda_sal = 0.5
    loss.beta_teacher = 0.25

    preds = {"saliency_pred": torch.ones((2, 1, 4, 4), dtype=torch.float32)}
    batch = {
        "gt_saliency_mask": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        "teacher_saliency_mask": torch.full((2, 1, 4, 4), 0.5, dtype=torch.float32),
    }

    original_loss = SRWLSalDetectionLoss.__mro__[1].loss
    try:
        SRWLSalDetectionLoss.__mro__[1].loss = lambda self, preds, batch: (
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        )
        total_items, detached_items = SRWLSalDetectionLoss.loss(loss, preds, batch)
    finally:
        SRWLSalDetectionLoss.__mro__[1].loss = original_loss

    assert total_items.shape[0] == 5
    assert detached_items.shape[0] == 5
    assert torch.isclose(total_items[3], torch.tensor(1.0))
    assert torch.isclose(total_items[4], torch.tensor(0.125))
    assert torch.isclose(detached_items[3], torch.tensor(0.5))
    assert torch.isclose(detached_items[4], torch.tensor(0.0625))
