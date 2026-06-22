from __future__ import annotations

import torch

from src.trainers.srw_lsal_trainer import SRWLSalDetectionLoss


def test_srw_lsal_loss_adds_teacher_term_only_when_enabled() -> None:
    loss = object.__new__(SRWLSalDetectionLoss)
    loss.loss_type = "mse"
    loss.saliency_loss_fn = lambda pred, target: torch.mean((pred - target) ** 2)
    loss.teacher_loss_fn = loss.saliency_loss_fn
    loss.lambda_sal = 0.5
    loss.beta_teacher = 0.25
    loss.size_aware = False

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


def test_srw_lsal_loss_aggregates_multiscale_saliency_with_scale_weights() -> None:
    loss = object.__new__(SRWLSalDetectionLoss)
    loss.loss_type = "mse"
    loss.saliency_loss_fn = lambda pred, target: torch.mean((pred - target) ** 2)
    loss.teacher_loss_fn = loss.saliency_loss_fn
    loss.lambda_sal = 0.5
    loss.beta_teacher = 0.0
    loss.size_aware = False
    loss.last_scale_losses = {}
    loss.last_teacher_scale_losses = {}

    preds = {
        "saliency_preds": {
            "P3": torch.ones((2, 1, 4, 4), dtype=torch.float32),
            "P4": torch.full((2, 1, 2, 2), 0.5, dtype=torch.float32),
        },
        "scale_weights": {"P3": 1.0, "P4": 0.25},
    }
    batch = {
        "gt_saliency_mask": torch.zeros((2, 1, 4, 4), dtype=torch.float32),
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

    expected_saliency = 1.0 + (0.25 * 0.25)
    assert torch.isclose(total_items[3], torch.tensor(expected_saliency, dtype=torch.float32))
    assert torch.isclose(detached_items[3], torch.tensor(expected_saliency * 0.5, dtype=torch.float32))
    assert loss.last_scale_losses == {"P3": 1.0, "P4": 0.25}
