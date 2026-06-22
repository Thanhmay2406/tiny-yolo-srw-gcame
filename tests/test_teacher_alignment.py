from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.training.teacher_alignment import (
    detect_incompatible_teacher_augmentations,
    enforce_teacher_augmentation_policy,
    snapshot_teacher_augmentation_values,
)


def build_args(**overrides):
    defaults = {
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_detect_incompatible_teacher_augmentations_finds_geometric_and_cross_image_ops() -> None:
    args = build_args()
    found = detect_incompatible_teacher_augmentations(args)
    assert found == ["mosaic", "mixup", "translate", "scale", "fliplr"]


def test_enforce_teacher_augmentation_policy_errors_by_default() -> None:
    args = build_args()
    with pytest.raises(ValueError, match="offline_xai_teacher is incompatible"):
        enforce_teacher_augmentation_policy(args, policy="error")


def test_enforce_teacher_augmentation_policy_can_disable_incompatible_ops() -> None:
    args = build_args()
    audit = enforce_teacher_augmentation_policy(args, policy="disable_incompatible")
    assert audit.disabled_keys == ["mosaic", "mixup", "translate", "scale", "fliplr"]
    assert args.mosaic == 0.0
    assert args.mixup == 0.0
    assert args.translate == 0.0
    assert args.scale == 0.0
    assert args.fliplr == 0.0


def test_enforce_teacher_augmentation_policy_noops_when_already_safe() -> None:
    args = build_args(mosaic=0.0, mixup=0.0, translate=0.0, scale=0.0, fliplr=0.0)
    audit = enforce_teacher_augmentation_policy(args, policy="disable_incompatible")
    assert audit.incompatible_keys == []
    assert audit.disabled_keys == []


def test_snapshot_teacher_augmentation_values_reflects_effective_state() -> None:
    args = build_args()
    enforce_teacher_augmentation_policy(args, policy="disable_incompatible")
    snapshot = snapshot_teacher_augmentation_values(args)
    assert snapshot["mosaic"] == 0.0
    assert snapshot["mixup"] == 0.0
    assert snapshot["translate"] == 0.0
    assert snapshot["scale"] == 0.0
    assert snapshot["fliplr"] == 0.0
