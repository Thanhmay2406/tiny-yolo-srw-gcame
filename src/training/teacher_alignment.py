from __future__ import annotations

from dataclasses import dataclass
from typing import Any


INCOMPATIBLE_TEACHER_AUGMENTATIONS: dict[str, Any] = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
}


@dataclass(frozen=True)
class TeacherAugmentationAudit:
    applied_policy: str
    incompatible_keys: list[str]
    disabled_keys: list[str]


def detect_incompatible_teacher_augmentations(args: Any) -> list[str]:
    incompatible: list[str] = []
    for key, safe_value in INCOMPATIBLE_TEACHER_AUGMENTATIONS.items():
        value = getattr(args, key, safe_value)
        if float(value) != float(safe_value):
            incompatible.append(key)
    return incompatible


def enforce_teacher_augmentation_policy(args: Any, policy: str) -> TeacherAugmentationAudit:
    normalized = str(policy).strip().lower()
    incompatible = detect_incompatible_teacher_augmentations(args)

    if not incompatible:
        return TeacherAugmentationAudit(
            applied_policy=normalized,
            incompatible_keys=[],
            disabled_keys=[],
        )

    if normalized == "error":
        raise ValueError(
            "offline_xai_teacher is incompatible with active geometric/cross-image augmentations: "
            f"{', '.join(incompatible)}. "
            "Use --teacher-augmentation-policy disable_incompatible to force a teacher-safe run."
        )

    if normalized != "disable_incompatible":
        raise ValueError(
            f"Unsupported teacher augmentation policy: {policy}. "
            "Expected one of: error, disable_incompatible."
        )

    disabled: list[str] = []
    for key in incompatible:
        setattr(args, key, INCOMPATIBLE_TEACHER_AUGMENTATIONS[key])
        disabled.append(key)

    return TeacherAugmentationAudit(
        applied_policy=normalized,
        incompatible_keys=incompatible,
        disabled_keys=disabled,
    )


def snapshot_teacher_augmentation_values(args: Any) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for key, safe_value in INCOMPATIBLE_TEACHER_AUGMENTATIONS.items():
        value = getattr(args, key, safe_value)
        snapshot[key] = float(value)
    return snapshot
