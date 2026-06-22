from __future__ import annotations

import numpy as np

from src.evaluation.error_analysis import classify_error_case


def test_classify_error_case_detects_candidate_recovery_and_tiny_miss() -> None:
    gt_boxes = np.asarray([[0, 0, 10, 10], [20, 20, 23, 23]], dtype=np.float32)
    gt_classes = np.asarray([0, 0], dtype=np.int32)

    baseline_predictions = {
        "boxes": np.asarray([[40, 40, 50, 50]], dtype=np.float32),
        "scores": np.asarray([0.9], dtype=np.float32),
        "classes": np.asarray([0], dtype=np.int32),
    }
    candidate_predictions = {
        "boxes": np.asarray([[0, 0, 10, 10], [60, 60, 70, 70]], dtype=np.float32),
        "scores": np.asarray([0.95, 0.5], dtype=np.float32),
        "classes": np.asarray([0, 0], dtype=np.int32),
    }

    output = classify_error_case(
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        image_size=(100, 100),
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        iou_threshold=0.5,
    )

    assert output["baseline_wrong_candidate_correct"] is True
    assert output["tiny_object_missed"] is True
    assert output["candidate_false_positive_near_background"] is True


def test_classify_error_case_detects_localization_error() -> None:
    gt_boxes = np.asarray([[0, 0, 10, 10]], dtype=np.float32)
    gt_classes = np.asarray([0], dtype=np.int32)
    candidate_predictions = {
        "boxes": np.asarray([[2, 2, 8, 8]], dtype=np.float32),
        "scores": np.asarray([0.8], dtype=np.float32),
        "classes": np.asarray([0], dtype=np.int32),
    }

    output = classify_error_case(
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        image_size=(100, 100),
        baseline_predictions={"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "classes": np.zeros((0,))},
        candidate_predictions=candidate_predictions,
        iou_threshold=0.5,
    )

    assert output["candidate_localization_error"] is True
