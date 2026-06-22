from __future__ import annotations

import numpy as np

from src.evaluation.small_object_metrics import compute_iou_matrix, recall_by_size_bucket


def test_compute_iou_matrix_handles_overlap() -> None:
    boxes_a = np.asarray([[0, 0, 10, 10]], dtype=np.float32)
    boxes_b = np.asarray([[5, 5, 15, 15]], dtype=np.float32)
    iou = compute_iou_matrix(boxes_a, boxes_b)
    assert iou.shape == (1, 1)
    assert 0.0 < float(iou[0, 0]) < 1.0


def test_recall_by_size_bucket_matches_boxes_by_class_and_iou() -> None:
    predictions = [
        {
            "image_id": "img1",
            "boxes": np.asarray([[0, 0, 3, 3], [10, 10, 18, 18], [20, 20, 60, 60]], dtype=np.float32),
            "scores": np.asarray([0.95, 0.9, 0.8], dtype=np.float32),
            "classes": np.asarray([0, 1, 2], dtype=np.int32),
        }
    ]
    ground_truths = [
        {
            "image_id": "img1",
            "boxes": np.asarray([[0, 0, 3, 3], [10, 10, 18, 18], [20, 20, 60, 60]], dtype=np.float32),
            "classes": np.asarray([0, 1, 2], dtype=np.int32),
            "image_size": (100, 100),
        }
    ]
    metrics = recall_by_size_bucket(predictions=predictions, ground_truths=ground_truths, iou_threshold=0.5)
    assert metrics["counts"]["tiny"] == 1
    assert metrics["counts"]["small"] == 1
    assert metrics["counts"]["medium_large"] == 1
    assert metrics["recall_tiny"] == 1.0
    assert metrics["recall_small"] == 1.0
    assert metrics["recall_medium_large"] == 1.0
