from __future__ import annotations

import numpy as np

from src.datasets.saliency_masks import (
    create_bbox_mask,
    create_gaussian_bbox_mask,
    resize_mask_to_feature,
    yolo_boxes_to_pixel_boxes,
)


def test_yolo_boxes_to_pixel_boxes_clips_and_preserves_min_size() -> None:
    boxes = np.asarray([[0, 0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    pixel_boxes = yolo_boxes_to_pixel_boxes(boxes, image_size=(100, 200))
    assert pixel_boxes.shape == (1, 4)
    assert pixel_boxes[0, 0] < pixel_boxes[0, 2]
    assert pixel_boxes[0, 1] < pixel_boxes[0, 3]


def test_create_bbox_mask_handles_multiple_boxes() -> None:
    boxes = np.asarray(
        [
            [0, 0.25, 0.25, 0.2, 0.2],
            [1, 0.75, 0.75, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    mask = create_bbox_mask(boxes, image_size=(100, 100))
    assert mask.shape == (100, 100)
    assert mask.max() == 1.0
    assert mask.sum() > 0


def test_create_gaussian_bbox_mask_empty_labels_returns_zeros() -> None:
    mask = create_gaussian_bbox_mask(np.zeros((0, 5), dtype=np.float32), image_size=(64, 64))
    assert mask.shape == (64, 64)
    assert float(mask.max()) == 0.0


def test_resize_mask_to_feature_preserves_range() -> None:
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:32, 16:32] = 1.0
    resized = resize_mask_to_feature(mask, feature_size=(8, 8))
    assert resized.shape == (8, 8)
    assert 0.0 <= float(resized.min()) <= float(resized.max()) <= 1.0
