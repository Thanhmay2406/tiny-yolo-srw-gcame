from __future__ import annotations

import torch

from src.losses.background_suppression import background_suppression_loss, create_ignore_mask_from_bbox_mask
from src.losses.energy_in_box import energy_in_box_loss
from src.losses.size_aware import bbox_area_weights, image_level_size_weight


def test_energy_in_box_is_lower_when_saliency_is_inside_bbox() -> None:
    bbox_mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    bbox_mask[:, :, 2:6, 2:6] = 1.0
    inside = torch.zeros_like(bbox_mask)
    inside[:, :, 2:6, 2:6] = 1.0
    outside = torch.zeros_like(bbox_mask)
    outside[:, :, 0:2, 0:2] = 1.0
    assert energy_in_box_loss(inside, bbox_mask) < energy_in_box_loss(outside, bbox_mask)


def test_background_suppression_penalizes_saliency_outside_ignore_region() -> None:
    bbox_mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    bbox_mask[:, :, 3:5, 3:5] = 1.0
    mostly_inside = torch.zeros_like(bbox_mask)
    mostly_inside[:, :, 3:5, 3:5] = 1.0
    outside = torch.zeros_like(bbox_mask)
    outside[:, :, 0:2, 0:2] = 1.0
    assert background_suppression_loss(mostly_inside, bbox_mask, dilation_radius=1) < background_suppression_loss(
        outside,
        bbox_mask,
        dilation_radius=1,
    )


def test_ignore_mask_dilation_expands_bbox_region() -> None:
    bbox_mask = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    bbox_mask[:, :, 3:4, 3:4] = 1.0
    ignore_mask = create_ignore_mask_from_bbox_mask(bbox_mask, dilation_radius=1)
    assert float(ignore_mask.sum()) > float(bbox_mask.sum())


def test_smaller_boxes_receive_higher_size_weights() -> None:
    bboxes = torch.tensor(
        [
            [0.5, 0.5, 0.05, 0.05],
            [0.5, 0.5, 0.25, 0.25],
        ],
        dtype=torch.float32,
    )
    weights = bbox_area_weights(bboxes, mode="log_inverse")
    assert weights[0] > weights[1]


def test_image_level_size_weight_handles_empty_images() -> None:
    batch_idx = torch.tensor([0, 0], dtype=torch.int64)
    bboxes = torch.tensor(
        [
            [0.5, 0.5, 0.05, 0.05],
            [0.5, 0.5, 0.08, 0.08],
        ],
        dtype=torch.float32,
    )
    weights = image_level_size_weight(batch_idx, bboxes, batch_size=3, mode="inverse_sqrt", max_weight=5.0)
    assert weights.shape == (3,)
    assert weights[0] > 1.0
    assert weights[1] == 1.0
    assert weights[2] == 1.0
