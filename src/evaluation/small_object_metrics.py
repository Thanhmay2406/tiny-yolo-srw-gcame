from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SizeBucketThresholds:
    tiny_max_area_ratio: float = 0.001
    small_max_area_ratio: float = 0.01


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = wh[..., 0] * wh[..., 1]

    area_a = np.clip(a[:, 2] - a[:, 0], a_min=0.0, a_max=None) * np.clip(a[:, 3] - a[:, 1], a_min=0.0, a_max=None)
    area_b = np.clip(b[:, 2] - b[:, 0], a_min=0.0, a_max=None) * np.clip(b[:, 3] - b[:, 1], a_min=0.0, a_max=None)
    union = np.clip(area_a[:, None] + area_b[None, :] - intersection, a_min=1e-9, a_max=None)
    return intersection / union


def box_area_ratios(boxes_xyxy: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    image_area = max(float(height * width), 1.0)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None
    )
    return areas / image_area


def classify_size_buckets(
    boxes_xyxy: np.ndarray,
    image_size: tuple[int, int],
    thresholds: SizeBucketThresholds | None = None,
) -> list[str]:
    limits = thresholds or SizeBucketThresholds()
    ratios = box_area_ratios(boxes_xyxy, image_size=image_size)
    labels: list[str] = []
    for ratio in ratios:
        if ratio <= limits.tiny_max_area_ratio:
            labels.append("tiny")
        elif ratio <= limits.small_max_area_ratio:
            labels.append("small")
        else:
            labels.append("medium_large")
    return labels


def _greedy_gt_matches(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    matched_gt = np.zeros((len(gt_boxes),), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched_gt

    order = np.argsort(-pred_scores.astype(np.float32))
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    for pred_index in order:
        best_gt = -1
        best_iou = float(iou_threshold)
        pred_class = int(pred_classes[pred_index])
        for gt_index in range(len(gt_boxes)):
            if matched_gt[gt_index]:
                continue
            if pred_class != int(gt_classes[gt_index]):
                continue
            score = float(iou_matrix[pred_index, gt_index])
            if score >= best_iou:
                best_iou = score
                best_gt = gt_index
        if best_gt >= 0:
            matched_gt[best_gt] = True
    return matched_gt


def recall_by_size_bucket(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    thresholds: SizeBucketThresholds | None = None,
) -> dict[str, Any]:
    gt_by_image = {str(item["image_id"]): item for item in ground_truths}
    counts = {"tiny": 0, "small": 0, "medium_large": 0}
    matched = {"tiny": 0, "small": 0, "medium_large": 0}

    for prediction in predictions:
        image_id = str(prediction["image_id"])
        gt_item = gt_by_image.get(image_id)
        if gt_item is None:
            continue

        gt_boxes = np.asarray(gt_item["boxes"], dtype=np.float32)
        gt_classes = np.asarray(gt_item["classes"], dtype=np.int32)
        pred_boxes = np.asarray(prediction.get("boxes", np.zeros((0, 4))), dtype=np.float32)
        pred_scores = np.asarray(prediction.get("scores", np.zeros((0,))), dtype=np.float32)
        pred_classes = np.asarray(prediction.get("classes", np.zeros((0,))), dtype=np.int32)

        bucket_labels = classify_size_buckets(gt_boxes, image_size=tuple(gt_item["image_size"]), thresholds=thresholds)
        gt_matches = _greedy_gt_matches(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_classes=pred_classes,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            iou_threshold=float(iou_threshold),
        )
        for gt_index, bucket in enumerate(bucket_labels):
            counts[bucket] += 1
            if gt_matches[gt_index]:
                matched[bucket] += 1

    output: dict[str, Any] = {
        "iou_threshold": float(iou_threshold),
        "tiny_max_area_ratio": float((thresholds or SizeBucketThresholds()).tiny_max_area_ratio),
        "small_max_area_ratio": float((thresholds or SizeBucketThresholds()).small_max_area_ratio),
        "counts": counts,
        "matched": matched,
    }
    for bucket in ("tiny", "small", "medium_large"):
        denom = counts[bucket]
        output[f"recall_{bucket}"] = float(matched[bucket] / denom) if denom > 0 else None
    return output
