from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.small_object_metrics import SizeBucketThresholds, classify_size_buckets, compute_iou_matrix


def resolve_run_dir(run_reference: str | Path, output_root: str | Path = "experiments/skyfusion") -> Path:
    candidate = Path(run_reference).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (Path(output_root).expanduser() / str(run_reference)).resolve()


def _match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float,
) -> dict[str, Any]:
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    pred_classes = np.asarray(pred_classes, dtype=np.int32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    gt_classes = np.asarray(gt_classes, dtype=np.int32)

    matched_gt = np.zeros((len(gt_boxes),), dtype=bool)
    matched_pred = np.zeros((len(pred_boxes),), dtype=bool)
    best_iou_per_pred = np.zeros((len(pred_boxes),), dtype=np.float32)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {
            "matched_gt": matched_gt,
            "matched_pred": matched_pred,
            "best_iou_per_pred": best_iou_per_pred,
        }

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores.astype(np.float32))
    for pred_index in order:
        same_class = pred_classes[pred_index] == gt_classes
        if same_class.any():
            best_iou_per_pred[pred_index] = float(iou_matrix[pred_index, same_class].max())

        best_gt = -1
        best_iou = float(iou_threshold)
        for gt_index in range(len(gt_boxes)):
            if matched_gt[gt_index]:
                continue
            if int(pred_classes[pred_index]) != int(gt_classes[gt_index]):
                continue
            iou = float(iou_matrix[pred_index, gt_index])
            if iou >= best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_gt >= 0:
            matched_gt[best_gt] = True
            matched_pred[pred_index] = True
            best_iou_per_pred[pred_index] = best_iou
    return {
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "best_iou_per_pred": best_iou_per_pred,
    }


def classify_error_case(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    image_size: tuple[int, int],
    baseline_predictions: dict[str, Any],
    candidate_predictions: dict[str, Any],
    iou_threshold: float = 0.5,
    thresholds: SizeBucketThresholds | None = None,
) -> dict[str, Any]:
    baseline_match = _match_predictions(
        baseline_predictions.get("boxes", np.zeros((0, 4), dtype=np.float32)),
        baseline_predictions.get("scores", np.zeros((0,), dtype=np.float32)),
        baseline_predictions.get("classes", np.zeros((0,), dtype=np.int32)),
        gt_boxes,
        gt_classes,
        iou_threshold=iou_threshold,
    )
    candidate_match = _match_predictions(
        candidate_predictions.get("boxes", np.zeros((0, 4), dtype=np.float32)),
        candidate_predictions.get("scores", np.zeros((0,), dtype=np.float32)),
        candidate_predictions.get("classes", np.zeros((0,), dtype=np.int32)),
        gt_boxes,
        gt_classes,
        iou_threshold=iou_threshold,
    )

    gt_bucket_labels = classify_size_buckets(gt_boxes, image_size=image_size, thresholds=thresholds)
    baseline_tp = int(baseline_match["matched_gt"].sum())
    candidate_tp = int(candidate_match["matched_gt"].sum())
    tiny_missed = any(
        bucket == "tiny" and not bool(candidate_match["matched_gt"][index]) for index, bucket in enumerate(gt_bucket_labels)
    )

    baseline_best_iou = np.asarray(baseline_match["best_iou_per_pred"], dtype=np.float32)
    candidate_best_iou = np.asarray(candidate_match["best_iou_per_pred"], dtype=np.float32)
    baseline_unmatched = ~np.asarray(baseline_match["matched_pred"], dtype=bool)
    candidate_unmatched = ~np.asarray(candidate_match["matched_pred"], dtype=bool)

    return {
        "baseline_tp_count": baseline_tp,
        "candidate_tp_count": candidate_tp,
        "baseline_wrong_candidate_correct": baseline_tp == 0 and candidate_tp > 0,
        "baseline_correct_candidate_wrong": baseline_tp > 0 and candidate_tp == 0,
        "both_wrong": baseline_tp == 0 and candidate_tp == 0,
        "tiny_object_missed": tiny_missed,
        "baseline_false_positive_near_background": bool(np.any(baseline_unmatched & (baseline_best_iou < 0.1))),
        "candidate_false_positive_near_background": bool(np.any(candidate_unmatched & (candidate_best_iou < 0.1))),
        "baseline_localization_error": bool(
            np.any(baseline_unmatched & (baseline_best_iou >= 0.1) & (baseline_best_iou < iou_threshold))
        ),
        "candidate_localization_error": bool(
            np.any(candidate_unmatched & (candidate_best_iou >= 0.1) & (candidate_best_iou < iou_threshold))
        ),
    }
