#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.saliency_masks import read_yolo_label_file, yolo_boxes_to_pixel_boxes
from src.datasets.yolo_dataset import list_split_samples
from src.evaluation.error_analysis import classify_error_case, resolve_run_dir
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logging
from src.utils.runtime import configure_runtime_environment

configure_runtime_environment()

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    default_data = os.environ.get("SKYFUSION_DATA") or (
        "/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml"
    )
    parser = argparse.ArgumentParser(description="Compare baseline and candidate YOLO runs with lightweight error analysis.")
    parser.add_argument("--dataset-yaml", type=Path, default=Path(default_data), help="Path to YOLO dataset YAML.")
    parser.add_argument("--baseline-run", type=str, required=True, help="Baseline run name or path.")
    parser.add_argument("--candidate-run", type=str, required=True, help="Candidate run name or path.")
    parser.add_argument("--split", type=str, default="val", choices=("train", "valid", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=128, help="Maximum number of images to analyze.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold used in case classification.")
    parser.add_argument("--device", type=str, default=None, help="Optional Ultralytics device string.")
    parser.add_argument("--save-images", type=int, default=16, help="How many comparison images to save.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def load_model_from_run(run_dir: Path) -> YOLO:
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    return YOLO(str(weights_path))


def predict_sample(model: YOLO, image_path: Path, imgsz: int, conf: float, device: str | None) -> dict[str, Any]:
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)
    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return {"boxes": np.zeros((0, 4), dtype=np.float32), "scores": np.zeros((0,), dtype=np.float32), "classes": np.zeros((0,), dtype=np.int32)}
    return {
        "boxes": boxes.xyxy.detach().cpu().numpy().astype(np.float32),
        "scores": boxes.conf.detach().cpu().numpy().astype(np.float32),
        "classes": boxes.cls.detach().cpu().numpy().astype(np.int32),
    }


def draw_boxes(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray | None, color: tuple[int, int, int], title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    for index, box in enumerate(np.asarray(boxes, dtype=np.float32)):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"cls={int(classes[index])}"
        if scores is not None and index < len(scores):
            label += f" {float(scores[index]):.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def maybe_find_xai_overlay(run_dir: Path, image_id: str) -> Path | None:
    xai_dir = run_dir / "xai_eval"
    if not xai_dir.is_dir():
        return None
    stem = Path(image_id).stem
    matches = sorted(xai_dir.glob(f"*_{stem}_*_overlay.jpg"))
    return matches[0] if matches else None


def save_comparison_image(
    output_dir: Path,
    sample_index: int,
    image_id: str,
    image_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    baseline_predictions: dict[str, Any],
    candidate_predictions: dict[str, Any],
    candidate_xai_overlay: Path | None,
) -> None:
    gt_panel = draw_boxes(image_bgr, gt_boxes, gt_classes, None, (0, 255, 0), "Ground Truth")
    baseline_panel = draw_boxes(
        image_bgr,
        baseline_predictions["boxes"],
        baseline_predictions["classes"],
        baseline_predictions["scores"],
        (255, 0, 0),
        "Baseline",
    )
    candidate_panel = draw_boxes(
        image_bgr,
        candidate_predictions["boxes"],
        candidate_predictions["classes"],
        candidate_predictions["scores"],
        (0, 0, 255),
        "Candidate",
    )
    panels = [gt_panel, baseline_panel, candidate_panel]
    if candidate_xai_overlay is not None:
        overlay = cv2.imread(str(candidate_xai_overlay), cv2.IMREAD_COLOR)
        if overlay is not None:
            overlay = cv2.resize(overlay, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.putText(overlay, "Candidate XAI", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            panels.append(overlay)
    montage = np.concatenate(panels, axis=1)
    safe_stem = Path(image_id).stem
    cv2.imwrite(str(output_dir / f"{sample_index:03d}_{safe_stem}_comparison.jpg"), montage)


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    data_yaml = args.dataset_yaml.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    baseline_run_dir = resolve_run_dir(args.baseline_run)
    candidate_run_dir = resolve_run_dir(args.candidate_run)
    if not baseline_run_dir.is_dir():
        raise SystemExit(f"Baseline run directory not found: {baseline_run_dir}")
    if not candidate_run_dir.is_dir():
        raise SystemExit(f"Candidate run directory not found: {candidate_run_dir}")

    output_target = args.output_dir.expanduser()
    output_dir = ensure_dir(output_target if output_target.is_absolute() else (Path.cwd() / output_target).resolve())
    image_output_dir = ensure_dir(output_dir / "images")

    baseline_model = load_model_from_run(baseline_run_dir)
    candidate_model = load_model_from_run(candidate_run_dir)
    samples = list_split_samples(data_yaml, split=args.split, limit=args.max_samples)
    if not samples:
        save_json(
            output_dir / "summary.json",
            {
                "baseline_run": str(baseline_run_dir),
                "candidate_run": str(candidate_run_dir),
                "split": args.split,
                "num_samples": 0,
                "warning": "No samples found for the requested split.",
            },
        )
        logger.warning("No samples found for split '%s'.", args.split)
        return

    rows: list[dict[str, Any]] = []
    saved_images = 0
    for sample_index, sample in enumerate(samples):
        image_bgr = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            logger.warning("Skipping unreadable image: %s", sample["image_path"])
            continue
        height, width = image_bgr.shape[:2]
        labels = read_yolo_label_file(sample["label_path"])
        gt_boxes = yolo_boxes_to_pixel_boxes(labels, image_size=(height, width))
        gt_classes = labels[:, 0].astype(np.int32) if labels.size > 0 else np.zeros((0,), dtype=np.int32)

        baseline_predictions = predict_sample(baseline_model, Path(sample["image_path"]), args.imgsz, args.conf, args.device)
        candidate_predictions = predict_sample(candidate_model, Path(sample["image_path"]), args.imgsz, args.conf, args.device)
        case = classify_error_case(
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            image_size=(height, width),
            baseline_predictions=baseline_predictions,
            candidate_predictions=candidate_predictions,
            iou_threshold=args.iou,
        )
        row = {
            "image_id": sample["image_id"],
            "num_gt": int(len(gt_boxes)),
            "baseline_num_pred": int(len(baseline_predictions["boxes"])),
            "candidate_num_pred": int(len(candidate_predictions["boxes"])),
            **case,
        }
        rows.append(row)

        should_save = saved_images < args.save_images and (
            case["baseline_wrong_candidate_correct"]
            or case["baseline_correct_candidate_wrong"]
            or case["tiny_object_missed"]
            or case["candidate_localization_error"]
            or case["candidate_false_positive_near_background"]
        )
        if should_save:
            save_comparison_image(
                image_output_dir,
                saved_images,
                str(sample["image_id"]),
                image_bgr,
                gt_boxes,
                gt_classes,
                baseline_predictions,
                candidate_predictions,
                maybe_find_xai_overlay(candidate_run_dir, str(sample["image_id"])),
            )
            saved_images += 1

    csv_path = output_dir / "error_analysis.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["image_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "baseline_run": str(baseline_run_dir),
        "candidate_run": str(candidate_run_dir),
        "split": args.split,
        "num_samples": len(rows),
        "saved_images": saved_images,
        "counts": {
            key: int(sum(bool(row.get(key)) for row in rows))
            for key in [
                "baseline_wrong_candidate_correct",
                "baseline_correct_candidate_wrong",
                "both_wrong",
                "tiny_object_missed",
                "baseline_false_positive_near_background",
                "candidate_false_positive_near_background",
                "baseline_localization_error",
                "candidate_localization_error",
            ]
        },
        "todo": [
            "Current categorization is sample-level and intentionally lightweight.",
            "Per-instance root-cause clustering can be added later if more detailed analysis is needed.",
            "Candidate saliency is visualized only when a matching overlay already exists in candidate_run/xai_eval.",
        ],
    }
    save_json(output_dir / "summary.json", summary)
    logger.info("Error analysis finished. CSV: %s", csv_path)


if __name__ == "__main__":
    main()
