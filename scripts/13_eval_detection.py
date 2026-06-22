#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.saliency_masks import read_yolo_label_file, yolo_boxes_to_pixel_boxes
from src.datasets.yolo_dataset import list_split_samples
from src.evaluation.small_object_metrics import recall_by_size_bucket
from src.utils.io import ensure_dir
from src.utils.logging import save_run_metrics, setup_logging
from src.utils.runtime import configure_runtime_environment
from src.utils.train_runs import normalize_metric_value

configure_runtime_environment()

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    default_data = os.environ.get("SKYFUSION_DATA") or (
        "/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml"
    )
    parser = argparse.ArgumentParser(description="Evaluate detection metrics for a trained YOLO checkpoint.")
    parser.add_argument("--data", type=Path, default=Path(default_data), help="Path to YOLO dataset YAML.")
    parser.add_argument("--weights", type=str, required=True, help="Trained weights path.")
    parser.add_argument("--split", type=str, default="valid", choices=("train", "valid", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640, help="Validation/prediction image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help="Ultralytics device string.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for custom recall buckets.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name under experiments/skyfusion.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/skyfusion"), help="Output root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional explicit output directory.")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        target = args.output_dir.expanduser()
        return ensure_dir(target if target.is_absolute() else (Path.cwd() / target).resolve())
    if args.run_name:
        return ensure_dir((args.output_root.expanduser() / args.run_name / "detection_eval").resolve())
    weights_path = Path(args.weights).expanduser()
    if weights_path.is_file() and weights_path.parent.name == "weights":
        return ensure_dir(weights_path.parent.parent / "detection_eval")
    return ensure_dir((args.output_root.expanduser() / "detection_eval").resolve())


def build_ground_truth(data_yaml: Path, split: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sample in list_split_samples(data_yaml, split=split):
        labels = read_yolo_label_file(sample["label_path"])
        import cv2

        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            raise RuntimeError(f"Failed to read image: {sample['image_path']}")
        height, width = image.shape[:2]
        boxes_xyxy = yolo_boxes_to_pixel_boxes(labels, image_size=(height, width))
        classes = labels[:, 0].astype(int) if labels.size > 0 else []
        items.append(
            {
                "image_id": sample["image_id"],
                "boxes": boxes_xyxy,
                "classes": classes,
                "image_size": (height, width),
            }
        )
    return items


def build_predictions(model: Any, data_yaml: Path, split: str, imgsz: int, conf: float, device: str | None) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample in list_split_samples(data_yaml, split=split):
        results = model.predict(
            source=str(sample["image_path"]),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        predictions.append(
            {
                "image_id": sample["image_id"],
                "boxes": boxes.xyxy.detach().cpu().numpy() if boxes is not None else [],
                "scores": boxes.conf.detach().cpu().numpy() if boxes is not None else [],
                "classes": boxes.cls.detach().cpu().numpy().astype(int) if boxes is not None else [],
            }
        )
    return predictions


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    data_yaml = args.data.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    output_dir = resolve_output_dir(args)
    logger.info("Detection evaluation output: %s", output_dir)

    model = YOLO(args.weights)
    ultralytics_split = "val" if args.split == "valid" else args.split
    val_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": ultralytics_split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)

    precision = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
    recall = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    gt_items = build_ground_truth(data_yaml, split=args.split)
    pred_items = build_predictions(model, data_yaml, split=args.split, imgsz=args.imgsz, conf=args.conf, device=args.device)
    size_metrics = recall_by_size_bucket(predictions=pred_items, ground_truths=gt_items, iou_threshold=args.iou)

    payload = {
        "weights": str(Path(args.weights).expanduser().resolve()),
        "data": str(data_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metrics": dict(metrics.results_dict),
        "f1": f1,
        "per_class_summary": metrics.summary(),
        "small_object_metrics": size_metrics,
    }
    save_run_metrics(output_dir, normalize_metric_value(payload))
    logger.info("Detection evaluation finished.")


if __name__ == "__main__":
    main()
