#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.saliency_masks import read_yolo_label_file, yolo_boxes_to_pixel_boxes
from src.datasets.yolo_dataset import list_split_samples
from src.evaluation.xai_metrics import aggregate_xai_metrics, evaluate_xai_image
from src.models.layer_resolver import resolve_target_layer
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logging
from src.utils.runtime import configure_runtime_environment
from src.xai.hooks import ForwardCapture
from src.xai.saliency_provider import SaliencyProvider

configure_runtime_environment()

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    default_data = os.environ.get("SKYFUSION_DATA") or (
        "/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml"
    )
    parser = argparse.ArgumentParser(description="Evaluate saliency localization quality on a YOLO checkpoint.")
    parser.add_argument("--data", type=Path, default=Path(default_data), help="Path to YOLO dataset YAML.")
    parser.add_argument("--weights", type=str, required=True, help="Trained weights path.")
    parser.add_argument("--split", type=str, default="valid", choices=("train", "valid", "val", "test"))
    parser.add_argument("--target-layers", nargs="+", default=["P3"], help="One or more YOLO target layers.")
    parser.add_argument(
        "--saliency-mode",
        type=str,
        default="auto",
        choices=("auto", "saliency_head", "gradcam_like"),
        help="How to generate saliency during evaluation.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO forward image size.")
    parser.add_argument("--device", type=str, default=None, help="Torch device for direct model forward, e.g. cpu or cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--save-overlays", type=int, default=16, help="How many qualitative overlays to save.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def preprocess_image(image_bgr: np.ndarray, imgsz: int) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def overlay_mask(image_bgr: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(saliency, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.65, heat, 0.35, 0.0)


def resolve_saliency_callable(
    core_model: Any,
    layer_name: str,
    feature_channels: int,
    mode: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    normalized = str(mode).strip().lower()
    if normalized in {"auto", "saliency_head"} and hasattr(core_model, "saliency_providers"):
        providers = getattr(core_model, "saliency_providers")
        if layer_name in providers:
            provider = providers[layer_name]
            return lambda feature_map: provider(feature_map)
        if normalized == "saliency_head":
            raise ValueError(f"Trained saliency head for layer '{layer_name}' is not available in this checkpoint.")

    if normalized == "auto":
        normalized = "gradcam_like"
    if normalized == "gradcam_like":
        provider = SaliencyProvider(mode="gradcam_like_online_debug", channels=feature_channels)
        return lambda feature_map: provider(feature_map)

    raise ValueError(f"Unsupported saliency evaluation mode: {mode}")


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    output_target = args.output_dir.expanduser()
    output_dir = ensure_dir(output_target if output_target.is_absolute() else (Path.cwd() / output_target).resolve())

    data_yaml = args.data.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    weights_path = Path(args.weights).expanduser()
    if not weights_path.is_absolute():
        weights_path = (Path.cwd() / weights_path).resolve()
    if not weights_path.is_file():
        raise SystemExit(f"Checkpoint not found: {weights_path}")

    model = YOLO(str(weights_path))
    core_model = model.model
    if args.device is not None:
        core_model = core_model.to(torch.device(args.device))
    first_parameter = next(core_model.parameters(), None)
    model_device = first_parameter.device if first_parameter is not None else torch.device(args.device or "cpu")

    captures: dict[str, ForwardCapture] = {}
    layer_indices: dict[str, int] = {}
    try:
        for target_layer in args.target_layers:
            layer_name, layer_index = resolve_target_layer(core_model, target_layer)
            captures[layer_name] = ForwardCapture(core_model.model[layer_index])
            layer_indices[layer_name] = layer_index
    except Exception as exc:
        raise SystemExit(f"Failed to resolve requested target layers {args.target_layers}: {exc}") from exc

    providers: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    per_layer_records: dict[str, list[dict[str, Any]]] = {layer_name: [] for layer_name in captures}
    skipped_samples: list[dict[str, str]] = []
    samples = list_split_samples(data_yaml, split=args.split, limit=args.limit)
    if not samples:
        summary = {
            "weights": str(weights_path),
            "data": str(data_yaml),
            "split": args.split,
            "target_layers": list(per_layer_records),
            "saliency_mode": args.saliency_mode,
            "warning": f"No samples found for split '{args.split}'.",
            "num_skipped_samples": 0,
            "per_layer": {
                layer_name: {
                    "summary": aggregate_xai_metrics([]),
                    "per_image_path": f"{layer_name}_per_image.json",
                }
                for layer_name in per_layer_records
            },
        }
        for layer_name in per_layer_records:
            save_json(output_dir / f"{layer_name}_per_image.json", [])
        save_json(output_dir / "xai_metrics.json", summary)
        logger.warning("No samples found for split '%s'.", args.split)
        return

    try:
        for sample_index, sample in enumerate(samples):
            try:
                image_bgr = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Failed to read image: {sample['image_path']}")
                height, width = image_bgr.shape[:2]
                labels = read_yolo_label_file(sample["label_path"])
                gt_boxes = yolo_boxes_to_pixel_boxes(labels, image_size=(height, width))

                tensor = preprocess_image(image_bgr, imgsz=args.imgsz)
                tensor = tensor.to(device=model_device)
                for capture in captures.values():
                    capture.clear()
                _ = core_model(tensor)

                for layer_name, capture in captures.items():
                    feature_map = capture.output
                    if feature_map is None:
                        raise RuntimeError(f"Forward hook did not capture feature map for layer: {layer_name}")
                    if layer_name not in providers:
                        providers[layer_name] = resolve_saliency_callable(
                            core_model=core_model,
                            layer_name=layer_name,
                            feature_channels=int(feature_map.shape[1]),
                            mode=args.saliency_mode,
                        )
                    saliency = providers[layer_name](feature_map).detach().cpu().squeeze().numpy().astype(np.float32)
                    saliency = cv2.resize(saliency, (width, height), interpolation=cv2.INTER_LINEAR)
                    metrics = evaluate_xai_image(saliency, gt_boxes)
                    record = {
                        "image_id": sample["image_id"],
                        "layer_name": layer_name,
                        "feature_shape": list(feature_map.shape),
                        **metrics,
                    }
                    per_layer_records[layer_name].append(record)

                    if sample_index < args.save_overlays:
                        overlay = overlay_mask(image_bgr, saliency)
                        file_stem = f"{sample_index:03d}_{Path(str(sample['image_id'])).stem}_{layer_name}"
                        cv2.imwrite(str(output_dir / f"{file_stem}_overlay.jpg"), overlay)
            except Exception as exc:
                logger.warning("Skipping sample '%s' during XAI evaluation: %s", sample["image_id"], exc)
                skipped_samples.append({"image_id": str(sample["image_id"]), "reason": str(exc)})
    finally:
        for capture in captures.values():
            capture.remove()

    has_any_record = any(records for records in per_layer_records.values())
    if not has_any_record:
        raise SystemExit(
            "XAI evaluation did not produce any valid records. "
            "Check checkpoint compatibility, target layer names, and requested saliency mode."
        )

    summary = {
        "weights": str(weights_path),
        "data": str(data_yaml),
        "split": args.split,
        "target_layers": list(per_layer_records),
        "saliency_mode": args.saliency_mode,
        "num_requested_samples": len(samples),
        "num_skipped_samples": len(skipped_samples),
        "skipped_samples_preview": skipped_samples[:16],
        "per_layer": {
            layer_name: {
                "summary": aggregate_xai_metrics(records),
                "per_image_path": f"{layer_name}_per_image.json",
            }
            for layer_name, records in per_layer_records.items()
        },
    }
    for layer_name, records in per_layer_records.items():
        save_json(output_dir / f"{layer_name}_per_image.json", records)
    save_json(output_dir / "xai_metrics.json", summary)
    logger.info("XAI evaluation finished.")


if __name__ == "__main__":
    main()
