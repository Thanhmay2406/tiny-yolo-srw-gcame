#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.saliency_masks import create_gaussian_bbox_mask, read_yolo_label_file
from src.datasets.yolo_dataset import list_split_samples
from src.models.layer_resolver import resolve_target_layer
from src.utils.io import ensure_dir
from src.utils.runtime import configure_runtime_environment
from src.xai.hooks import ForwardCapture
from src.xai.saliency_provider import SaliencyProvider

configure_runtime_environment()

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug saliency providers on YOLO feature maps.")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset yaml.")
    parser.add_argument("--split", type=str, default="valid", choices=("train", "valid", "val", "test"))
    parser.add_argument("--target-layers", type=str, default="P3", help="Target FPN level such as P3/P4/P5.")
    parser.add_argument(
        "--saliency-provider",
        type=str,
        default="saliency_head",
        choices=(
            "saliency_head",
            "gt_mask_debug",
            "offline_xai_teacher",
            "gradcam_like_online_debug",
            "gcame_placeholder",
        ),
    )
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="YOLO weights or model config.")
    parser.add_argument("--teacher-dir", type=Path, default=None, help="Offline teacher directory if used.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO forward.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def preprocess_image(image_bgr: np.ndarray, imgsz: int) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor


def overlay_mask(image_bgr: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((saliency * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.65, heat, 0.35, 0.0)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    model = YOLO(args.weights)
    target_name, target_index = resolve_target_layer(model, args.target_layers)
    capture = ForwardCapture(model.model.model[target_index])
    samples = list_split_samples(args.data, split=args.split, limit=args.num_samples)
    report_items: list[dict[str, object]] = []

    try:
        for sample in samples:
            image_bgr = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {sample['image_path']}")
            tensor = preprocess_image(image_bgr, imgsz=args.imgsz)
            capture.clear()
            _ = model.model(tensor)
            feature_map = capture.output
            if feature_map is None:
                raise RuntimeError("YOLO forward hook did not capture the target feature map.")

            labels = read_yolo_label_file(sample["label_path"])
            gt_mask_np = create_gaussian_bbox_mask(labels, image_size=(image_bgr.shape[0], image_bgr.shape[1]))
            gt_mask_tensor = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0)

            provider = SaliencyProvider(
                mode=args.saliency_provider,
                channels=int(feature_map.shape[1]),
                teacher_dir=args.teacher_dir,
            )
            saliency = provider(
                feature_map,
                image_ids=[str(sample["image_id"])],
                gt_mask=gt_mask_tensor,
            )
            saliency_np = saliency.detach().cpu().squeeze().numpy().astype(np.float32)
            saliency_np = cv2.resize(saliency_np, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            saliency_np = np.clip(saliency_np, 0.0, 1.0)
            overlay = overlay_mask(image_bgr, saliency_np)

            base_name = Path(str(sample["image_id"])).stem
            cv2.imwrite(str(output_dir / f"{base_name}_overlay.jpg"), overlay)
            cv2.imwrite(str(output_dir / f"{base_name}_gt_mask.jpg"), (gt_mask_np * 255.0).astype(np.uint8))

            report_items.append(
                {
                    "image_id": sample["image_id"],
                    "target_layer": target_name,
                    "target_index": target_index,
                    "feature_shape": list(feature_map.shape),
                    "saliency_shape": list(saliency.shape),
                    "saliency_provider": args.saliency_provider,
                    "saliency_min": float(saliency.min().detach().cpu().item()),
                    "saliency_max": float(saliency.max().detach().cpu().item()),
                    "saliency_mean": float(saliency.mean().detach().cpu().item()),
                }
            )
    finally:
        capture.remove()

    (output_dir / "report.json").write_text(json.dumps(report_items, indent=2) + "\n", encoding="utf-8")
    print(f"Saved saliency debug artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
