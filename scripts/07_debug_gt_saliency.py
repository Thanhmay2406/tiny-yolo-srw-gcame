#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.saliency_masks import (
    create_bbox_mask,
    create_gaussian_bbox_mask,
    read_yolo_label_file,
    resize_mask_to_feature,
    yolo_boxes_to_pixel_boxes,
)
from src.datasets.yolo_dataset import list_split_samples
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug GT saliency masks derived from YOLO labels.")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset yaml.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "valid", "val", "test"))
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to visualize.")
    parser.add_argument("--sigma-ratio", type=float, default=0.04, help="Gaussian sigma ratio.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for debug images.")
    return parser.parse_args()


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    heat = np.zeros_like(image)
    for channel_index, value in enumerate(color):
        heat[:, :, channel_index] = (mask * value).astype(np.uint8)
    return cv2.addWeighted(image, 0.7, heat, 0.3, 0.0)


def draw_boxes(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    output = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    return output


def save_preview(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    samples = list_split_samples(args.data, split=args.split, limit=args.num_samples)

    for index, sample in enumerate(samples):
        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {sample['image_path']}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        labels = read_yolo_label_file(sample["label_path"])
        pixel_boxes = yolo_boxes_to_pixel_boxes(labels, image_size=(height, width))
        hard_mask = create_bbox_mask(labels, image_size=(height, width))
        gaussian_mask = create_gaussian_bbox_mask(labels, image_size=(height, width), sigma_ratio=args.sigma_ratio)

        drawn = draw_boxes(image_rgb, pixel_boxes)
        hard_overlay = overlay_mask(drawn, hard_mask, (255, 0, 0))
        gaussian_overlay = overlay_mask(drawn, gaussian_mask, (0, 255, 0))

        base_name = f"{index:03d}_{Path(str(sample['image_id'])).stem}"
        save_preview(output_dir / f"{base_name}_image_boxes.jpg", cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        save_preview(output_dir / f"{base_name}_hard_overlay.jpg", cv2.cvtColor(hard_overlay, cv2.COLOR_RGB2BGR))
        save_preview(output_dir / f"{base_name}_gaussian_overlay.jpg", cv2.cvtColor(gaussian_overlay, cv2.COLOR_RGB2BGR))

        for level_name, stride in (("P3", 8), ("P4", 16), ("P5", 32)):
            feature_size = (max(height // stride, 1), max(width // stride, 1))
            resized = resize_mask_to_feature(gaussian_mask, feature_size=feature_size)
            preview = (resized * 255.0).astype(np.uint8)
            save_preview(output_dir / f"{base_name}_{level_name}_mask.jpg", preview)

    print(f"Saved GT saliency debug artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
