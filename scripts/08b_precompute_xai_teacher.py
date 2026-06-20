#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.yolo_dataset import list_split_samples
from src.models.layer_resolver import resolve_target_layer
from src.utils.io import ensure_dir
from src.utils.runtime import configure_runtime_environment
from src.xai.hooks import ForwardCapture
from src.xai.saliency_provider import SaliencyProvider

configure_runtime_environment()

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute offline XAI teacher saliency maps.")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset yaml.")
    parser.add_argument("--weights", type=str, required=True, help="YOLO weights or model config.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "valid", "val", "test"))
    parser.add_argument("--target-layers", type=str, default="P3", help="Target FPN level such as P3/P4/P5.")
    parser.add_argument("--xai-method", type=str, default="gradcam_like", choices=("gradcam_like",))
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO forward image size.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for teacher saliency.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    return parser.parse_args()


def preprocess_image(image_bgr: np.ndarray, imgsz: int) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def safe_name(image_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", image_id)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    maps_dir = ensure_dir(output_dir / "maps")

    model = YOLO(args.weights)
    _, target_index = resolve_target_layer(model, args.target_layers)
    capture = ForwardCapture(model.model.model[target_index])
    provider_mode = "gradcam_like_online_debug" if args.xai_method == "gradcam_like" else args.xai_method
    manifest: dict[str, str] = {}

    try:
        for sample in list_split_samples(args.data, split=args.split, limit=args.limit):
            image_bgr = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {sample['image_path']}")
            tensor = preprocess_image(image_bgr, imgsz=args.imgsz)
            capture.clear()
            _ = model.model(tensor)
            feature_map = capture.output
            if feature_map is None:
                raise RuntimeError("YOLO forward hook did not capture the target feature map.")

            provider = SaliencyProvider(mode=provider_mode, channels=int(feature_map.shape[1]))
            saliency = provider(feature_map).detach().cpu().squeeze().numpy().astype(np.float32)
            file_name = f"{safe_name(str(sample['image_id']))}.npy"
            relative_path = str(Path("maps") / file_name)
            np.save(maps_dir / file_name, saliency)
            manifest[str(sample["image_id"])] = relative_path
    finally:
        capture.remove()

    payload = {
        "split": args.split,
        "target_layers": args.target_layers,
        "xai_method": args.xai_method,
        "items": manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved teacher saliency maps to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
