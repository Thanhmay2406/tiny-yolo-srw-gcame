#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.srw import SRWModule
from src.models.yolo_srw_wrapper import YoloSRWDebugWrapper
from src.utils.io import ensure_dir
from src.utils.runtime import configure_runtime_environment

configure_runtime_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug SRW shape and gradient behavior.")
    parser.add_argument("--batch", type=int, default=2, help="Synthetic batch size.")
    parser.add_argument("--channels", type=int, default=256, help="Synthetic feature channels.")
    parser.add_argument("--height", type=int, default=80, help="Synthetic feature height.")
    parser.add_argument("--width", type=int, default=80, help="Synthetic feature width.")
    parser.add_argument("--alpha-init", type=float, default=0.1, help="Initial SRW alpha.")
    parser.add_argument("--from-yolo", action="store_true", help="Run the debug path on a YOLO feature hook.")
    parser.add_argument("--model", type=str, default="yolov8s.yaml", help="YOLO weights or config for --from-yolo.")
    parser.add_argument("--target-layers", type=str, default="P3", help="Target YOLO feature level.")
    parser.add_argument(
        "--saliency-provider",
        type=str,
        default="saliency_head",
        choices=("saliency_head", "gradcam_like_online_debug", "gcame_placeholder"),
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size for --from-yolo.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/skyfusion/debug_srw_shapes"), help="Output directory.")
    return parser.parse_args()


def run_synthetic_debug(args: argparse.Namespace) -> dict[str, object]:
    feature_map = torch.randn(args.batch, args.channels, args.height, args.width, requires_grad=True)
    saliency_map = torch.rand(args.batch, 1, args.height, args.width)
    srw = SRWModule(channels=args.channels, alpha_init=args.alpha_init)
    output, gate_s, gate_c, alpha = srw(feature_map, saliency_map, return_gates=True)
    loss = output.mean()
    loss.backward()
    return {
        "mode": "synthetic",
        "feature_shape": list(feature_map.shape),
        "saliency_shape": list(saliency_map.shape),
        "output_shape": list(output.shape),
        "gate_s_range": [float(gate_s.min().item()), float(gate_s.max().item())],
        "gate_c_range": [float(gate_c.min().item()), float(gate_c.max().item())],
        "alpha": float(alpha.detach().cpu().item()),
        "feature_grad_norm": float(feature_map.grad.norm().item()),
    }


def run_yolo_debug(args: argparse.Namespace) -> dict[str, object]:
    from ultralytics import YOLO

    model = YOLO(args.model)
    wrapper = YoloSRWDebugWrapper(
        yolo_model=model,
        target_layer=args.target_layers,
        saliency_mode=args.saliency_provider,
        alpha_init=args.alpha_init,
    )
    try:
        image_tensor = torch.randn(1, 3, args.imgsz, args.imgsz)
        saliency_map, gate_s, gate_c, report = wrapper.dry_run(image_tensor=image_tensor)
        return {
            "mode": "from_yolo",
            "target_name": report.target_name,
            "target_index": report.target_index,
            "feature_shape": report.feature_shape,
            "saliency_shape": report.saliency_shape,
            "srw_output_shape": report.srw_output_shape,
            "saliency_mode": report.saliency_mode,
            "alpha": report.alpha,
            "trainable_params": report.trainable_params,
            "gate_s_range": [float(gate_s.min().item()), float(gate_s.max().item())],
            "gate_c_range": [float(gate_c.min().item()), float(gate_c.max().item())],
            "saliency_range": [float(saliency_map.min().item()), float(saliency_map.max().item())],
        }
    finally:
        wrapper.close()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    report = run_yolo_debug(args) if args.from_yolo else run_synthetic_debug(args)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
