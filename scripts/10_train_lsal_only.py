#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.runtime import configure_runtime_environment

configure_runtime_environment()

from src.trainers.lsal_trainer import LSalDetectionTrainer
from src.utils.cli_config import namespace_to_config_reference, parse_args_with_optional_config
from src.utils.io import ensure_dir
from src.utils.logging import save_run_config, save_run_metrics, setup_logging
from src.utils.seed import seed_everything
from src.utils.train_runs import collect_train_metrics

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:  # pragma: no cover
    ULTRALYTICS_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    default_data = os.environ.get("SKYFUSION_DATA") or (
        "/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml"
    )
    parser = argparse.ArgumentParser(description="Train YOLO with L_sal only using a differentiable saliency head.")
    parser.add_argument("--data", type=Path, default=Path(default_data), help="Path to the YOLO dataset YAML.")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Ultralytics model checkpoint or config.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default=None, help="Optional Ultralytics device string.")
    parser.add_argument("--workers", type=int, default=None, help="Optional dataloader worker count.")
    parser.add_argument("--run-name", type=str, required=True, help="Run directory name under experiments/skyfusion.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/skyfusion"), help="Output root.")
    parser.add_argument("--target-layers", type=str, default="P3", help="Target FPN level, default P3.")
    parser.add_argument("--saliency-provider", type=str, default="saliency_head", choices=("saliency_head",))
    parser.add_argument("--loss-type", type=str, default="mse", choices=("mse", "bce", "dice"))
    parser.add_argument("--lambda-sal", type=float, default=0.1, help="Weight applied to saliency alignment loss.")
    parser.add_argument("--sigma-ratio", type=float, default=0.04, help="Gaussian sigma ratio for GT saliency masks.")
    return parse_args_with_optional_config(parser)


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    if YOLO is None:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is not installed. Install project requirements first. "
            f"Original import error: {ULTRALYTICS_IMPORT_ERROR}"
        )
    if args.saliency_provider != "saliency_head":
        raise SystemExit("L_sal-only training currently supports only --saliency-provider saliency_head.")

    data_yaml = args.data.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    run_dir = ensure_dir(output_root / args.run_name)

    seed_everything(args.seed, deterministic=False)

    config = {
        "data": str(data_yaml),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "patience": args.patience,
        "device": args.device,
        "workers": args.workers,
        "run_name": args.run_name,
        "output_root": str(output_root),
        "config": namespace_to_config_reference(args),
        "target_layers": args.target_layers,
        "saliency_provider": args.saliency_provider,
        "loss_type": args.loss_type,
        "lambda_sal": args.lambda_sal,
        "sigma_ratio": args.sigma_ratio,
    }
    save_run_config(run_dir, config)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "patience": args.patience,
        "project": str(output_root),
        "name": args.run_name,
        "exist_ok": True,
        "target_layers": args.target_layers,
        "loss_type": args.loss_type,
        "lambda_sal": args.lambda_sal,
        "sigma_ratio": args.sigma_ratio,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    logger.info("Starting L_sal-only training.")
    logger.info("Run directory: %s", run_dir)

    model = YOLO(args.model)
    model.train(trainer=LSalDetectionTrainer, **train_kwargs)

    metrics = collect_train_metrics(model=model, run_dir=run_dir)
    save_run_metrics(run_dir, metrics)
    logger.info("L_sal-only training finished.")


if __name__ == "__main__":
    main()
