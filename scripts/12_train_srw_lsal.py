#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.runtime import configure_runtime_environment

configure_runtime_environment()

from src.trainers.srw_lsal_trainer import SRWLSalDetectionTrainer
from src.utils.io import ensure_dir
from src.utils.logging import save_run_config, save_run_metrics, setup_logging
from src.utils.seed import seed_everything

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
    parser = argparse.ArgumentParser(description="Train YOLO with joint SRW + L_sal.")
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
    parser.add_argument("--teacher-dir", type=Path, default=None, help="Optional offline teacher manifest directory.")
    parser.add_argument("--beta-teacher", type=float, default=0.0, help="Teacher loss weight.")
    parser.add_argument("--loss-type", type=str, default="mse", choices=("mse", "bce", "dice"))
    parser.add_argument("--lambda-sal", type=float, default=0.1, help="Weight applied to GT saliency alignment loss.")
    parser.add_argument("--sigma-ratio", type=float, default=0.04, help="Gaussian sigma ratio for GT saliency masks.")
    parser.add_argument("--alpha-init", type=float, default=0.1, help="Initial SRW alpha.")
    return parser.parse_args()


def csv_last_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    metrics: dict[str, Any] = {}
    for key, value in rows[-1].items():
        if value is None:
            continue
        text = value.strip()
        if not text:
            continue
        try:
            metrics[key] = float(text)
        except ValueError:
            metrics[key] = text
    return metrics


def collect_train_metrics(model: Any, run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        for attr in ("best", "last", "save_dir", "fitness"):
            value = getattr(trainer, attr, None)
            if value is not None:
                metrics[attr] = str(value) if isinstance(value, Path) else value
        inner_model = getattr(trainer, "model", None)
        debug_stats = getattr(inner_model, "last_srw_debug", None)
        if isinstance(debug_stats, dict):
            metrics["srw_debug"] = debug_stats
    csv_metrics = csv_last_row(run_dir / "results.csv")
    metrics["results_csv_last_row"] = csv_metrics
    metrics.update(csv_metrics)
    return metrics


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    if YOLO is None:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is not installed. Install project requirements first. "
            f"Original import error: {ULTRALYTICS_IMPORT_ERROR}"
        )
    if args.saliency_provider != "saliency_head":
        raise SystemExit("SRW + L_sal training currently supports only --saliency-provider saliency_head.")
    if args.beta_teacher > 0.0 and args.teacher_dir is None:
        raise SystemExit("--teacher-dir is required when --beta-teacher > 0.")

    data_yaml = args.data.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    run_dir = ensure_dir(output_root / args.run_name)

    teacher_dir = args.teacher_dir.expanduser().resolve() if args.teacher_dir is not None else None
    if teacher_dir is not None and not teacher_dir.is_dir():
        raise SystemExit(f"Teacher directory not found: {teacher_dir}")

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
        "target_layers": args.target_layers,
        "saliency_provider": args.saliency_provider,
        "teacher_dir": str(teacher_dir) if teacher_dir else None,
        "beta_teacher": args.beta_teacher,
        "loss_type": args.loss_type,
        "lambda_sal": args.lambda_sal,
        "sigma_ratio": args.sigma_ratio,
        "alpha_init": args.alpha_init,
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
        "saliency_provider": args.saliency_provider,
        "teacher_dir": str(teacher_dir) if teacher_dir else None,
        "beta_teacher": args.beta_teacher,
        "loss_type": args.loss_type,
        "lambda_sal": args.lambda_sal,
        "sigma_ratio": args.sigma_ratio,
        "alpha_init": args.alpha_init,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    logger.info("Starting SRW + L_sal training.")
    logger.info("Run directory: %s", run_dir)

    model = YOLO(args.model)
    model.train(trainer=SRWLSalDetectionTrainer, **train_kwargs)

    metrics = collect_train_metrics(model=model, run_dir=run_dir)
    save_run_metrics(run_dir, metrics)
    logger.info("SRW + L_sal training finished.")


if __name__ == "__main__":
    main()
