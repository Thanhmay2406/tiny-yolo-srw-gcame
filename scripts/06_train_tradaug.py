#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, load_yaml
from src.utils.logging import save_run_config, save_run_metrics, setup_logging
from src.utils.runtime import configure_runtime_environment
from src.utils.seed import seed_everything

configure_runtime_environment()

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
    parser = argparse.ArgumentParser(description="Train a traditional-augmentation YOLO baseline for SkyFusion.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(default_data),
        help="Path to the YOLO dataset YAML.",
    )
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
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=0.15, help="MixUp augmentation probability.")
    parser.add_argument("--copy-paste", dest="copy_paste", type=float, default=0.0, help="Copy-paste augmentation probability.")
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.015, help="Hue jitter.")
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.7, help="Saturation jitter.")
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.4, help="Value jitter.")
    parser.add_argument("--scale", type=float, default=0.5, help="Random scale factor.")
    parser.add_argument("--translate", type=float, default=0.1, help="Random translation factor.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Left-right flip probability.")
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


def collect_train_metrics(train_result: Any, model: Any, run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if isinstance(train_result, dict):
        metrics.update(train_result)
    results_dict = getattr(train_result, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics.update(results_dict)
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        for attr in ("best", "last", "save_dir"):
            value = getattr(trainer, attr, None)
            if value is not None:
                metrics[attr] = str(value)
    metrics["results_csv_last_row"] = csv_last_row(run_dir / "results.csv")
    metrics.update(metrics["results_csv_last_row"])
    return metrics


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    if YOLO is None:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is not installed. Install project requirements first. "
            f"Original import error: {ULTRALYTICS_IMPORT_ERROR}"
        )

    data_yaml = args.data.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")
    _ = load_yaml(data_yaml)

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
        "augmentation": {
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy_paste": args.copy_paste,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "scale": args.scale,
            "translate": args.translate,
            "fliplr": args.fliplr,
        },
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
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "scale": args.scale,
        "translate": args.translate,
        "fliplr": args.fliplr,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    logger.info("Starting traditional augmentation baseline training.")
    logger.info("Run directory: %s", run_dir)

    model = YOLO(args.model)
    train_result = model.train(**train_kwargs)
    metrics = collect_train_metrics(train_result=train_result, model=model, run_dir=run_dir)
    save_run_metrics(run_dir, metrics)
    logger.info("Traditional augmentation baseline finished.")


if __name__ == "__main__":
    main()
