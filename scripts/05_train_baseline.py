#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, load_yaml
from src.utils.cli_config import namespace_to_config_reference, parse_args_with_optional_config
from src.utils.logging import save_run_config, save_run_metrics, setup_logging
from src.utils.runtime import configure_runtime_environment
from src.utils.seed import seed_everything
from src.utils.train_runs import collect_train_metrics

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
    parser = argparse.ArgumentParser(description="Train a clean YOLO baseline for SkyFusion.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(default_data),
        help="Path to the YOLO dataset YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Ultralytics model checkpoint or model config.",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience in epochs, following Ultralytics native behavior.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Ultralytics device string, for example '0' or 'cpu'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional dataloader worker count.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Run directory name under experiments/skyfusion.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/skyfusion"),
        help="Root directory where run outputs will be stored.",
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=0,
        help="If > 0, save prediction visualizations for this many validation/test images after training.",
    )
    parser.add_argument(
        "--visualize-split",
        type=str,
        default="val",
        choices=("train", "valid", "val", "test"),
        help="Dataset split used for post-training prediction visualizations.",
    )
    parser.add_argument(
        "--visualize-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for post-training prediction visualizations.",
    )
    return parse_args_with_optional_config(parser)


def resolve_run_dir(output_root: Path, run_name: str) -> Path:
    run_dir = output_root / run_name
    ensure_dir(run_dir)
    return run_dir


def resolve_visualize_source(data_yaml: Path, split: str) -> list[Path]:
    data_config = load_yaml(data_yaml)
    raw_value = data_config.get(split)
    if raw_value is None and split == "val":
        raw_value = data_config.get("valid")
    if raw_value is None:
        raise ValueError(f"Split '{split}' is not defined in dataset yaml: {data_yaml}")

    candidates = raw_value if isinstance(raw_value, list) else [raw_value]
    resolved: list[Path] = []
    base_dir = data_yaml.parent
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = (base_dir / candidate_path).resolve()
        resolved.append(candidate_path)
    return resolved


def collect_image_files(sources: list[Path], limit: int) -> list[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images: list[Path] = []
    for source in sources:
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.suffix.lower() in image_exts:
                    images.append(path)
                    if len(images) >= limit:
                        return images
        elif source.is_file() and source.suffix.lower() in image_exts:
            images.append(source)
            if len(images) >= limit:
                return images
    return images


def visualize_predictions(
    model: Any,
    data_yaml: Path,
    run_dir: Path,
    split: str,
    sample_count: int,
    conf: float,
    logger: Any,
) -> list[str]:
    if sample_count <= 0:
        return []

    sources = resolve_visualize_source(data_yaml=data_yaml, split=split)
    images = collect_image_files(sources=sources, limit=sample_count)
    if not images:
        logger.warning("No images found for visualization on split '%s'.", split)
        return []

    prediction_dir = run_dir / "predictions"
    ensure_dir(prediction_dir)
    model.predict(
        source=[str(path) for path in images],
        conf=conf,
        save=True,
        project=str(run_dir),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )
    return [str(path) for path in images]


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

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    run_dir = resolve_run_dir(output_root=output_root, run_name=args.run_name)

    seed_everything(args.seed, deterministic=False)

    resolved_config = {
        "data": str(data_yaml),
        "model": args.model,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "device": args.device,
        "workers": args.workers,
        "run_name": args.run_name,
        "output_root": str(output_root),
        "config": namespace_to_config_reference(args),
        "run_dir": str(run_dir),
        "visualize_samples": args.visualize_samples,
        "visualize_split": args.visualize_split,
        "visualize_conf": args.visualize_conf,
    }
    save_run_config(run_dir, resolved_config)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "project": str(output_root),
        "name": args.run_name,
        "exist_ok": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    logger.info("Starting baseline training.")
    logger.info("Run directory: %s", run_dir)

    model = YOLO(args.model)
    train_result = model.train(**train_kwargs)

    metrics = collect_train_metrics(model=model, run_dir=run_dir, train_result=train_result)

    visualization_sources = visualize_predictions(
        model=model,
        data_yaml=data_yaml,
        run_dir=run_dir,
        split=args.visualize_split,
        sample_count=args.visualize_samples,
        conf=args.visualize_conf,
        logger=logger,
    )
    if visualization_sources:
        metrics["visualized_sources"] = visualization_sources
        metrics["prediction_dir"] = str(run_dir / "predictions")

    if metrics:
        save_run_metrics(run_dir, metrics)
        logger.info("Saved metrics to %s", run_dir / "metrics.json")
    else:
        logger.warning("Training finished but no metrics were collected.")

    logger.info("Baseline training finished.")


if __name__ == "__main__":
    main()
