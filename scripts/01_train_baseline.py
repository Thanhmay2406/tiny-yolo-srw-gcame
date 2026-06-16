"""Train a baseline Ultralytics YOLO detector for Kaggle experiments."""

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline YOLO training."""
    parser = argparse.ArgumentParser(
        description="Train a baseline Ultralytics YOLO model without saliency loss."
    )
    parser.add_argument(
        "--data",
        default="/kaggle/input/datasets/thanhmay2406/datasettop/YOLO_format/data.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Ultralytics model name or checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--run-name",
        required=True,
        help="Experiment run name. Outputs are saved under experiments/<run_name>/.",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Optional Ultralytics device override, for example '0' or 'cpu'.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def prepare_dataset_yaml(data_path: Path, output_path: Path) -> Path:
    """Create a writable dataset YAML with a resolved dataset root."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle) or {}

    if not isinstance(data_config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")

    dataset_root = data_path.parent.resolve()
    configured_root = data_config.get("path")
    if configured_root:
        configured_path = Path(configured_root).expanduser()
        if not configured_path.is_absolute():
            configured_path = (REPO_ROOT / configured_path).resolve()
        if configured_path != dataset_root:
            data_config["path"] = str(dataset_root)
    else:
        data_config["path"] = str(dataset_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_config, handle, sort_keys=False)

    return output_path


def build_train_config(args: argparse.Namespace, experiment_dir: Path) -> dict[str, Any]:
    """Build the resolved training configuration saved for reproducibility."""
    data_path = prepare_dataset_yaml(
        resolve_repo_path(args.data),
        experiment_dir / "resolved_data.yaml",
    )
    config: dict[str, Any] = {
        "data": str(data_path),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "run_name": args.run_name,
        "device": args.device,
        "project": str(experiment_dir.parent),
        "name": experiment_dir.name,
        "exist_ok": True,
        "task": "detect",
    }
    return config


def build_train_kwargs(train_config: dict[str, Any]) -> dict[str, Any]:
    """Build the subset of configuration accepted by ``YOLO.train``."""
    allowed_keys = {
        "data",
        "epochs",
        "imgsz",
        "batch",
        "seed",
        "device",
        "project",
        "name",
        "exist_ok",
    }
    return {
        key: value
        for key, value in train_config.items()
        if key in allowed_keys and value is not None
    }


def save_config(config: dict[str, Any], output_path: Path) -> None:
    """Save a YAML copy of the resolved configuration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    """Run baseline Ultralytics YOLO training."""
    args = parse_args()
    experiment_dir = (REPO_ROOT / "experiments" / args.run_name).resolve()
    train_config = build_train_config(args, experiment_dir)
    save_config(train_config, experiment_dir / "config.yaml")

    model = YOLO(args.model)
    train_kwargs = build_train_kwargs(train_config)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
