"""Evaluate a trained Ultralytics YOLO detector and save experiment metrics."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for detection evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Ultralytics YOLO detector on val or test data."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained YOLO weights, for example experiments/<run_name>/weights/best.pt.",
    )
    parser.add_argument(
        "--data",
        default="/kaggle/input/datasets/thanhmay2406/datasettop/YOLO_format/data.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--split",
        choices=("auto", "val", "test"),
        default="test",
        help="Dataset split to evaluate. 'auto' prefers test when available, otherwise val.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument(
        "--device",
        default=0,
        help="Optional Ultralytics device override, for example '0' or 'cpu'.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment run name. Defaults to the parent experiment folder inferred from --weights.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save prediction visualizations for a few sample images.",
    )
    parser.add_argument(
        "--visualize-split",
        choices=("same", "val", "test"),
        default="test",
        help="Split to use for prediction visualizations when --visualize is enabled.",
    )
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=8,
        help="Maximum number of images to visualize.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold used for optional prediction visualizations.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_dataset_config(data_path: Path) -> dict[str, Any]:
    """Load the dataset YAML and normalize its root path."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle) or {}

    if not isinstance(data_config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")

    if "path" not in data_config or not data_config["path"]:
        data_config["path"] = str(data_path.parent.resolve())
    else:
        dataset_root = Path(str(data_config["path"])).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (REPO_ROOT / dataset_root).resolve()
        data_config["path"] = str(dataset_root)

    return data_config


def save_dataset_config(data_config: dict[str, Any], target_path: Path) -> Path:
    """Write a resolved dataset YAML copy into the experiment folder."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_config, handle, sort_keys=False)
    return target_path


def infer_experiment_dir(weights_path: Path, run_name: str | None) -> Path:
    """Infer the experiment directory from weights or an explicit run name."""
    if run_name:
        return (REPO_ROOT / "experiments" / run_name).resolve()

    resolved_weights = weights_path.resolve()
    if resolved_weights.parent.name == "weights":
        return resolved_weights.parent.parent
    return resolved_weights.parent


def split_is_available(data_config: dict[str, Any], split: str) -> bool:
    """Check whether the requested dataset split exists on disk."""
    split_value = data_config.get(split)
    if not split_value:
        return False

    dataset_root = Path(str(data_config["path"]))
    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    return split_path.exists()


def resolve_eval_split(data_config: dict[str, Any], requested_split: str) -> str:
    """Resolve which split to evaluate, falling back from test to val when needed."""
    if requested_split == "auto":
        if split_is_available(data_config, "test"):
            return "test"
        return "val"

    if requested_split == "test" and not split_is_available(data_config, "test"):
        raise ValueError("Requested --split test, but the dataset YAML does not expose a usable test split.")

    if requested_split == "val" and not split_is_available(data_config, "val"):
        raise ValueError("Requested --split val, but the dataset YAML does not expose a usable val split.")

    return requested_split


def collect_metrics(results: Any, split: str, weights_path: Path, data_path: Path) -> dict[str, Any]:
    """Convert Ultralytics metrics objects into JSON-serializable output."""
    metrics: dict[str, Any] = {
        "split": split,
        "weights": str(weights_path.resolve()),
        "data": str(data_path.resolve()),
    }

    for key in ("box", "speed", "results_dict"):
        value = getattr(results, key, None)
        if value is None:
            continue
        if hasattr(value, "dict"):
            metrics[key] = value.dict()
        elif isinstance(value, dict):
            metrics[key] = value
        else:
            metrics[key] = str(value)

    for key in ("fitness", "maps", "names"):
        value = getattr(results, key, None)
        if value is None:
            continue
        if hasattr(value, "tolist"):
            metrics[key] = value.tolist()
        else:
            metrics[key] = value

    return metrics


def gather_visualization_sources(data_config: dict[str, Any], split: str, count: int) -> list[str]:
    """Collect a small set of image paths for optional prediction visualizations."""
    split_value = data_config.get(split)
    if not split_value:
        return []

    dataset_root = Path(str(data_config["path"]))
    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = dataset_root / split_path

    if not split_path.exists():
        return []

    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        str(path)
        for path in sorted(split_path.iterdir())
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]
    return image_paths[:count]


def main() -> None:
    """Run evaluation and optionally export prediction visualizations."""
    args = parse_args()
    weights_path = resolve_repo_path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    experiment_dir = infer_experiment_dir(weights_path, args.run_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = load_dataset_config(resolve_repo_path(args.data))
    resolved_split = resolve_eval_split(dataset_config, args.split)
    resolved_data_path = save_dataset_config(
        dataset_config, experiment_dir / f"resolved_data_{resolved_split}.yaml"
    )

    model = YOLO(str(weights_path))
    val_kwargs = {
        "data": str(resolved_data_path),
        "split": resolved_split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(experiment_dir.parent),
        "name": experiment_dir.name,
        "exist_ok": True,
        "plots": True,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device

    results = model.val(**val_kwargs)
    metrics = collect_metrics(results, resolved_split, weights_path, resolved_data_path)
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if not args.visualize:
        return

    prediction_split = resolved_split if args.visualize_split == "same" else args.visualize_split
    if prediction_split == "test" and not split_is_available(dataset_config, "test"):
        prediction_split = "val"

    sources = gather_visualization_sources(dataset_config, prediction_split, args.visualize_count)
    if not sources:
        return

    predict_kwargs = {
        "source": sources,
        "conf": args.conf,
        "project": str(experiment_dir),
        "name": f"predictions_{prediction_split}",
        "exist_ok": True,
        "save": True,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    model.predict(**predict_kwargs)


if __name__ == "__main__":
    main()
