from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_kaggle_runtime() -> bool:
    return Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_experiments_root() -> Path:
    return repo_root() / "experiments" / "skyfusion"


def default_paper_root() -> Path:
    return repo_root() / "paper"


def kaggle_input_root() -> Path:
    return Path("/kaggle/input")


def kaggle_working_root() -> Path:
    return Path("/kaggle/working")


def allowed_reconverted_yolo_root() -> Path:
    return kaggle_working_root() / "data" / "SkyFusion_yolo_reconverted"


def dataset_roots(base_dir: str | Path | None = None) -> Dict[str, Path]:
    root = Path(base_dir) if base_dir is not None else repo_root() / "data"
    return {
        "base": root,
        "skyfusion_coco": root / "SkyFusion",
        "skyfusion_yolo": root / "SkyFusion_yolo",
        "skyfusion_yolo_reconverted": root / "SkyFusion_yolo_reconverted",
        "legacy_yolo_format": root / "YOLO_format",
    }


def ensure_experiment_dir(run_name: str, root: str | Path | None = None) -> Path:
    experiments_root = Path(root) if root is not None else default_experiments_root()
    run_dir = experiments_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def detect_skyfusion_layout(base_dir: str | Path | None = None) -> Dict[str, object]:
    roots = dataset_roots(base_dir)
    coco_root = roots["skyfusion_coco"]
    yolo_root = roots["skyfusion_yolo"]

    coco_ok = all((coco_root / split / "_annotations.coco.json").is_file() for split in ("train", "valid", "test"))
    yolo_ok = (
        (yolo_root / "data.yaml").is_file()
        and all((yolo_root / "images" / split).is_dir() for split in ("train", "valid", "test"))
        and all((yolo_root / "labels" / split).is_dir() for split in ("train", "valid", "test"))
    )

    return {
        "coco_root": coco_root,
        "coco_present": coco_ok,
        "yolo_root": yolo_root,
        "yolo_present": yolo_ok,
        "reconvert_target": roots["skyfusion_yolo_reconverted"],
    }
