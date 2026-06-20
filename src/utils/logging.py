from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from .io import ensure_dir, save_json, save_yaml


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("skyfusion")
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger

    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_run_config(run_dir: str | Path, config: Mapping[str, Any], filename: str = "config.yaml") -> Path:
    output_dir = ensure_dir(run_dir)
    return save_yaml(output_dir / filename, dict(config))


def save_run_metrics(run_dir: str | Path, metrics: Mapping[str, Any], filename: str = "metrics.json") -> Path:
    output_dir = ensure_dir(run_dir)
    return save_json(output_dir / filename, dict(metrics))
