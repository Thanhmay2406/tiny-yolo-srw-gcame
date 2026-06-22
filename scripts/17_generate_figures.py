#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.convergence import load_results_csv, plot_metric_curves
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate common training figures from an experiment directory.")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="Experiment directory containing results.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"), help="Output directory for figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    experiment_dir = args.experiment_dir.expanduser()
    if not experiment_dir.is_absolute():
        experiment_dir = (Path.cwd() / experiment_dir).resolve()
    if not experiment_dir.is_dir():
        raise SystemExit(f"Experiment directory not found: {experiment_dir}")

    frame = load_results_csv(experiment_dir)
    output_dir = ensure_dir(args.output_dir.expanduser())
    prefix = experiment_dir.name
    figure_specs = {
        "detection_metrics": [column for column in frame.columns if column.startswith("metrics/")],
        "training_losses": [column for column in frame.columns if column.startswith("train/") and column.endswith("_loss")],
        "validation_losses": [column for column in frame.columns if column.startswith("val/") and column.endswith("_loss")],
        "optimization": [column for column in frame.columns if column.startswith("lr/") or column == "lambda_sal"],
        "gates": [column for column in frame.columns if "gate_" in column or column.startswith("train/alpha_")],
    }
    for figure_name, columns in figure_specs.items():
        if not columns:
            continue
        plot_metric_curves(
            frame,
            metrics=columns,
            output_path=output_dir / f"{prefix}_{figure_name}.png",
            title=f"{prefix} - {figure_name.replace('_', ' ').title()}",
        )
    logger.info("Figure generation finished.")


if __name__ == "__main__":
    main()
