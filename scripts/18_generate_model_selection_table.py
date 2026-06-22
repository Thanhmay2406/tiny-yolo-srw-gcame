#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.model_selection import (
    assign_recommended_use,
    load_model_selection_row,
    write_model_selection_csv,
    write_model_selection_markdown,
)
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a balanced model-selection summary table from experiment outputs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run names or directories to include in the table.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/skyfusion"),
        help="Root used when --runs are provided as run names.",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        default="baseline_yolov8s",
        help="Baseline run name or directory used for delta columns.",
    )
    parser.add_argument("--preferred-layer", type=str, default="P3", help="Preferred XAI layer when multiple summaries exist.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper/tables"), help="Directory for markdown/csv outputs.")
    return parser.parse_args()


def resolve_run_path(run_reference: str, output_root: Path) -> Path:
    candidate = Path(run_reference).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return (output_root / run_reference).resolve()


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    output_dir_target = args.output_dir.expanduser()
    output_dir = ensure_dir(output_dir_target if output_dir_target.is_absolute() else (Path.cwd() / output_dir_target).resolve())

    baseline_run_dir = resolve_run_path(args.baseline_run, output_root=output_root)
    baseline_row = load_model_selection_row(baseline_run_dir, preferred_layer=args.preferred_layer)
    rows = [
        load_model_selection_row(
            resolve_run_path(run_reference, output_root=output_root),
            baseline_map50_95=baseline_row.get("mAP50-95"),
            baseline_tiny_recall=baseline_row.get("recall_tiny"),
            preferred_layer=args.preferred_layer,
        )
        for run_reference in args.runs
    ]
    rows = assign_recommended_use(rows)

    markdown_path = write_model_selection_markdown(rows, output_dir / "model_selection_summary.md")
    csv_path = write_model_selection_csv(rows, output_dir / "model_selection_summary.csv")
    logger.info("Model-selection tables written to %s and %s", markdown_path, csv_path)


if __name__ == "__main__":
    main()
