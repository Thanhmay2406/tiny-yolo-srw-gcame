#!/usr/bin/env python3

from __future__ import annotations

"""Collect V2 tiny-aware detection and XAI metrics into markdown and CSV tables.

This script never trains. It only reads existing result files when present and
marks missing artifacts as `MISSING` in the output tables.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, load_json, load_yaml


DEFAULT_RUNS = [
    "baseline_yolov8s",
    "tradaug_yolov8s_seed0",
    "srw_lsal_energy_bg_seed0",
    "baseline_yolov8s_p2_seed0",
    "srw_lsal_p2p3_mse_sizeaware_seed0",
    "srw_lsal_p2_mse_sizeaware_seed0",
    "srw_lsal_p2p3_mse_no_sizeaware_seed0",
    "srw_lsal_p2p3_energy_bg_sizeaware_seed0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect V2 tiny-aware experiment metrics into CSV and markdown tables.")
    parser.add_argument("--root", type=Path, default=Path("experiments/skyfusion"), help="Experiment root directory.")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Run names to collect.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/v2_tiny_aware_ablation_results.md"),
        help="Output markdown path. A CSV with the same stem will also be written.",
    )
    return parser.parse_args()


def optional_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = load_yaml(path)
    return payload if isinstance(payload, dict) else None


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def stringify_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def infer_group(run_name: str) -> str:
    mapping = {
        "baseline_yolov8s": "P3 baseline",
        "tradaug_yolov8s_seed0": "Traditional aug",
        "srw_lsal_energy_bg_seed0": "P3 best detection",
        "baseline_yolov8s_p2_seed0": "P2 baseline",
        "srw_lsal_p2p3_mse_sizeaware_seed0": "V2 main",
        "srw_lsal_p2_mse_sizeaware_seed0": "V2 layer ablation",
        "srw_lsal_p2p3_mse_no_sizeaware_seed0": "V2 size ablation",
        "srw_lsal_p2p3_energy_bg_sizeaware_seed0": "V2 optional",
    }
    return mapping.get(run_name, "Other")


def infer_notes(run_name: str) -> str:
    mapping = {
        "baseline_yolov8s": "Existing seed0 baseline",
        "tradaug_yolov8s_seed0": "Existing seed0 augmentation baseline",
        "srw_lsal_energy_bg_seed0": "Existing best mAP candidate",
        "baseline_yolov8s_p2_seed0": "Architecture effect",
        "srw_lsal_p2p3_mse_sizeaware_seed0": "Main tiny-aware candidate",
        "srw_lsal_p2_mse_sizeaware_seed0": "P2-only effect",
        "srw_lsal_p2p3_mse_no_sizeaware_seed0": "Size-aware effect",
        "srw_lsal_p2p3_energy_bg_sizeaware_seed0": "Optional after main V2",
    }
    return mapping.get(run_name, "")


def load_row(run_dir: Path) -> dict[str, Any]:
    run_name = run_dir.name
    config = optional_yaml(run_dir / "config.yaml") or optional_yaml(run_dir / "args.yaml") or {}
    detection = optional_json(run_dir / "detection_eval" / "metrics.json") or {}
    xai = optional_json(run_dir / "xai_eval" / "xai_metrics.json") or {}

    metrics = detection.get("metrics", {}) if isinstance(detection, dict) else {}
    small = detection.get("small_object_metrics", {}) if isinstance(detection, dict) else {}
    per_layer = xai.get("per_layer", {}) if isinstance(xai, dict) else {}
    xai_summary = None
    if "P2" in per_layer and isinstance(per_layer["P2"], dict):
        xai_summary = per_layer["P2"].get("summary")
    if xai_summary is None and "P3" in per_layer and isinstance(per_layer["P3"], dict):
        xai_summary = per_layer["P3"].get("summary")
    if xai_summary is None:
        for payload in per_layer.values():
            if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
                xai_summary = payload["summary"]
                break

    target_layers = config.get("target_layers")
    if isinstance(target_layers, list):
        target_layers_text = "+".join(str(item) for item in target_layers)
    elif target_layers is None:
        target_layers_text = "default"
    else:
        target_layers_text = str(target_layers)

    row = {
        "group": infer_group(run_name),
        "run": run_name,
        "model": config.get("model", "MISSING"),
        "target_layers": target_layers_text,
        "loss": config.get("loss_type", "none" if run_name.startswith(("baseline", "tradaug")) else "MISSING"),
        "size_aware": config.get("size_aware", "no" if run_name.startswith(("baseline", "tradaug")) else "MISSING"),
        "mAP50": metrics.get("metrics/mAP50(B)"),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)"),
        "precision": metrics.get("metrics/precision(B)"),
        "recall": metrics.get("metrics/recall(B)"),
        "recall_tiny": small.get("recall_tiny"),
        "recall_small": small.get("recall_small"),
        "xai_pointing_game": None if not isinstance(xai_summary, dict) else xai_summary.get("pointing_game"),
        "energy_in_box": None if not isinstance(xai_summary, dict) else xai_summary.get("energy_in_box"),
        "bg_energy_ratio": None if not isinstance(xai_summary, dict) else xai_summary.get("background_energy_ratio"),
        "notes": infer_notes(run_name),
    }
    return row


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "group",
        "run",
        "model",
        "target_layers",
        "loss",
        "size_aware",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "recall_tiny",
        "recall_small",
        "xai_pointing_game",
        "energy_in_box",
        "bg_energy_ratio",
        "notes",
    ]
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    lines = [
        "# V2 Tiny-Aware Ablation Results",
        "",
        "This table is generated from available local artifacts only. Missing files are marked as `MISSING`.",
        "",
        "| Group | Run | Model | Target layers | Loss | Size-aware | mAP50 | mAP50-95 | Precision | Recall | Recall tiny | Recall small | XAI pointing game | Energy in box | BG energy ratio | Notes |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    row["run"],
                    str(row["model"]),
                    str(row["target_layers"]),
                    str(row["loss"]),
                    str(row["size_aware"]),
                    stringify_metric(row["mAP50"]),
                    stringify_metric(row["mAP50-95"]),
                    stringify_metric(row["precision"]),
                    stringify_metric(row["recall"]),
                    stringify_metric(row["recall_tiny"]),
                    stringify_metric(row["recall_small"]),
                    stringify_metric(row["xai_pointing_game"]),
                    stringify_metric(row["energy_in_box"]),
                    stringify_metric(row["bg_energy_ratio"]),
                    row["notes"],
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    root = args.root.expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    output_md = args.output.expanduser()
    if not output_md.is_absolute():
        output_md = (Path.cwd() / output_md).resolve()
    output_csv = output_md.with_suffix(".csv")

    rows: list[dict[str, Any]] = []
    for run_name in args.runs:
        rows.append(load_row(root / run_name))

    write_csv(rows, output_csv)
    write_markdown(rows, output_md)
    print(f"Wrote markdown: {output_md}")
    print(f"Wrote csv: {output_csv}")


if __name__ == "__main__":
    main()
