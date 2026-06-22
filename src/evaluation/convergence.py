from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir


def load_results_csv(experiment_dir: str | Path) -> pd.DataFrame:
    path = Path(experiment_dir) / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"results.csv not found in experiment directory: {path.parent}")
    return pd.read_csv(path)


def numeric_metric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"epoch", "time"}
    return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]


def best_epoch(frame: pd.DataFrame, metric: str, maximize: bool = True) -> dict[str, Any]:
    if metric not in frame.columns:
        raise KeyError(f"Metric column not found: {metric}")
    series = frame[metric].dropna()
    if series.empty:
        return {"metric": metric, "best_epoch": None, "best_value": None, "maximize": maximize}
    index = int(series.idxmax() if maximize else series.idxmin())
    return {
        "metric": metric,
        "best_epoch": int(frame.loc[index, "epoch"]) if "epoch" in frame.columns else int(index + 1),
        "best_value": float(frame.loc[index, metric]),
        "maximize": bool(maximize),
    }


def epoch_to_threshold(frame: pd.DataFrame, metric: str, threshold: float, mode: str = "gte") -> int | None:
    if metric not in frame.columns:
        raise KeyError(f"Metric column not found: {metric}")
    if mode not in {"gte", "lte"}:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    for _, row in frame.iterrows():
        value = row[metric]
        if pd.isna(value):
            continue
        if (mode == "gte" and value >= threshold) or (mode == "lte" and value <= threshold):
            return int(row["epoch"]) if "epoch" in frame.columns else None
    return None


def plot_metric_curves(
    frame: pd.DataFrame,
    metrics: list[str],
    output_path: str | Path,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(10, 6))
    x = frame["epoch"] if "epoch" in frame.columns else range(1, len(frame) + 1)
    for metric in metrics:
        if metric not in frame.columns:
            continue
        plt.plot(x, frame[metric], label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output


def summarize_convergence(
    experiment_dir: str | Path,
    primary_metric: str,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    frame = load_results_csv(experiment_dir)
    summary = {
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "epochs_recorded": int(len(frame)),
        "available_metrics": numeric_metric_columns(frame),
        "primary_metric": primary_metric,
        "best": best_epoch(frame, metric=primary_metric, maximize=True),
    }
    thresholds = thresholds or []
    summary["epoch_to_threshold"] = {
        str(threshold): epoch_to_threshold(frame, metric=primary_metric, threshold=float(threshold), mode="gte")
        for threshold in thresholds
    }
    return summary
