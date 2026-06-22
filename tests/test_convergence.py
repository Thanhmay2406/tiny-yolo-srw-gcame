from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.convergence import best_epoch, epoch_to_threshold, load_results_csv, summarize_convergence


def write_results_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {"epoch": 1, "metrics/mAP50-95(B)": 0.10, "train/box_loss": 1.0, "lambda_sal": 0.2},
            {"epoch": 2, "metrics/mAP50-95(B)": 0.25, "train/box_loss": 0.8, "lambda_sal": 0.1},
            {"epoch": 3, "metrics/mAP50-95(B)": 0.20, "train/box_loss": 0.7, "lambda_sal": 0.05},
        ]
    ).to_csv(path, index=False)


def test_best_epoch_and_threshold(tmp_path: Path) -> None:
    write_results_csv(tmp_path / "results.csv")
    frame = load_results_csv(tmp_path)
    best = best_epoch(frame, metric="metrics/mAP50-95(B)")
    threshold = epoch_to_threshold(frame, metric="metrics/mAP50-95(B)", threshold=0.2)
    assert best["best_epoch"] == 2
    assert best["best_value"] == 0.25
    assert threshold == 2


def test_summarize_convergence_reports_primary_metric(tmp_path: Path) -> None:
    write_results_csv(tmp_path / "results.csv")
    summary = summarize_convergence(tmp_path, primary_metric="metrics/mAP50-95(B)", thresholds=[0.2])
    assert summary["best"]["best_epoch"] == 2
    assert summary["epoch_to_threshold"]["0.2"] == 2
