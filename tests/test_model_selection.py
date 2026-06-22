from __future__ import annotations

from pathlib import Path

from src.evaluation.model_selection import assign_recommended_use, load_model_selection_row
from src.utils.io import ensure_dir, save_json


def write_detection_eval(run_dir: Path, map50_95: float, recall_tiny: float) -> None:
    save_json(
        run_dir / "detection_eval" / "metrics.json",
        {
            "metrics": {
                "metrics/mAP50(B)": map50_95 + 0.2,
                "metrics/mAP50-95(B)": map50_95,
            },
            "small_object_metrics": {
                "recall_tiny": recall_tiny,
                "recall_small": 0.9,
                "recall_medium_large": 0.95,
            },
        },
    )


def write_convergence(run_dir: Path, best_epoch: int) -> None:
    save_json(
        run_dir / "convergence_eval" / "convergence_metrics.json",
        {
            "best": {
                "best_epoch": best_epoch,
                "best_value": 0.3,
            }
        },
    )


def test_load_model_selection_row_handles_missing_xai(tmp_path: Path) -> None:
    run_dir = ensure_dir(tmp_path / "baseline")
    write_detection_eval(run_dir, map50_95=0.35, recall_tiny=0.75)
    write_convergence(run_dir, best_epoch=90)

    row = load_model_selection_row(run_dir, baseline_map50_95=0.35, baseline_tiny_recall=0.75)
    assert row["run"] == "baseline"
    assert row["pointing_game"] is None
    assert row["delta_vs_baseline"] == 0.0


def test_assign_recommended_use_marks_detection_and_xai_candidates(tmp_path: Path) -> None:
    baseline_dir = ensure_dir(tmp_path / "baseline")
    write_detection_eval(baseline_dir, map50_95=0.35, recall_tiny=0.75)
    write_convergence(baseline_dir, best_epoch=90)

    detection_dir = ensure_dir(tmp_path / "det")
    write_detection_eval(detection_dir, map50_95=0.37, recall_tiny=0.74)
    write_convergence(detection_dir, best_epoch=95)

    xai_dir = ensure_dir(tmp_path / "xai")
    write_detection_eval(xai_dir, map50_95=0.36, recall_tiny=0.76)
    write_convergence(xai_dir, best_epoch=96)
    save_json(
        xai_dir / "xai_eval" / "xai_metrics.json",
        {
            "per_layer": {
                "P3": {
                    "summary": {
                        "pointing_game": 0.1,
                        "energy_in_box": 0.2,
                        "background_energy_ratio": 0.7,
                    }
                }
            }
        },
    )

    rows = [
        load_model_selection_row(baseline_dir, baseline_map50_95=0.35, baseline_tiny_recall=0.75),
        load_model_selection_row(detection_dir, baseline_map50_95=0.35, baseline_tiny_recall=0.75),
        load_model_selection_row(xai_dir, baseline_map50_95=0.35, baseline_tiny_recall=0.75),
    ]
    rows = assign_recommended_use(rows)

    assert any(row["run"] == "det" and "best detection candidate" in row["recommended_use"] for row in rows)
    assert any(row["run"] == "xai" and "better XAI localization candidate" in row["recommended_use"] for row in rows)
