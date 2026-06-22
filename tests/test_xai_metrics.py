from __future__ import annotations

import numpy as np

from src.evaluation.xai_metrics import (
    aggregate_xai_metrics,
    background_energy_ratio,
    energy_in_box_score,
    pointing_game,
)


def test_pointing_game_hits_when_peak_inside_bbox() -> None:
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[3, 4] = 1.0
    boxes = np.asarray([[4, 3, 6, 5]], dtype=np.float32)
    assert pointing_game(saliency, boxes) == 1.0


def test_energy_metrics_split_inside_and_background_mass() -> None:
    saliency = np.zeros((8, 8), dtype=np.float32)
    saliency[0:2, 0:2] = 1.0
    saliency[6:8, 6:8] = 1.0
    boxes = np.asarray([[0, 0, 2, 2]], dtype=np.float32)
    inside = energy_in_box_score(saliency, boxes)
    background = background_energy_ratio(saliency, boxes)
    assert inside == 0.5
    assert background == 0.5


def test_aggregate_xai_metrics_averages_valid_values() -> None:
    summary = aggregate_xai_metrics(
        [
            {"pointing_game": 1.0, "energy_in_box": 0.8, "background_energy_ratio": 0.2, "saliency_mass_inside_bbox": 0.8},
            {"pointing_game": 0.0, "energy_in_box": 0.6, "background_energy_ratio": 0.4, "saliency_mass_inside_bbox": 0.6},
        ]
    )
    assert summary["pointing_game"] == 0.5
    assert summary["energy_in_box"] == 0.7
