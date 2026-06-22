from __future__ import annotations

import pytest

from src.training.multiscale_srw import parse_scale_weights, parse_target_layers


def test_parse_target_layers_accepts_strings_and_iterables() -> None:
    assert parse_target_layers("P3") == ["P3"]
    assert parse_target_layers("P3 P4,P5") == ["P3", "P4", "P5"]
    assert parse_target_layers(["p3", "P4"]) == ["P3", "P4"]


def test_parse_target_layers_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate target layer"):
        parse_target_layers(["P3", "P3"])


def test_parse_scale_weights_defaults_to_ones() -> None:
    assert parse_scale_weights(None, num_layers=3) == [1.0, 1.0, 1.0]


def test_parse_scale_weights_requires_matching_length() -> None:
    with pytest.raises(ValueError, match="must match target layers count"):
        parse_scale_weights([1.0, 0.5], num_layers=3)
