from __future__ import annotations

import pytest

from src.training.lambda_scheduler import LambdaSchedulerConfig, compute_lambda


def test_constant_schedule_stays_fixed() -> None:
    config = LambdaSchedulerConfig(
        mode="constant",
        total_epochs=10,
        warmup_epochs=0,
        lambda_max=0.2,
        lambda_min=0.01,
        constant_lambda=0.1,
    )
    assert compute_lambda(config, 0) == 0.1
    assert compute_lambda(config, 5) == 0.1


def test_linear_warmup_reaches_lambda_max() -> None:
    config = LambdaSchedulerConfig(
        mode="linear_warmup",
        total_epochs=10,
        warmup_epochs=5,
        lambda_max=0.2,
        lambda_min=0.01,
        constant_lambda=0.1,
    )
    assert compute_lambda(config, 0) == pytest.approx(0.04)
    assert compute_lambda(config, 4) == pytest.approx(0.2)
    assert compute_lambda(config, 9) == pytest.approx(0.2)


def test_cosine_decay_moves_from_max_to_min() -> None:
    config = LambdaSchedulerConfig(
        mode="cosine_decay",
        total_epochs=5,
        warmup_epochs=0,
        lambda_max=0.2,
        lambda_min=0.01,
        constant_lambda=0.1,
    )
    assert compute_lambda(config, 0) == pytest.approx(0.2)
    assert compute_lambda(config, 4) == pytest.approx(0.01)


def test_warmup_cosine_decay_warms_up_then_decays() -> None:
    config = LambdaSchedulerConfig(
        mode="warmup_cosine_decay",
        total_epochs=10,
        warmup_epochs=2,
        lambda_max=0.2,
        lambda_min=0.01,
        constant_lambda=0.1,
    )
    assert compute_lambda(config, 0) == pytest.approx(0.1)
    assert compute_lambda(config, 1) == pytest.approx(0.2)
    assert compute_lambda(config, 2) == pytest.approx(0.2)
    assert compute_lambda(config, 9) == pytest.approx(0.01)
