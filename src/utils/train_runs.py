from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable


def csv_last_row(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    metrics: dict[str, Any] = {}
    for key, value in rows[-1].items():
        if value is None:
            continue
        text = value.strip()
        if not text:
            continue
        try:
            metrics[key] = float(text)
        except ValueError:
            metrics[key] = text
    return metrics


def normalize_metric_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): normalize_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_metric_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            return str(value)
    return str(value)


def collect_train_metrics(
    model: Any,
    run_dir: str | Path,
    train_result: Any = None,
    extra_collector: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if isinstance(train_result, dict):
        metrics.update(train_result)
    results_dict = getattr(train_result, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics.update(results_dict)

    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        for attr in ("best", "last", "save_dir", "fitness"):
            value = getattr(trainer, attr, None)
            if value is not None:
                metrics[attr] = value
        trainer_metrics = getattr(trainer, "metrics", None)
        if isinstance(trainer_metrics, dict):
            metrics.update(trainer_metrics)
        if extra_collector is not None:
            metrics.update(extra_collector(trainer))

    run_path = Path(run_dir)
    row = csv_last_row(run_path / "results.csv")
    if row:
        metrics["results_csv_last_row"] = row
        metrics.update({key: value for key, value in row.items() if key not in metrics})
    return normalize_metric_value(metrics)
