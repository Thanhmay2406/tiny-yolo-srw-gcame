from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, load_json


def _optional_json(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    payload = load_json(candidate)
    return payload if isinstance(payload, dict) else None


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _xai_summary(run_dir: str | Path, preferred_layer: str = "P3") -> dict[str, Any] | None:
    payload = _optional_json(Path(run_dir) / "xai_eval" / "xai_metrics.json")
    if payload is None:
        return None
    per_layer = payload.get("per_layer", {})
    if preferred_layer in per_layer:
        return per_layer[preferred_layer].get("summary")
    for layer_payload in per_layer.values():
        summary = layer_payload.get("summary")
        if isinstance(summary, dict):
            return summary
    return None


def _xai_score(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    pointing_game = summary.get("pointing_game")
    energy_in_box = summary.get("energy_in_box")
    background_energy_ratio = summary.get("background_energy_ratio")
    if pointing_game is None or energy_in_box is None or background_energy_ratio is None:
        return None
    return float((float(pointing_game) + float(energy_in_box) + (1.0 - float(background_energy_ratio))) / 3.0)


def load_model_selection_row(
    run_dir: str | Path,
    baseline_map50_95: float | None = None,
    baseline_tiny_recall: float | None = None,
    preferred_layer: str = "P3",
) -> dict[str, Any]:
    run_path = Path(run_dir)
    detection = _optional_json(run_path / "detection_eval" / "metrics.json")
    if detection is None:
        raise FileNotFoundError(f"Detection evaluation metrics not found: {run_path / 'detection_eval' / 'metrics.json'}")

    xai_summary = _xai_summary(run_path, preferred_layer=preferred_layer)
    convergence = _optional_json(run_path / "convergence_eval" / "convergence_metrics.json")
    small_object_metrics = detection.get("small_object_metrics", {})
    metrics = detection.get("metrics", {})

    map50_95 = metrics.get("metrics/mAP50-95(B)")
    recall_tiny = small_object_metrics.get("recall_tiny")
    note_parts: list[str] = []
    if baseline_tiny_recall is not None and recall_tiny is not None and float(recall_tiny) < float(baseline_tiny_recall):
        note_parts.append("tiny recall below baseline")
    if convergence and convergence.get("best", {}).get("best_epoch") is not None:
        note_parts.append(f"best epoch={convergence['best']['best_epoch']}")

    return {
        "run": run_path.name,
        "mAP50": metrics.get("metrics/mAP50(B)"),
        "mAP50-95": map50_95,
        "delta_vs_baseline": (float(map50_95) - float(baseline_map50_95))
        if map50_95 is not None and baseline_map50_95 is not None
        else None,
        "recall_tiny": recall_tiny,
        "recall_small": small_object_metrics.get("recall_small"),
        "pointing_game": None if xai_summary is None else xai_summary.get("pointing_game"),
        "energy_in_box": None if xai_summary is None else xai_summary.get("energy_in_box"),
        "BER": None if xai_summary is None else xai_summary.get("background_energy_ratio"),
        "xai_score_proxy": _xai_score(xai_summary),
        "recommended_use": "needs multi-seed confirmation",
        "note": "; ".join(note_parts) if note_parts else "",
    }


def assign_recommended_use(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    detection_candidates = [row for row in rows if row.get("mAP50-95") is not None]
    best_detection = max(detection_candidates, key=lambda row: float(row["mAP50-95"])) if detection_candidates else None

    xai_candidates = [row for row in rows if row.get("xai_score_proxy") is not None]
    best_xai = max(xai_candidates, key=lambda row: float(row["xai_score_proxy"])) if xai_candidates else None

    for row in rows:
        labels: list[str] = []
        if best_detection is not None and row["run"] == best_detection["run"]:
            labels.append("best detection candidate")
        if best_xai is not None and row["run"] == best_xai["run"]:
            labels.append("better XAI localization candidate")
        if row.get("pointing_game") is None:
            labels.append("not enough XAI data")
        if not labels:
            labels.append("needs multi-seed confirmation")
        row["recommended_use"] = "; ".join(labels)
    return rows


def write_model_selection_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    fieldnames = [
        "run",
        "mAP50",
        "mAP50-95",
        "delta_vs_baseline",
        "recall_tiny",
        "recall_small",
        "pointing_game",
        "energy_in_box",
        "BER",
        "recommended_use",
        "note",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return output


def write_model_selection_markdown(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    lines = [
        "# Balanced Model Selection Summary",
        "",
        "This table is a lightweight comparison aid, not a proof of absolute model superiority.",
        "",
        "| Run | mAP50 | mAP50-95 | Delta vs baseline | Recall_tiny | Recall_small | Pointing Game | Energy-in-Box | BER | Recommended use | Note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run']}`",
                    _format_value(row.get("mAP50")),
                    _format_value(row.get("mAP50-95")),
                    _format_value(row.get("delta_vs_baseline")),
                    _format_value(row.get("recall_tiny")),
                    _format_value(row.get("recall_small")),
                    _format_value(row.get("pointing_game")),
                    _format_value(row.get("energy_in_box")),
                    _format_value(row.get("BER")),
                    row.get("recommended_use") or "",
                    row.get("note") or "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `recommended_use` is intentionally conservative and does not declare a universally best model.",
            "- Detection ranking and XAI ranking may disagree.",
            "- Multi-seed confirmation is still required before making strong claims.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
