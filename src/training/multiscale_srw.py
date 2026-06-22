from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.models.layer_resolver import resolve_target_layer


@dataclass(frozen=True)
class ScaleTarget:
    name: str
    index: int
    weight: float


def parse_target_layers(target_layers: str | Iterable[str]) -> list[str]:
    if isinstance(target_layers, str):
        raw_items = target_layers.replace(",", " ").split()
    else:
        raw_items = [str(item) for item in target_layers]

    layers = [item.strip().upper() for item in raw_items if item and item.strip()]
    if not layers:
        raise ValueError("At least one target layer is required.")

    deduped: list[str] = []
    seen: set[str] = set()
    for layer in layers:
        if layer in seen:
            raise ValueError(f"Duplicate target layer is not allowed: {layer}")
        seen.add(layer)
        deduped.append(layer)
    return deduped


def parse_scale_weights(scale_weights: Iterable[float] | None, num_layers: int) -> list[float]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if scale_weights is None:
        return [1.0] * num_layers

    weights = [float(weight) for weight in scale_weights]
    if len(weights) != num_layers:
        raise ValueError(
            f"Scale weights count ({len(weights)}) must match target layers count ({num_layers})."
        )
    return weights


def resolve_scale_targets(
    model,
    target_layers: str | Iterable[str],
    scale_weights: Iterable[float] | None = None,
) -> list[ScaleTarget]:
    layers = parse_target_layers(target_layers)
    weights = parse_scale_weights(scale_weights, num_layers=len(layers))
    resolved: list[ScaleTarget] = []
    for layer_name, weight in zip(layers, weights, strict=True):
        resolved_name, target_index = resolve_target_layer(model, layer_name)
        resolved.append(ScaleTarget(name=resolved_name, index=target_index, weight=float(weight)))
    return resolved
