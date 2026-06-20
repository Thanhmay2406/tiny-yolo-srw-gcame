from __future__ import annotations

from typing import Any


def _unwrap_model(model: Any) -> Any:
    return model.model if hasattr(model, "model") else model


def resolve_detect_feature_indices(model: Any) -> dict[str, int]:
    core_model = _unwrap_model(model)
    modules = getattr(core_model, "model", None)
    if modules is None or len(modules) == 0:
        raise ValueError("Could not resolve YOLO module list.")

    detect_module = modules[-1]
    feature_indices = getattr(detect_module, "f", None)
    if not isinstance(feature_indices, (list, tuple)) or len(feature_indices) < 3:
        raise ValueError("Could not read P3/P4/P5 source indices from YOLO Detect module.")

    return {
        "P3": int(feature_indices[0]),
        "P4": int(feature_indices[1]),
        "P5": int(feature_indices[2]),
    }


def resolve_target_layer(model: Any, target_layer: str) -> tuple[str, int]:
    target = str(target_layer).upper()
    if target.startswith("P") and target[1:].isdigit():
        mapping = resolve_detect_feature_indices(model)
        if target not in mapping:
            raise ValueError(f"Unsupported target layer '{target_layer}'. Available: {sorted(mapping)}")
        return target, mapping[target]

    if str(target_layer).isdigit():
        return f"IDX_{target_layer}", int(target_layer)

    raise ValueError(f"Unsupported target layer specifier: {target_layer}")
