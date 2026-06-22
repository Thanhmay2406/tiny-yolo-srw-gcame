from __future__ import annotations

from typing import Any

import numpy as np


def build_bbox_mask(boxes_xyxy: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.float32)
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=np.float32):
        left = int(np.clip(np.floor(x1), 0, width - 1))
        top = int(np.clip(np.floor(y1), 0, height - 1))
        right = int(np.clip(np.ceil(x2), 0, width))
        bottom = int(np.clip(np.ceil(y2), 0, height))
        if right <= left:
            right = min(left + 1, width)
        if bottom <= top:
            bottom = min(top + 1, height)
        mask[top:bottom, left:right] = 1.0
    return mask


def _safe_normalized_energy(saliency: np.ndarray) -> np.ndarray:
    array = np.asarray(saliency, dtype=np.float32)
    array = np.clip(array, a_min=0.0, a_max=None)
    total = float(array.sum())
    if total <= 0.0:
        return np.zeros_like(array)
    return array / total


def pointing_game(saliency: np.ndarray, boxes_xyxy: np.ndarray) -> float | None:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return None
    energy = _safe_normalized_energy(saliency)
    if float(energy.sum()) <= 0.0:
        return 0.0
    peak_index = int(np.argmax(energy))
    height, width = energy.shape
    peak_y, peak_x = divmod(peak_index, width)
    for x1, y1, x2, y2 in boxes:
        if x1 <= peak_x < x2 and y1 <= peak_y < y2:
            return 1.0
    return 0.0


def energy_in_box_score(saliency: np.ndarray, boxes_xyxy: np.ndarray) -> float | None:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return None
    energy = _safe_normalized_energy(saliency)
    mask = build_bbox_mask(boxes, image_size=energy.shape)
    return float((energy * mask).sum())


def background_energy_ratio(saliency: np.ndarray, boxes_xyxy: np.ndarray) -> float | None:
    score = energy_in_box_score(saliency, boxes_xyxy)
    if score is None:
        return None
    return float(max(0.0, 1.0 - score))


def evaluate_xai_image(saliency: np.ndarray, boxes_xyxy: np.ndarray) -> dict[str, float | None]:
    return {
        "pointing_game": pointing_game(saliency, boxes_xyxy),
        "energy_in_box": energy_in_box_score(saliency, boxes_xyxy),
        "background_energy_ratio": background_energy_ratio(saliency, boxes_xyxy),
        "saliency_mass_inside_bbox": energy_in_box_score(saliency, boxes_xyxy),
    }


def aggregate_xai_metrics(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_images": len(per_image)}
    for key in ("pointing_game", "energy_in_box", "background_energy_ratio", "saliency_mass_inside_bbox"):
        values = [float(item[key]) for item in per_image if item.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
        summary[f"{key}_num_valid"] = len(values)
    return summary
