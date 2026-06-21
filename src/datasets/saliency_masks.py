from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def read_yolo_label_file(label_path: str | Path) -> np.ndarray:
    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values = [float(item) for item in stripped.split()]
        if len(values) != 5:
            raise ValueError(f"Expected 5 values per YOLO row in {path}, got {len(values)}")
        rows.append(values)

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def yolo_boxes_to_pixel_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.int32)

    box_array = np.asarray(boxes, dtype=np.float32)
    coords = box_array[:, -4:]

    x_center = coords[:, 0] * width
    y_center = coords[:, 1] * height
    box_width = coords[:, 2] * width
    box_height = coords[:, 3] * height

    x1 = np.clip(np.floor(x_center - box_width / 2.0), 0, width - 1)
    y1 = np.clip(np.floor(y_center - box_height / 2.0), 0, height - 1)
    x2 = np.clip(np.ceil(x_center + box_width / 2.0), 0, width)
    y2 = np.clip(np.ceil(y_center + box_height / 2.0), 0, height)

    pixel_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
    pixel_boxes[:, 2] = np.maximum(pixel_boxes[:, 2], pixel_boxes[:, 0] + 1)
    pixel_boxes[:, 3] = np.maximum(pixel_boxes[:, 3], pixel_boxes[:, 1] + 1)
    return pixel_boxes


def create_bbox_mask(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.float32)
    for x1, y1, x2, y2 in yolo_boxes_to_pixel_boxes(boxes=boxes, image_size=image_size):
        mask[y1:y2, x1:x2] = 1.0
    return mask


def create_gaussian_bbox_mask(
    boxes: np.ndarray,
    image_size: tuple[int, int],
    sigma_ratio: float = 0.04,
    sigma_px: float | None = None,
) -> np.ndarray:
    mask = create_bbox_mask(boxes=boxes, image_size=image_size)
    if mask.max() <= 0:
        return mask

    height, width = image_size
    sigma = sigma_px if sigma_px is not None else max(float(min(height, width)) * sigma_ratio, 1.0)
    kernel = max(int(round(sigma * 6)), 3)
    if kernel % 2 == 0:
        kernel += 1

    blurred = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    max_value = float(blurred.max())
    if max_value > 0:
        blurred = blurred / max_value
    return np.clip(blurred.astype(np.float32), 0.0, 1.0)


def resize_mask_to_feature(mask: np.ndarray, feature_size: tuple[int, int]) -> np.ndarray:
    feature_height, feature_width = feature_size
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError(f"Invalid feature size: {feature_size}")
    resized = cv2.resize(mask.astype(np.float32), (feature_width, feature_height), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized.astype(np.float32), 0.0, 1.0)


def build_batch_gaussian_masks_from_targets(
    batch_idx: torch.Tensor,
    bboxes: torch.Tensor,
    batch_size: int,
    image_size: tuple[int, int],
    sigma_ratio: float = 0.04,
    device: torch.device | None = None,
) -> torch.Tensor:
    height, width = image_size
    batch_masks: list[torch.Tensor] = []
    for image_index in range(batch_size):
        image_mask = np.zeros((height, width), dtype=np.float32)
        selected = bboxes[batch_idx.view(-1) == image_index]
        if selected.numel() > 0:
            cls_column = torch.zeros((selected.shape[0], 1), dtype=selected.dtype, device=selected.device)
            box_rows = torch.cat([cls_column, selected], dim=1).detach().cpu().numpy()
            image_mask = create_gaussian_bbox_mask(box_rows, image_size=image_size, sigma_ratio=sigma_ratio)
        batch_masks.append(torch.from_numpy(image_mask))

    stacked = torch.stack(batch_masks, dim=0).unsqueeze(1)
    if device is not None:
        stacked = stacked.to(device=device)
    return stacked.float()


def build_batch_bbox_masks_from_targets(
    batch_idx: torch.Tensor,
    bboxes: torch.Tensor,
    batch_size: int,
    image_size: tuple[int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    height, width = image_size
    batch_masks: list[torch.Tensor] = []
    for image_index in range(batch_size):
        image_mask = np.zeros((height, width), dtype=np.float32)
        selected = bboxes[batch_idx.view(-1) == image_index]
        if selected.numel() > 0:
            cls_column = torch.zeros((selected.shape[0], 1), dtype=selected.dtype, device=selected.device)
            box_rows = torch.cat([cls_column, selected], dim=1).detach().cpu().numpy()
            image_mask = create_bbox_mask(box_rows, image_size=image_size)
        batch_masks.append(torch.from_numpy(image_mask))

    stacked = torch.stack(batch_masks, dim=0).unsqueeze(1)
    if device is not None:
        stacked = stacked.to(device=device)
    return stacked.float()
