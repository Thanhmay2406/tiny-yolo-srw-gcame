from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.xai.gcame_detector import GCAMEPlaceholder
from src.xai.gradcam_detector import GradCAMLikeDetector
from src.xai.saliency_base import BaseSaliencyProvider
from src.xai.saliency_head import SaliencyHead
from src.xai.saliency_normalization import normalize_saliency_tensor


def _resize_tensor(mask: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected tensor with shape [B,1,H,W] or [B,H,W], got {tuple(mask.shape)}")
    return F.interpolate(mask.float(), size=spatial_size, mode="bilinear", align_corners=False)


class SaliencyProvider(BaseSaliencyProvider):
    def __init__(self, mode: str, channels: int, teacher_dir: str | Path | None = None) -> None:
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.teacher_dir = Path(teacher_dir).resolve() if teacher_dir is not None else None
        self.saliency_head = SaliencyHead(channels) if mode == "saliency_head" else None
        self.gradcam_like = GradCAMLikeDetector() if mode == "gradcam_like_online_debug" else None
        self.gcame = GCAMEPlaceholder() if mode == "gcame_placeholder" else None
        self.teacher_manifest = self._load_teacher_manifest() if mode == "offline_xai_teacher" else None

    def _load_teacher_manifest(self) -> dict[str, str]:
        if self.teacher_dir is None:
            raise ValueError("teacher_dir is required when mode='offline_xai_teacher'")
        manifest_path = self.teacher_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Teacher manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload.get("items", {})

    def _load_teacher_tensor(self, image_ids: list[str], spatial_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        if self.teacher_dir is None or self.teacher_manifest is None:
            raise ValueError("Teacher manifest is not initialized.")

        saliency_batch = []
        for image_id in image_ids:
            relative_path = self.teacher_manifest.get(image_id)
            if relative_path is None:
                raise KeyError(f"Teacher saliency not found for image id: {image_id}")
            saliency_path = (self.teacher_dir / relative_path).resolve()
            saliency = np.load(saliency_path).astype(np.float32)
            saliency_tensor = torch.from_numpy(saliency).to(device=device)
            if saliency_tensor.ndim == 2:
                saliency_tensor = saliency_tensor.unsqueeze(0)
            saliency_batch.append(saliency_tensor)

        stacked = torch.stack(saliency_batch, dim=0)
        resized = _resize_tensor(stacked, spatial_size=spatial_size)
        return normalize_saliency_tensor(resized).detach()

    def forward(
        self,
        feature_map: torch.Tensor,
        image_ids: list[str] | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spatial_size = tuple(feature_map.shape[-2:])
        if self.mode == "saliency_head":
            assert self.saliency_head is not None
            return self.saliency_head(feature_map)
        if self.mode == "gt_mask_debug":
            if gt_mask is None:
                raise ValueError("gt_mask is required when mode='gt_mask_debug'")
            return normalize_saliency_tensor(_resize_tensor(gt_mask, spatial_size=spatial_size))
        if self.mode == "offline_xai_teacher":
            if not image_ids:
                raise ValueError("image_ids are required when mode='offline_xai_teacher'")
            return self._load_teacher_tensor(image_ids, spatial_size=spatial_size, device=feature_map.device)
        if self.mode == "gradcam_like_online_debug":
            assert self.gradcam_like is not None
            return self.gradcam_like(feature_map)
        if self.mode == "gcame_placeholder":
            assert self.gcame is not None
            return self.gcame(feature_map, image_ids=image_ids, gt_mask=gt_mask)
        raise ValueError(f"Unsupported saliency provider mode: {self.mode}")
