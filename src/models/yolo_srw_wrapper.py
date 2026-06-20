from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.models.layer_resolver import resolve_target_layer
from src.models.srw import SRWModule
from src.xai.hooks import ForwardCapture
from src.xai.saliency_provider import SaliencyProvider


@dataclass
class DryRunReport:
    target_name: str
    target_index: int
    feature_shape: list[int]
    saliency_shape: list[int]
    srw_output_shape: list[int]
    saliency_mode: str
    alpha: float
    trainable_params: int


class YoloSRWDebugWrapper:
    def __init__(
        self,
        yolo_model: Any,
        target_layer: str = "P3",
        saliency_mode: str = "saliency_head",
        teacher_dir: str | None = None,
        alpha_init: float = 0.1,
    ) -> None:
        self.yolo_model = yolo_model
        self.core_model = yolo_model.model
        self.target_name, self.target_index = resolve_target_layer(yolo_model, target_layer)
        self.capture = ForwardCapture(self.core_model.model[self.target_index])
        self.saliency_mode = saliency_mode
        self.teacher_dir = teacher_dir
        self.alpha_init = alpha_init

    def close(self) -> None:
        self.capture.remove()

    def dry_run(
        self,
        image_tensor: torch.Tensor,
        image_ids: list[str] | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DryRunReport]:
        self.capture.clear()
        _ = self.core_model(image_tensor)
        feature_map = self.capture.output
        if feature_map is None:
            raise RuntimeError("Forward hook did not capture the target YOLO feature map.")

        provider = SaliencyProvider(
            mode=self.saliency_mode,
            channels=int(feature_map.shape[1]),
            teacher_dir=self.teacher_dir,
        ).to(feature_map.device)
        saliency_map = provider(feature_map, image_ids=image_ids, gt_mask=gt_mask)

        srw = SRWModule(channels=int(feature_map.shape[1]), alpha_init=self.alpha_init).to(feature_map.device)
        srw_output, gate_s, gate_c, alpha = srw(feature_map, saliency_map, return_gates=True)

        report = DryRunReport(
            target_name=self.target_name,
            target_index=self.target_index,
            feature_shape=list(feature_map.shape),
            saliency_shape=list(saliency_map.shape),
            srw_output_shape=list(srw_output.shape),
            saliency_mode=self.saliency_mode,
            alpha=float(alpha.detach().cpu().item()),
            trainable_params=sum(param.numel() for param in list(provider.parameters()) + list(srw.parameters()) if param.requires_grad),
        )
        return saliency_map, gate_s, gate_c, report
