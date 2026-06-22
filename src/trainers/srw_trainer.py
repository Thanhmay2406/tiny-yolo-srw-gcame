from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG

from src.models.layer_resolver import resolve_target_layer
from src.models.srw import SRWModule
from src.trainers.lsal_trainer import _infer_module_out_channels
from src.xai.saliency_provider import SaliencyProvider


class SRWOnlyDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str = "yolov8s.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        target_layer: str = "P3",
        saliency_provider: str = "saliency_head",
        teacher_dir: str | None = None,
        alpha_init: float = 0.1,
    ) -> None:
        self._srw_target_layer = target_layer
        self._srw_saliency_provider = saliency_provider
        self._srw_teacher_dir = teacher_dir
        self._srw_alpha_init = alpha_init
        self._pending_image_ids: list[str] | None = None
        self._pending_gt_mask: torch.Tensor | None = None
        self.last_srw_debug: dict[str, float] = {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.srw_target_name, self.srw_target_index = resolve_target_layer(self, target_layer)
        feature_channels = _infer_module_out_channels(self.model[self.srw_target_index])
        self.saliency_provider = SaliencyProvider(
            mode=saliency_provider,
            channels=feature_channels,
            teacher_dir=teacher_dir,
        )
        self.srw_module = SRWModule(channels=feature_channels, alpha_init=alpha_init)

    def _predict_once(self, x, profile: bool = False, visualize: bool = False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        target_index = getattr(self, "srw_target_index", None)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if target_index is not None and m.i == target_index and isinstance(x, torch.Tensor):
                saliency = self.saliency_provider(
                    x,
                    image_ids=self._pending_image_ids,
                    gt_mask=self._pending_gt_mask,
                )
                x, gate_s, gate_c, alpha = self.srw_module(x, saliency, return_gates=True)
                self.last_srw_debug = {
                    "gate_s_mean": float(gate_s.mean().detach().cpu().item()),
                    "gate_s_std": float(gate_s.std(unbiased=False).detach().cpu().item()),
                    "gate_c_mean": float(gate_c.mean().detach().cpu().item()),
                    "gate_c_std": float(gate_c.std(unbiased=False).detach().cpu().item()),
                    "saliency_mean": float(saliency.mean().detach().cpu().item()),
                    "saliency_std": float(saliency.std(unbiased=False).detach().cpu().item()),
                    "alpha": float(alpha.detach().cpu().item()),
                }
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        self._pending_image_ids = batch.get("image_ids")
        self._pending_gt_mask = batch.get("gt_saliency_mask")
        try:
            return super().loss(batch, preds)
        finally:
            self._pending_image_ids = None
            self._pending_gt_mask = None


class SRWOnlyDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        self.custom_overrides = {
            "target_layers": overrides.pop("target_layers", "P3"),
            "saliency_provider": overrides.pop("saliency_provider", "saliency_head"),
            "teacher_dir": overrides.pop("teacher_dir", None),
            "alpha_init": overrides.pop("alpha_init", 0.1),
        }
        super().__init__(cfg=cfg or DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)
        for key, value in self.custom_overrides.items():
            setattr(self.args, key, value)

    def _validator_args(self):
        args = copy(self.args)
        for key in self.custom_overrides:
            if hasattr(args, key):
                delattr(args, key)
        return args

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        if "im_file" in batch:
            batch["image_ids"] = [str(path) for path in batch["im_file"]]
        return batch

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        saliency_provider = str(getattr(self.args, "saliency_provider", "saliency_head"))
        if saliency_provider != "saliency_head":
            raise ValueError("SRW-only training currently supports only saliency_head provider.")
        model = SRWOnlyDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            target_layer=str(getattr(self.args, "target_layers", "P3")),
            saliency_provider=saliency_provider,
            teacher_dir=getattr(self.args, "teacher_dir", None),
            alpha_init=float(getattr(self.args, "alpha_init", 0.1)),
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self._validator_args(),
            _callbacks=self.callbacks,
        )
