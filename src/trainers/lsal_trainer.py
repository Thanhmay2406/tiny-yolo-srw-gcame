from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from src.datasets.saliency_masks import build_batch_gaussian_masks_from_targets
from src.losses.saliency_alignment import get_saliency_loss
from src.models.layer_resolver import resolve_target_layer
from src.xai.saliency_head import SaliencyHead


def _infer_module_out_channels(module: torch.nn.Module) -> int:
    for attr_name in ("conv", "cv2", "cv1"):
        attr = getattr(module, attr_name, None)
        if attr is None:
            continue
        conv = getattr(attr, "conv", None)
        if conv is not None and hasattr(conv, "out_channels"):
            return int(conv.out_channels)
        if hasattr(attr, "out_channels"):
            return int(attr.out_channels)
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, torch.nn.Conv2d):
            return int(submodule.out_channels)
    raise ValueError(f"Could not infer output channels for module type {type(module).__name__}")


class LSalDetectionLoss(v8DetectionLoss):
    def __init__(self, model: torch.nn.Module, loss_type: str = "mse", lambda_sal: float = 0.1):
        super().__init__(model)
        self.saliency_loss_fn = get_saliency_loss(loss_type)
        self.lambda_sal = float(lambda_sal)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        det_loss, det_detach = super().loss(preds, batch)
        saliency_pred = preds["saliency_pred"]
        saliency_target = batch["gt_saliency_mask"]
        if tuple(saliency_target.shape[-2:]) != tuple(saliency_pred.shape[-2:]):
            saliency_target = torch.nn.functional.interpolate(
                saliency_target.float(),
                size=saliency_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        saliency_loss = self.saliency_loss_fn(saliency_pred, saliency_target)
        saliency_item = saliency_loss.unsqueeze(0) * saliency_pred.shape[0] * self.lambda_sal
        total_items = torch.cat([det_loss, saliency_item], dim=0)
        detached_items = torch.cat(
            [det_detach, saliency_loss.detach().unsqueeze(0) * self.lambda_sal],
            dim=0,
        )
        return total_items, detached_items


class LSalDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str = "yolov8s.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        target_layer: str = "P3",
        loss_type: str = "mse",
        lambda_sal: float = 0.1,
        sigma_ratio: float = 0.04,
    ) -> None:
        self._lsal_target_layer = target_layer
        self._lsal_loss_type = loss_type
        self._lsal_lambda_sal = lambda_sal
        self._lsal_sigma_ratio = sigma_ratio
        self._captured_saliency_feature: torch.Tensor | None = None
        self._force_saliency_output = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.saliency_target_name, self.saliency_target_index = resolve_target_layer(self, target_layer)
        feature_channels = _infer_module_out_channels(self.model[self.saliency_target_index])
        self.saliency_head = SaliencyHead(in_channels=feature_channels)

    def _inject_saliency(self, output):
        if self._captured_saliency_feature is None:
            return output
        saliency_pred = self.saliency_head(self._captured_saliency_feature)
        if isinstance(output, dict):
            output = dict(output)
            output["saliency_feature"] = self._captured_saliency_feature
            output["saliency_pred"] = saliency_pred
            return output
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            pred, payload = output
            payload = dict(payload)
            payload["saliency_feature"] = self._captured_saliency_feature
            payload["saliency_pred"] = saliency_pred
            return pred, payload
        raise TypeError("Unexpected YOLO output type while adding saliency predictions.")

    def _predict_once(self, x, profile: bool = False, visualize: bool = False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        self._captured_saliency_feature = None
        target_index = getattr(self, "saliency_target_index", None)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if target_index is not None and m.i == target_index and isinstance(x, torch.Tensor):
                self._captured_saliency_feature = x
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        if (self.training or self._force_saliency_output) and self._captured_saliency_feature is not None:
            x = self._inject_saliency(x)
        return x

    def init_criterion(self):
        return LSalDetectionLoss(self, loss_type=self._lsal_loss_type, lambda_sal=self._lsal_lambda_sal)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        parsed_preds = preds[1] if isinstance(preds, tuple) else preds
        if preds is None or not isinstance(parsed_preds, dict) or "saliency_pred" not in parsed_preds:
            self._force_saliency_output = True
            try:
                preds = self.predict(batch["img"])
            finally:
                self._force_saliency_output = False
        if "gt_saliency_mask" not in batch:
            batch["gt_saliency_mask"] = build_batch_gaussian_masks_from_targets(
                batch_idx=batch["batch_idx"],
                bboxes=batch["bboxes"],
                batch_size=batch["img"].shape[0],
                image_size=tuple(batch["img"].shape[-2:]),
                sigma_ratio=float(self._lsal_sigma_ratio),
                device=batch["img"].device,
            )
        return self.criterion(preds, batch)


class LSalDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        self.custom_overrides = {
            "target_layers": overrides.pop("target_layers", "P3"),
            "loss_type": overrides.pop("loss_type", "mse"),
            "lambda_sal": overrides.pop("lambda_sal", 0.1),
            "sigma_ratio": overrides.pop("sigma_ratio", 0.04),
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
        batch["gt_saliency_mask"] = build_batch_gaussian_masks_from_targets(
            batch_idx=batch["batch_idx"],
            bboxes=batch["bboxes"],
            batch_size=batch["img"].shape[0],
            image_size=tuple(batch["img"].shape[-2:]),
            sigma_ratio=float(getattr(self.args, "sigma_ratio", 0.04)),
            device=self.device,
        )
        return batch

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = LSalDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            target_layer=str(getattr(self.args, "target_layers", "P3")),
            loss_type=str(getattr(self.args, "loss_type", "mse")),
            lambda_sal=float(getattr(self.args, "lambda_sal", 0.1)),
            sigma_ratio=float(getattr(self.args, "sigma_ratio", 0.04)),
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "sal_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=self._validator_args(), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys
