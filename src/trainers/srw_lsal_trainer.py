from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from src.datasets.saliency_masks import build_batch_bbox_masks_from_targets, build_batch_gaussian_masks_from_targets
from src.datasets.yolo_dataset import image_id_from_path
from src.losses.background_suppression import combined_energy_bg_loss
from src.losses.energy_in_box import energy_in_box_loss
from src.losses.saliency_alignment import get_saliency_loss
from src.losses.size_aware import image_level_size_weight
from src.models.layer_resolver import resolve_target_layer
from src.models.srw import SRWModule
from src.training.lambda_scheduler import LambdaScheduler, LambdaSchedulerConfig, save_lambda_curve
from src.trainers.lsal_trainer import _infer_module_out_channels
from src.xai.saliency_provider import SaliencyProvider
from src.losses.saliency_alignment import mse_saliency_loss


class SRWLSalDetectionLoss(v8DetectionLoss):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_type: str = "mse",
        lambda_sal: float = 0.1,
        beta_teacher: float = 0.0,
        lambda_schedule: str = "constant",
        lambda_max: float | None = None,
        lambda_min: float = 0.0,
        warmup_epochs: int = 0,
        total_epochs: int = 100,
        beta_bg: float = 0.5,
        dilation_radius: int = 3,
        size_aware: bool = False,
        size_weight_mode: str = "log_inverse",
        size_weight_max: float | None = None,
    ) -> None:
        super().__init__(model)
        self.loss_type = loss_type
        self.saliency_loss_fn = get_saliency_loss(loss_type) if loss_type in {"mse", "bce", "dice"} else None
        self.teacher_loss_fn = mse_saliency_loss
        self.lambda_schedule = str(lambda_schedule)
        self.lambda_scheduler = LambdaScheduler(
            LambdaSchedulerConfig(
                mode=self.lambda_schedule,
                total_epochs=int(total_epochs),
                warmup_epochs=int(warmup_epochs),
                lambda_max=float(lambda_max if lambda_max is not None else lambda_sal),
                lambda_min=float(lambda_min),
                constant_lambda=float(lambda_sal),
            )
        )
        self.lambda_sal = float(self.lambda_scheduler.current_value)
        self.beta_teacher = float(beta_teacher)
        self.beta_bg = float(beta_bg)
        self.dilation_radius = int(dilation_radius)
        self.size_aware = bool(size_aware)
        self.size_weight_mode = str(size_weight_mode)
        self.size_weight_max = float(size_weight_max) if size_weight_max is not None else None

    def _image_weights(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if not getattr(self, "size_aware", False):
            return None
        return image_level_size_weight(
            batch_idx=batch["batch_idx"],
            bboxes=batch["bboxes"],
            batch_size=int(batch["img"].shape[0]),
            mode=getattr(self, "size_weight_mode", "log_inverse"),
            max_weight=getattr(self, "size_weight_max", None),
        )

    def _gt_saliency_loss(self, saliency_pred: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image_weights = self._image_weights(batch)
        if self.loss_type in {"mse", "bce", "dice"}:
            saliency_target = batch["gt_saliency_mask"]
            if tuple(saliency_target.shape[-2:]) != tuple(saliency_pred.shape[-2:]):
                saliency_target = torch.nn.functional.interpolate(
                    saliency_target.float(),
                    size=saliency_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if image_weights is None:
                assert self.saliency_loss_fn is not None
                return self.saliency_loss_fn(saliency_pred, saliency_target)
            if self.loss_type == "mse":
                per_image = (saliency_pred - saliency_target).pow(2).mean(dim=(1, 2, 3))
            elif self.loss_type == "bce":
                pred = saliency_pred.clamp(min=1e-6, max=1.0 - 1e-6)
                per_image = torch.nn.functional.binary_cross_entropy(
                    pred,
                    saliency_target,
                    reduction="none",
                ).mean(dim=(1, 2, 3))
            else:
                pred_flat = saliency_pred.flatten(1)
                target_flat = saliency_target.flatten(1)
                intersection = (pred_flat * target_flat).sum(dim=1)
                denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
                per_image = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
            return (per_image * image_weights.to(device=per_image.device, dtype=per_image.dtype)).mean()

        bbox_mask = batch["gt_bbox_mask"]
        if tuple(bbox_mask.shape[-2:]) != tuple(saliency_pred.shape[-2:]):
            bbox_mask = torch.nn.functional.interpolate(
                bbox_mask.float(),
                size=saliency_pred.shape[-2:],
                mode="nearest",
            )
        if self.loss_type == "energy":
            return energy_in_box_loss(saliency_pred, bbox_mask, image_weights=image_weights)
        if self.loss_type == "energy_bg":
            return combined_energy_bg_loss(
                saliency_pred,
                bbox_mask,
                beta_bg=self.beta_bg,
                dilation_radius=self.dilation_radius,
                image_weights=image_weights,
            )
        raise ValueError(f"Unsupported saliency loss type: {self.loss_type}")

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        det_loss, det_detach = super().loss(preds, batch)
        saliency_pred = preds["saliency_pred"]
        saliency_loss = self._gt_saliency_loss(saliency_pred, batch)
        saliency_item = saliency_loss.unsqueeze(0) * saliency_pred.shape[0] * self.lambda_sal

        teacher_target = batch.get("teacher_saliency_mask")
        if teacher_target is not None and tuple(teacher_target.shape[-2:]) != tuple(saliency_pred.shape[-2:]):
            teacher_target = torch.nn.functional.interpolate(
                teacher_target.float(),
                size=saliency_pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if teacher_target is not None and self.beta_teacher > 0.0:
            teacher_loss = self.teacher_loss_fn(saliency_pred, teacher_target)
        else:
            teacher_loss = saliency_pred.new_zeros(())
        teacher_item = teacher_loss.unsqueeze(0) * saliency_pred.shape[0] * self.beta_teacher

        total_items = torch.cat([det_loss, saliency_item, teacher_item], dim=0)
        detached_items = torch.cat(
            [
                det_detach,
                saliency_loss.detach().unsqueeze(0) * self.lambda_sal,
                teacher_loss.detach().unsqueeze(0) * self.beta_teacher,
            ],
            dim=0,
        )
        return total_items, detached_items

    def update(self) -> None:
        self.lambda_sal = float(self.lambda_scheduler.step())


class SRWLSalDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str = "yolov8s.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        target_layer: str = "P3",
        saliency_provider: str = "saliency_head",
        alpha_init: float = 0.1,
        loss_type: str = "mse",
        lambda_sal: float = 0.1,
        beta_teacher: float = 0.0,
        lambda_schedule: str = "constant",
        lambda_max: float | None = None,
        lambda_min: float = 0.0,
        warmup_epochs: int = 0,
        beta_bg: float = 0.5,
        dilation_radius: int = 3,
        size_aware: bool = False,
        size_weight_mode: str = "log_inverse",
        size_weight_max: float | None = None,
        total_epochs: int = 100,
        sigma_ratio: float = 0.04,
    ) -> None:
        self._srw_target_layer = target_layer
        self._srw_saliency_provider = saliency_provider
        self._srw_alpha_init = alpha_init
        self._srw_loss_type = loss_type
        self._srw_lambda_sal = lambda_sal
        self._srw_beta_teacher = beta_teacher
        self._srw_lambda_schedule = lambda_schedule
        self._srw_lambda_max = lambda_max
        self._srw_lambda_min = lambda_min
        self._srw_warmup_epochs = warmup_epochs
        self._srw_beta_bg = beta_bg
        self._srw_dilation_radius = dilation_radius
        self._srw_size_aware = size_aware
        self._srw_size_weight_mode = size_weight_mode
        self._srw_size_weight_max = size_weight_max
        self._srw_total_epochs = total_epochs
        self._srw_sigma_ratio = sigma_ratio
        self._captured_saliency_pred: torch.Tensor | None = None
        self._captured_gate_s: torch.Tensor | None = None
        self._captured_gate_c: torch.Tensor | None = None
        self._force_saliency_output = False
        self.last_srw_debug: dict[str, float] = {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.srw_target_name, self.srw_target_index = resolve_target_layer(self, target_layer)
        feature_channels = _infer_module_out_channels(self.model[self.srw_target_index])
        self.saliency_provider = SaliencyProvider(mode=saliency_provider, channels=feature_channels)
        self.srw_module = SRWModule(channels=feature_channels, alpha_init=alpha_init)

    def _inject_saliency(self, output):
        if self._captured_saliency_pred is None:
            return output
        if isinstance(output, dict):
            output = dict(output)
            output["saliency_pred"] = self._captured_saliency_pred
            if self._captured_gate_s is not None:
                output["srw_gate_s"] = self._captured_gate_s
            if self._captured_gate_c is not None:
                output["srw_gate_c"] = self._captured_gate_c
            return output
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            pred, payload = output
            payload = dict(payload)
            payload["saliency_pred"] = self._captured_saliency_pred
            if self._captured_gate_s is not None:
                payload["srw_gate_s"] = self._captured_gate_s
            if self._captured_gate_c is not None:
                payload["srw_gate_c"] = self._captured_gate_c
            return pred, payload
        raise TypeError("Unexpected YOLO output type while adding SRW + L_sal predictions.")

    def _predict_once(self, x, profile: bool = False, visualize: bool = False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        target_index = getattr(self, "srw_target_index", None)
        self._captured_saliency_pred = None
        self._captured_gate_s = None
        self._captured_gate_c = None
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if target_index is not None and m.i == target_index and isinstance(x, torch.Tensor):
                saliency_pred = self.saliency_provider(x)
                x, gate_s, gate_c, alpha = self.srw_module(x, saliency_pred, return_gates=True)
                self._captured_saliency_pred = saliency_pred
                self._captured_gate_s = gate_s
                self._captured_gate_c = gate_c
                self.last_srw_debug = {
                    "gate_s_mean": float(gate_s.mean().detach().cpu().item()),
                    "gate_s_std": float(gate_s.std().detach().cpu().item()),
                    "gate_c_mean": float(gate_c.mean().detach().cpu().item()),
                    "gate_c_std": float(gate_c.std().detach().cpu().item()),
                    "saliency_min": float(saliency_pred.min().detach().cpu().item()),
                    "saliency_max": float(saliency_pred.max().detach().cpu().item()),
                    "saliency_mean": float(saliency_pred.mean().detach().cpu().item()),
                    "saliency_std": float(saliency_pred.std().detach().cpu().item()),
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

        if (self.training or self._force_saliency_output) and self._captured_saliency_pred is not None:
            x = self._inject_saliency(x)
        return x

    def init_criterion(self):
        return SRWLSalDetectionLoss(
            self,
            loss_type=self._srw_loss_type,
            lambda_sal=self._srw_lambda_sal,
            beta_teacher=self._srw_beta_teacher,
            lambda_schedule=self._srw_lambda_schedule,
            lambda_max=self._srw_lambda_max,
            lambda_min=self._srw_lambda_min,
            warmup_epochs=self._srw_warmup_epochs,
            beta_bg=self._srw_beta_bg,
            dilation_radius=self._srw_dilation_radius,
            size_aware=self._srw_size_aware,
            size_weight_mode=self._srw_size_weight_mode,
            size_weight_max=self._srw_size_weight_max,
            total_epochs=self._srw_total_epochs,
        )

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
                sigma_ratio=float(self._srw_sigma_ratio),
                device=batch["img"].device,
            )
        if "gt_bbox_mask" not in batch:
            batch["gt_bbox_mask"] = build_batch_bbox_masks_from_targets(
                batch_idx=batch["batch_idx"],
                bboxes=batch["bboxes"],
                batch_size=batch["img"].shape[0],
                image_size=tuple(batch["img"].shape[-2:]),
                device=batch["img"].device,
            )
        return self.criterion(preds, batch)


class SRWLSalDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        self.custom_overrides = {
            "target_layers": overrides.pop("target_layers", "P3"),
            "saliency_provider": overrides.pop("saliency_provider", "saliency_head"),
            "teacher_dir": overrides.pop("teacher_dir", None),
            "beta_teacher": overrides.pop("beta_teacher", 0.0),
            "loss_type": overrides.pop("loss_type", "mse"),
            "lambda_sal": overrides.pop("lambda_sal", 0.1),
            "lambda_schedule": overrides.pop("lambda_schedule", "constant"),
            "lambda_max": overrides.pop("lambda_max", None),
            "lambda_min": overrides.pop("lambda_min", 0.0),
            "warmup_epochs": overrides.pop("warmup_epochs", 0),
            "beta_bg": overrides.pop("beta_bg", 0.5),
            "dilation_radius": overrides.pop("dilation_radius", 3),
            "size_aware": overrides.pop("size_aware", False),
            "size_weight_mode": overrides.pop("size_weight_mode", "log_inverse"),
            "size_weight_max": overrides.pop("size_weight_max", None),
            "sigma_ratio": overrides.pop("sigma_ratio", 0.04),
            "alpha_init": overrides.pop("alpha_init", 0.1),
        }
        super().__init__(cfg=cfg or DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)
        for key, value in self.custom_overrides.items():
            setattr(self.args, key, value)
        self._data_yaml_path = Path(str(self.args.data)).expanduser().resolve()
        saliency_provider = str(getattr(self.args, "saliency_provider", "saliency_head"))
        if saliency_provider != "saliency_head":
            raise ValueError("SRW + L_sal training currently supports only saliency_head as the online provider.")
        teacher_dir = getattr(self.args, "teacher_dir", None)
        beta_teacher = float(getattr(self.args, "beta_teacher", 0.0))
        if beta_teacher > 0.0 and not teacher_dir:
            raise ValueError("--teacher-dir is required when --beta-teacher > 0.")
        self.teacher_provider = (
            SaliencyProvider(mode="offline_xai_teacher", channels=1, teacher_dir=teacher_dir)
            if teacher_dir
            else None
        )
        self.lambda_history: list[dict[str, float]] = []

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
        batch["gt_bbox_mask"] = build_batch_bbox_masks_from_targets(
            batch_idx=batch["batch_idx"],
            bboxes=batch["bboxes"],
            batch_size=batch["img"].shape[0],
            image_size=tuple(batch["img"].shape[-2:]),
            device=self.device,
        )
        if "im_file" in batch:
            image_ids = [image_id_from_path(self._data_yaml_path, path) for path in batch["im_file"]]
            batch["image_ids"] = image_ids
            if self.teacher_provider is not None:
                batch["teacher_saliency_mask"] = self.teacher_provider(batch["img"][:, :1], image_ids=image_ids)
        return batch

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = SRWLSalDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            target_layer=str(getattr(self.args, "target_layers", "P3")),
            saliency_provider=str(getattr(self.args, "saliency_provider", "saliency_head")),
            alpha_init=float(getattr(self.args, "alpha_init", 0.1)),
            loss_type=str(getattr(self.args, "loss_type", "mse")),
            lambda_sal=float(getattr(self.args, "lambda_sal", 0.1)),
            beta_teacher=float(getattr(self.args, "beta_teacher", 0.0)),
            lambda_schedule=str(getattr(self.args, "lambda_schedule", "constant")),
            lambda_max=getattr(self.args, "lambda_max", None),
            lambda_min=float(getattr(self.args, "lambda_min", 0.0)),
            warmup_epochs=int(getattr(self.args, "warmup_epochs", 0)),
            beta_bg=float(getattr(self.args, "beta_bg", 0.5)),
            dilation_radius=int(getattr(self.args, "dilation_radius", 3)),
            size_aware=bool(getattr(self.args, "size_aware", False)),
            size_weight_mode=str(getattr(self.args, "size_weight_mode", "log_inverse")),
            size_weight_max=getattr(self.args, "size_weight_max", None),
            total_epochs=int(getattr(self.args, "epochs", 100)),
            sigma_ratio=float(getattr(self.args, "sigma_ratio", 0.04)),
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "sal_loss", "teacher_sal_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self._validator_args(),
            _callbacks=self.callbacks,
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def save_metrics(self, metrics):
        criterion = getattr(getattr(self, "model", None), "criterion", None)
        lambda_sal = float(getattr(criterion, "lambda_sal", getattr(self.args, "lambda_sal", 0.1)))
        beta_teacher = float(getattr(criterion, "beta_teacher", getattr(self.args, "beta_teacher", 0.0)))
        merged_metrics = dict(metrics)
        merged_metrics["lambda_sal"] = lambda_sal
        merged_metrics["beta_teacher"] = beta_teacher
        super().save_metrics(merged_metrics)
        self.lambda_history.append({"epoch": float(self.epoch + 1), "lambda_sal": lambda_sal})
        save_lambda_curve(
            self.lambda_history,
            output_csv=self.save_dir / "lambda_curve.csv",
            output_png=self.save_dir / "lambda_curve.png",
        )
