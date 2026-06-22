#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.runtime import configure_runtime_environment

configure_runtime_environment()

from src.trainers.srw_lsal_trainer import SRWLSalDetectionTrainer
from src.training.multiscale_srw import parse_scale_weights, parse_target_layers
from src.training.teacher_alignment import snapshot_teacher_augmentation_values
from src.utils.cli_config import namespace_to_config_reference, parse_args_with_optional_config
from src.utils.io import ensure_dir
from src.utils.logging import save_run_config, save_run_metrics, setup_logging
from src.utils.seed import seed_everything
from src.utils.train_runs import collect_train_metrics

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:  # pragma: no cover
    ULTRALYTICS_IMPORT_ERROR = None


def collect_extra_metrics(trainer) -> dict[str, object]:
    metrics: dict[str, object] = {}
    teacher_audit = getattr(trainer, "teacher_augmentation_audit", None)
    if teacher_audit is not None:
        metrics["teacher_augmentation_policy"] = getattr(teacher_audit, "applied_policy", None)
        metrics["teacher_incompatible_augmentations"] = list(getattr(teacher_audit, "incompatible_keys", []))
        metrics["teacher_disabled_augmentations"] = list(getattr(teacher_audit, "disabled_keys", []))

    debug_stats = getattr(getattr(trainer, "model", None), "last_srw_debug", None)
    if isinstance(debug_stats, dict):
        metrics["srw_debug"] = debug_stats
    return metrics


def parse_args() -> argparse.Namespace:
    default_data = os.environ.get("SKYFUSION_DATA") or (
        "/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml"
    )
    parser = argparse.ArgumentParser(description="Train YOLO with joint SRW + L_sal.")
    parser.add_argument("--data", type=Path, default=Path(default_data), help="Path to the YOLO dataset YAML.")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Ultralytics model checkpoint or config.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default=None, help="Optional Ultralytics device string.")
    parser.add_argument("--workers", type=int, default=None, help="Optional dataloader worker count.")
    parser.add_argument("--run-name", type=str, required=True, help="Run directory name under experiments/skyfusion.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/skyfusion"), help="Output root.")
    parser.add_argument(
        "--target-layers",
        nargs="+",
        default=["P3"],
        help="One or more target FPN levels, e.g. P3 or P3 P4 P5.",
    )
    parser.add_argument(
        "--scale-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-scale loss weights aligned with --target-layers.",
    )
    parser.add_argument("--saliency-provider", type=str, default="saliency_head", choices=("saliency_head",))
    parser.add_argument("--teacher-dir", type=Path, default=None, help="Optional offline teacher manifest directory.")
    parser.add_argument("--beta-teacher", type=float, default=0.0, help="Teacher loss weight.")
    parser.add_argument(
        "--teacher-augmentation-policy",
        type=str,
        default="error",
        choices=("error", "disable_incompatible"),
        help="How to handle augmentations that break offline teacher spatial alignment.",
    )
    parser.add_argument("--loss-type", type=str, default="mse", choices=("mse", "bce", "dice", "energy", "energy_bg"))
    parser.add_argument("--lambda-sal", type=float, default=0.1, help="Weight applied to GT saliency alignment loss.")
    parser.add_argument(
        "--lambda-schedule",
        type=str,
        default="constant",
        choices=("constant", "linear_warmup", "cosine_decay", "warmup_cosine_decay"),
        help="Lambda schedule mode for saliency loss.",
    )
    parser.add_argument("--lambda-max", type=float, default=None, help="Peak lambda for scheduled modes.")
    parser.add_argument("--lambda-min", type=float, default=0.0, help="Minimum lambda for decay schedules.")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs for lambda scheduling.")
    parser.add_argument("--beta-bg", type=float, default=0.5, help="Background suppression weight for energy_bg.")
    parser.add_argument("--dilation-radius", type=int, default=3, help="Ignore-mask dilation radius for energy_bg.")
    parser.add_argument("--size-aware", action="store_true", help="Enable image-level size-aware saliency weighting.")
    parser.add_argument(
        "--size-weight-mode",
        type=str,
        default="log_inverse",
        choices=("log_inverse", "inverse_sqrt"),
        help="Weighting mode when --size-aware is enabled.",
    )
    parser.add_argument("--size-weight-max", type=float, default=None, help="Optional clamp for size-aware weights.")
    parser.add_argument("--sigma-ratio", type=float, default=0.04, help="Gaussian sigma ratio for GT saliency masks.")
    parser.add_argument("--alpha-init", type=float, default=0.1, help="Initial SRW alpha.")
    return parse_args_with_optional_config(parser)


def collect_effective_run_config(model: Any, base_config: dict[str, Any]) -> dict[str, Any]:
    effective = dict(base_config)
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return effective

    teacher_audit = getattr(trainer, "teacher_augmentation_audit", None)
    if teacher_audit is not None:
        effective["teacher_augmentation_policy"] = getattr(teacher_audit, "applied_policy", None)
        effective["teacher_incompatible_augmentations"] = list(getattr(teacher_audit, "incompatible_keys", []))
        effective["teacher_disabled_augmentations"] = list(getattr(teacher_audit, "disabled_keys", []))

    trainer_args = getattr(trainer, "args", None)
    if trainer_args is not None:
        effective["effective_teacher_augmentations"] = snapshot_teacher_augmentation_values(trainer_args)
        effective["effective_target_layers"] = list(getattr(trainer_args, "target_layers", effective["target_layers"]))
        effective["effective_scale_weights"] = list(getattr(trainer_args, "scale_weights", effective["scale_weights"]))
    return effective


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    args.target_layers = parse_target_layers(args.target_layers)
    args.scale_weights = parse_scale_weights(args.scale_weights, num_layers=len(args.target_layers))

    if YOLO is None:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is not installed. Install project requirements first. "
            f"Original import error: {ULTRALYTICS_IMPORT_ERROR}"
        )
    if args.saliency_provider != "saliency_head":
        raise SystemExit("SRW + L_sal training currently supports only --saliency-provider saliency_head.")
    if args.beta_teacher > 0.0 and args.teacher_dir is None:
        raise SystemExit("--teacher-dir is required when --beta-teacher > 0.")

    data_yaml = args.data.expanduser()
    if not data_yaml.is_absolute():
        data_yaml = (Path.cwd() / data_yaml).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    run_dir = ensure_dir(output_root / args.run_name)

    teacher_dir = args.teacher_dir.expanduser().resolve() if args.teacher_dir is not None else None
    if teacher_dir is not None and not teacher_dir.is_dir():
        raise SystemExit(f"Teacher directory not found: {teacher_dir}")

    seed_everything(args.seed, deterministic=False)

    config = {
        "data": str(data_yaml),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "patience": args.patience,
        "device": args.device,
        "workers": args.workers,
        "run_name": args.run_name,
        "output_root": str(output_root),
        "config": namespace_to_config_reference(args),
        "target_layers": list(args.target_layers),
        "scale_weights": list(args.scale_weights),
        "saliency_provider": args.saliency_provider,
        "teacher_dir": str(teacher_dir) if teacher_dir else None,
        "beta_teacher": args.beta_teacher,
        "teacher_augmentation_policy": args.teacher_augmentation_policy,
        "loss_type": args.loss_type,
        "lambda_sal": args.lambda_sal,
        "lambda_schedule": args.lambda_schedule,
        "lambda_max": args.lambda_max,
        "lambda_min": args.lambda_min,
        "warmup_epochs": args.warmup_epochs,
        "beta_bg": args.beta_bg,
        "dilation_radius": args.dilation_radius,
        "size_aware": args.size_aware,
        "size_weight_mode": args.size_weight_mode,
        "size_weight_max": args.size_weight_max,
        "sigma_ratio": args.sigma_ratio,
        "alpha_init": args.alpha_init,
    }
    save_run_config(run_dir, config)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "patience": args.patience,
        "project": str(output_root),
        "name": args.run_name,
        "exist_ok": True,
        "target_layers": list(args.target_layers),
        "scale_weights": list(args.scale_weights),
        "saliency_provider": args.saliency_provider,
        "teacher_dir": str(teacher_dir) if teacher_dir else None,
        "beta_teacher": args.beta_teacher,
        "teacher_augmentation_policy": args.teacher_augmentation_policy,
        "loss_type": args.loss_type,
        "lambda_sal": args.lambda_sal,
        "lambda_schedule": args.lambda_schedule,
        "lambda_max": args.lambda_max,
        "lambda_min": args.lambda_min,
        "warmup_epochs": args.warmup_epochs,
        "beta_bg": args.beta_bg,
        "dilation_radius": args.dilation_radius,
        "size_aware": args.size_aware,
        "size_weight_mode": args.size_weight_mode,
        "size_weight_max": args.size_weight_max,
        "sigma_ratio": args.sigma_ratio,
        "alpha_init": args.alpha_init,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    logger.info("Starting SRW + L_sal training.")
    logger.info("Run directory: %s", run_dir)

    model = YOLO(args.model)
    model.train(trainer=SRWLSalDetectionTrainer, **train_kwargs)

    effective_config = collect_effective_run_config(model=model, base_config=config)
    save_run_config(run_dir, effective_config)
    metrics = collect_train_metrics(
        model=model,
        run_dir=run_dir,
        extra_collector=collect_extra_metrics,
    )
    save_run_metrics(run_dir, metrics)
    logger.info("SRW + L_sal training finished.")


if __name__ == "__main__":
    main()
