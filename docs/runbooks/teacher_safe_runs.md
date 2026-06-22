# Offline Teacher Safety Notes

This repository supports optional offline XAI teacher supervision in `scripts/12_train_srw_lsal.py`.

## Why a safety policy is needed

The offline teacher manifest maps each `image_id` to a precomputed saliency array.

That teacher saliency is valid only if the training image still represents the same spatial content.

The following augmentations break that assumption because they either mix multiple images or move object geometry:

- `mosaic`
- `mixup`
- `copy_paste`
- `degrees`
- `translate`
- `scale`
- `shear`
- `perspective`
- `fliplr`
- `flipud`

Without protection, `beta_teacher > 0` would produce spatially misaligned supervision.

## Current repository policy

When `--beta-teacher > 0` and `--teacher-dir` is set:

- `--teacher-augmentation-policy error`
  Fails fast if any incompatible augmentation is active.
- `--teacher-augmentation-policy disable_incompatible`
  Forces the incompatible augmentations above to `0.0` for a teacher-safe run.

## Audit artifacts

Each `SRW + L_sal` run now records:

- `metrics.json`
  Includes `teacher_augmentation_policy`, `teacher_incompatible_augmentations`, and `teacher_disabled_augmentations`.
- `results.csv`
  Includes numeric counts only:
  `teacher_incompatible_augmentation_count` and `teacher_disabled_augmentation_count`.
- `config.yaml`
  Rewritten after trainer initialization to include effective teacher policy and effective augmentation values.

## Kaggle-safe commands

P3-only teacher-safe run:

```bash
python scripts/12_train_srw_lsal.py \
  --data /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_p3_teacher_safe_seed0 \
  --target-layers P3 \
  --teacher-dir experiments/skyfusion/xai_teacher/baseline_p3_train \
  --beta-teacher 0.1 \
  --teacher-augmentation-policy disable_incompatible \
  --loss-type mse \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5 \
  --alpha-init 0.1
```

Multi-scale teacher-safe run:

```bash
python scripts/12_train_srw_lsal.py \
  --data /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_multiscale_teacher_safe_seed0 \
  --target-layers P3 P4 P5 \
  --scale-weights 1.0 0.5 0.25 \
  --teacher-dir experiments/skyfusion/xai_teacher/baseline_p3_train \
  --beta-teacher 0.1 \
  --teacher-augmentation-policy disable_incompatible \
  --loss-type mse \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5 \
  --alpha-init 0.1
```
