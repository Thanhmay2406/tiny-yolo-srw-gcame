# Project Handoff — 2026-06-21 (Phase 14 Ready)

This document records the project state after completing and validating:

- baseline YOLO training
- traditional augmentation baseline
- GT saliency mask generation and debug tools
- saliency provider and offline teacher debug tools
- SRW standalone module and YOLO hook debug
- `L_sal-only` training
- `SRW-only` training
- `SRW + L_sal` joint training
- lambda scheduling for `SRW + L_sal`
- advanced saliency-loss variants: `energy`, `energy_bg`, `size-aware`

Use this file as the primary resume point for the next session.

## 1. Dataset and path conventions

Current Kaggle source-of-truth path:

```text
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
```

Important rules already enforced in code and docs:

- do not write to `/kaggle/input`
- reconverted dataset, if ever needed, must go to `/kaggle/working/data/SkyFusion_yolo_reconverted`
- experiment outputs go under `experiments/skyfusion/<run_name>/`
- baseline scripts must not import SRW or `L_sal`

Current local dataset copy used for smoke/debug:

```text
data/SkyFusion_yolo/data.yaml
```

## 2. Repository status summary

Implemented enough to run:

- Phase 4: baseline training
- Phase 5: traditional augmentation baseline
- Phase 6: GT saliency mask generation
- Phase 7: saliency provider + saliency head + teacher precompute debug
- Phase 8: standalone SRW module
- Phase 10: `L_sal-only` training
- Phase 11: `SRW-only` training
- Phase 12: `SRW + L_sal` joint training
- Phase 13: lambda scheduling for `SRW + L_sal`
- Phase 14: `energy` / `energy_bg` / `size-aware` saliency-loss ablations

Still not implemented:

- Phase 15: optional multi-scale `SRW + L_sal`
- Phase 16+: consolidated evaluation/report tooling

## 3. Key files added or materially changed

### Training scripts

- `scripts/10_train_lsal_only.py`
- `scripts/11_train_srw_only.py`
- `scripts/12_train_srw_lsal.py`

### Trainers and training utilities

- `src/trainers/lsal_trainer.py`
- `src/trainers/srw_trainer.py`
- `src/trainers/srw_lsal_trainer.py`
- `src/training/lambda_scheduler.py`

### Losses

- `src/losses/saliency_alignment.py`
- `src/losses/energy_in_box.py`
- `src/losses/background_suppression.py`
- `src/losses/size_aware.py`

### Dataset helpers

- `src/datasets/saliency_masks.py`
- `src/datasets/yolo_dataset.py`

### Configs

- `configs/train/lsal_only_yolov8s.yaml`
- `configs/train/srw_only_yolov8s.yaml`
- `configs/train/srw_lsal_yolov8s.yaml`

### Tests

- `tests/test_saliency_masks.py`
- `tests/test_saliency_losses.py`
- `tests/test_saliency_head.py`
- `tests/test_srw.py`
- `tests/test_srw_lsal_trainer.py`
- `tests/test_lambda_scheduler.py`
- `tests/test_energy_bg_losses.py`

### Docs

- `README.md`

## 4. Important implementation details

### 4.1 `SRW + L_sal` trainer

Implemented in:

- `src/trainers/srw_lsal_trainer.py`
- `scripts/12_train_srw_lsal.py`

Current design:

- capture YOLO neck feature at target layer `P3`
- produce `saliency_pred` from a differentiable saliency head
- apply `SRW` on the same feature map
- optimize:

```text
L_total = L_det + lambda_sal * L_sal + beta_teacher * L_teacher
```

Current training support:

- `saliency_provider=saliency_head`
- optional `offline_xai_teacher`
- `loss_type=mse|bce|dice|energy|energy_bg`

### 4.2 Lambda scheduling

Implemented in `src/training/lambda_scheduler.py`.

Supported modes:

- `constant`
- `linear_warmup`
- `cosine_decay`
- `warmup_cosine_decay`

Integrated behavior:

- current `lambda_sal` is logged into `results.csv`
- `lambda_curve.csv` and `lambda_curve.png` are exported in each run directory

### 4.3 Advanced saliency loss variants

Implemented in:

- `src/losses/energy_in_box.py`
- `src/losses/background_suppression.py`
- `src/losses/size_aware.py`

Notes:

- `energy` encourages saliency energy to stay inside bbox masks
- `energy_bg` adds background suppression outside a dilated ignore region
- `size-aware` applies image-level weighting so tiny-object images contribute more
- hard bbox masks are now generated in `src/datasets/saliency_masks.py` for these variants
- validation path was patched so `gt_bbox_mask` is auto-generated even when the validator calls `model.loss()` directly

### 4.4 Current limitations

- `GCAME` is still a placeholder
- `gradcam_like_online_debug` is only a lightweight debug surrogate
- multi-scale `P3/P4/P5` training is not implemented yet
- advanced losses are implemented and locally smoke-tested, but full Kaggle ablation runs have not been recorded yet

## 5. Local verification done

Current local verification:

```bash
.venv/bin/pytest -q
```

Status at handoff:

- `20 passed`

Successful local smoke/debug runs:

```bash
.venv/bin/python scripts/10_train_lsal_only.py --data /tmp/skyfusion_mini/data.yaml --model yolov8n.yaml --epochs 1 --imgsz 160 --batch 1 --workers 0 --device cpu --run-name smoke_lsal_only_local --output-root experiments/skyfusion

.venv/bin/python scripts/11_train_srw_only.py --data /tmp/skyfusion_mini/data.yaml --model yolov8n.yaml --epochs 1 --imgsz 160 --batch 1 --workers 0 --device cpu --run-name smoke_srw_only_local --output-root experiments/skyfusion

.venv/bin/python scripts/12_train_srw_lsal.py --data /tmp/skyfusion_mini/data.yaml --model yolov8n.yaml --epochs 1 --imgsz 160 --batch 1 --workers 0 --device cpu --run-name smoke_srw_lsal_local --output-root experiments/skyfusion

.venv/bin/python scripts/12_train_srw_lsal.py --data /tmp/skyfusion_mini/data.yaml --model yolov8n.yaml --epochs 3 --imgsz 160 --batch 1 --workers 0 --device cpu --run-name smoke_srw_lsal_schedule_local --output-root experiments/skyfusion --lambda-schedule warmup_cosine_decay --lambda-max 0.2 --lambda-min 0.01 --warmup-epochs 1

.venv/bin/python scripts/12_train_srw_lsal.py --data /tmp/skyfusion_mini/data.yaml --model yolov8n.yaml --epochs 1 --imgsz 160 --batch 1 --workers 0 --device cpu --run-name smoke_srw_lsal_energy_bg_local --output-root experiments/skyfusion --loss-type energy_bg --beta-bg 0.5 --dilation-radius 1 --size-aware --size-weight-mode log_inverse --size-weight-max 5.0
```

## 6. Kaggle experiment results currently available

### Baseline

Run:

```text
experiments/skyfusion/baseline_yolov8s
```

Key metrics:

- `precision`: `0.70789`
- `recall`: `0.66708`
- `mAP50`: `0.65716`
- `mAP50-95`: `0.35906`

### Traditional augmentation baseline

Run:

```text
experiments/skyfusion/tradaug_yolov8s_seed0
```

Key metrics:

- `precision`: `0.71160`
- `recall`: `0.65333`
- `mAP50`: `0.65608`
- `mAP50-95`: `0.35755`

### `L_sal-only`

Run:

```text
experiments/skyfusion/lsal_only_mse_seed0
```

Key metrics:

- `precision`: `0.70588`
- `recall`: `0.65533`
- `mAP50`: `0.65501`
- `mAP50-95`: `0.35867`

### `SRW-only`

Run:

```text
experiments/skyfusion/srw_only_p3_seed0
```

Key metrics:

- `precision`: `0.70896`
- `recall`: `0.67117`
- `mAP50`: `0.66477`
- `mAP50-95`: `0.36226`

### `SRW + L_sal` with constant lambda

Run:

```text
experiments/skyfusion/srw_lsal_p3_mse_seed0
```

Key metrics:

- `precision`: `0.72818`
- `recall`: `0.66510`
- `mAP50`: `0.66081`
- `mAP50-95`: `0.36117`

### `SRW + L_sal` with `warmup_cosine_decay`

Run:

```text
experiments/skyfusion/srw_lsal_p3_warmup_decay_seed0
```

Key metrics:

- `precision`: `0.71585`
- `recall`: `0.67252`
- `mAP50`: `0.66400`
- `mAP50-95`: `0.36309`
- final `lambda_sal`: `0.01`

Interpretation so far:

- `warmup_cosine_decay` improved over constant-lambda `SRW + L_sal`
- `warmup_cosine_decay` also slightly exceeded the current `SRW-only` run on `mAP50-95`
- this is enough reason to continue to Phase 14 Kaggle ablations next

## 7. Commands to run next on Kaggle

### 7.1 `energy`

```bash
python scripts/12_train_srw_lsal.py \
  --data /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_energy_seed0 \
  --target-layers P3 \
  --loss-type energy \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5 \
  --alpha-init 0.1
```

### 7.2 `energy_bg`

```bash
python scripts/12_train_srw_lsal.py \
  --data /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_energy_bg_seed0 \
  --target-layers P3 \
  --loss-type energy_bg \
  --beta-bg 0.5 \
  --dilation-radius 3 \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5 \
  --alpha-init 0.1
```

### 7.3 `energy_bg + size-aware`

```bash
python scripts/12_train_srw_lsal.py \
  --data /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_energy_bg_sizeaware_seed0 \
  --target-layers P3 \
  --loss-type energy_bg \
  --beta-bg 0.5 \
  --dilation-radius 3 \
  --size-aware \
  --size-weight-mode log_inverse \
  --size-weight-max 5.0 \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5 \
  --alpha-init 0.1
```

## 8. Recommended next-session checklist

1. Read this file first.
2. Compare these runs first:
   - `baseline_yolov8s`
   - `srw_only_p3_seed0`
   - `srw_lsal_p3_mse_seed0`
   - `srw_lsal_p3_warmup_decay_seed0`
3. Run Phase 14 Kaggle ablations:
   - `srw_lsal_energy_seed0`
   - `srw_lsal_energy_bg_seed0`
   - `srw_lsal_energy_bg_sizeaware_seed0`
4. If Phase 14 is stable, proceed to Phase 15 optional multi-scale support.
5. Keep baseline and tradaug scripts untouched for fairness.

## 9. Minimal resume prompt

If another coding session needs to continue from here, this prompt is enough:

```text
Read docs/project_handoff_2026-06-21_phase14.md first.
Current completed phases: baseline, tradaug, GT saliency masks, saliency provider debug, SRW module, L_sal-only trainer, SRW-only trainer, SRW+L_sal trainer, lambda scheduling, and Phase 14 advanced saliency losses.
Best current Kaggle result among saliency variants is srw_lsal_p3_warmup_decay_seed0 with mAP50-95 = 0.36309.
Next target: run and analyze Phase 14 Kaggle ablations (energy, energy_bg, energy_bg + size-aware), then continue to optional Phase 15 multi-scale support if needed.
```
