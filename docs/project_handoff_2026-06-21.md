# Project Handoff — 2026-06-21

This document records the exact project state after implementing and debugging the repo through:

- baseline YOLO training
- traditional augmentation baseline
- GT saliency mask generation and debug tools
- saliency provider and offline teacher debug tools
- SRW standalone module and YOLO hook debug
- `L_sal-only` training
- `SRW-only` training

The goal is to make it easy to resume the project later without re-discovering context.

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

The repo started as a mostly Phase-0 Kaggle bootstrap project. The following phases are now implemented enough to run:

- Phase 4: baseline training
- Phase 5: traditional augmentation baseline
- Phase 6: GT saliency mask generation
- Phase 7: saliency provider + saliency head + teacher precompute debug
- Phase 8: standalone SRW module
- Phase 10: `L_sal-only` training
- Phase 11: `SRW-only` training

The following phase is still not implemented as a real trainer:

- Phase 12: `SRW + L_sal` joint training

The following is partially implemented only as debug infrastructure:

- Phase 9: YOLO-SRW integration exists for debug and now also supports `SRW-only` train path through a custom model, but there is not yet a joint `SRW + L_sal` trainer

## 3. Files added or materially changed

### Training scripts

- `scripts/05_train_baseline.py`
- `scripts/06_train_tradaug.py`
- `scripts/07_debug_gt_saliency.py`
- `scripts/08_debug_xai_saliency.py`
- `scripts/08b_precompute_xai_teacher.py`
- `scripts/09_debug_srw_shapes.py`
- `scripts/10_train_lsal_only.py`
- `scripts/11_train_srw_only.py`

### Core modules

- `src/datasets/yolo_dataset.py`
- `src/datasets/saliency_masks.py`
- `src/losses/saliency_alignment.py`
- `src/models/layer_resolver.py`
- `src/models/srw.py`
- `src/models/yolo_srw_wrapper.py`
- `src/trainers/lsal_trainer.py`
- `src/trainers/srw_trainer.py`
- `src/xai/saliency_base.py`
- `src/xai/saliency_head.py`
- `src/xai/saliency_provider.py`
- `src/xai/gradcam_detector.py`
- `src/xai/gcame_detector.py`
- `src/xai/hooks.py`
- `src/xai/saliency_normalization.py`
- `src/utils/runtime.py`

### Configs

- `configs/train/baseline_yolov8s.yaml`
- `configs/train/tradaug_yolov8s.yaml`
- `configs/train/lsal_only_yolov8s.yaml`
- `configs/train/srw_only_yolov8s.yaml`
- `configs/dataset/skyfusion.yaml`

### Tests

- `tests/test_saliency_masks.py`
- `tests/test_saliency_head.py`
- `tests/test_saliency_losses.py`
- `tests/test_srw.py`
- `tests/conftest.py`

### README

- `README.md` was updated to reflect current implemented phases and commands

## 4. Important implementation details

### 4.1 Baseline and TradAug

`scripts/05_train_baseline.py`

- clean baseline
- no SRW import
- no `L_sal` import
- uses Ultralytics early stopping through `--patience`
- fixed fallback handling so an empty `SKYFUSION_DATA` environment variable no longer resolves to `/kaggle/working`

`scripts/06_train_tradaug.py`

- same base interface as baseline
- adds explicit augmentation knobs:
  - `mosaic`
  - `mixup`
  - `copy_paste`
  - `hsv_h`
  - `hsv_s`
  - `hsv_v`
  - `scale`
  - `translate`
  - `fliplr`
- also uses Ultralytics `--patience`

### 4.2 GT saliency masks

Implemented in `src/datasets/saliency_masks.py`.

Key functions:

- `read_yolo_label_file`
- `yolo_boxes_to_pixel_boxes`
- `create_bbox_mask`
- `create_gaussian_bbox_mask`
- `resize_mask_to_feature`
- `build_batch_gaussian_masks_from_targets`

Design notes:

- empty labels return zero masks
- masks are normalized into `[0, 1]`
- Gaussian blur is built on top of bbox hard masks
- resizing uses bilinear interpolation

### 4.3 Saliency provider stack

Implemented in `src/xai/`.

Current provider modes:

- `saliency_head`
- `gt_mask_debug`
- `offline_xai_teacher`
- `gradcam_like_online_debug`
- `gcame_placeholder`

Important limitation:

- `gradcam_like_online_debug` is a lightweight surrogate for debugging and teacher precompute
- it is not a full detection-target Grad-CAM implementation
- `gcame_placeholder` intentionally raises `NotImplementedError`

### 4.4 SRW module

Implemented in `src/models/srw.py`.

Current design:

- spatial gate from saliency map
- channel gate from pooled feature statistics
- residual form:

```text
F* = F + alpha * (F * G_s * G_c)
```

Debug support:

- optional return of `gate_s`, `gate_c`, and `alpha`

### 4.5 `L_sal-only` trainer

Implemented in:

- `src/trainers/lsal_trainer.py`
- `scripts/10_train_lsal_only.py`

Approach:

- custom `DetectionModel` subclass injects `saliency_pred`
- custom criterion extends detection loss and adds `sal_loss`
- `gt_saliency_mask` is generated automatically from batch boxes
- trainer strips custom keys before handing args to Ultralytics validator to avoid config validation crashes

This path was locally smoke-tested end-to-end with:

- train
- validation
- checkpoint saving
- final best-weight validation

### 4.6 `SRW-only` trainer

Implemented in:

- `src/trainers/srw_trainer.py`
- `scripts/11_train_srw_only.py`

Approach:

- custom `DetectionModel` subclass intercepts target feature map `P3`
- saliency is produced by `SaliencyProvider`
- SRW is applied in-place to the selected feature map before continuing to the detection head
- no saliency loss is added
- debug stats are stored in `model.last_srw_debug`

Current supported provider:

- `saliency_head` only

Not yet supported in real training:

- `offline_xai_teacher`

This path was also locally smoke-tested end-to-end with:

- train
- validation
- checkpoint saving
- final best-weight validation

## 5. Local verification done

### Unit tests

The following currently pass:

```bash
.venv/bin/pytest -q
```

At the time of handoff:

- 10 tests passed

Covered areas:

- saliency mask generation
- saliency head shape/range/gradient
- saliency losses
- SRW shape/range/gradient

### Smoke runs performed locally

Successful local smoke/debug commands:

```bash
.venv/bin/python scripts/07_debug_gt_saliency.py --data data/SkyFusion_yolo/data.yaml --split valid --num-samples 2 --output-dir experiments/skyfusion/debug_gt_saliency_local

.venv/bin/python scripts/08_debug_xai_saliency.py --data data/SkyFusion_yolo/data.yaml --split valid --target-layers P3 --saliency-provider saliency_head --weights yolov8s.yaml --num-samples 2 --imgsz 320 --output-dir experiments/skyfusion/debug_saliency_head_local

.venv/bin/python scripts/08b_precompute_xai_teacher.py --data data/SkyFusion_yolo/data.yaml --weights yolov8s.yaml --split valid --target-layers P3 --xai-method gradcam_like --imgsz 320 --limit 2 --output-dir experiments/skyfusion/xai_teacher_local_smoke

.venv/bin/python scripts/09_debug_srw_shapes.py --from-yolo --model yolov8s.yaml --target-layers P3 --saliency-provider saliency_head --imgsz 320 --output-dir experiments/skyfusion/debug_srw_yolo_shapes_local
```

Successful local mini-train smoke runs:

```bash
.venv/bin/python scripts/10_train_lsal_only.py --data /tmp/skyfusion_mini/data.yaml --model yolov8n.yaml --epochs 1 --imgsz 160 --batch 1 --workers 0 --device cpu --run-name smoke_lsal_only_local --output-root experiments/skyfusion

.venv/bin/python scripts/11_train_srw_only.py --data /tmp/skyfusion_mini/data.yaml --model yolov8n.yaml --epochs 1 --imgsz 160 --batch 1 --workers 0 --device cpu --run-name smoke_srw_only_local --output-root experiments/skyfusion
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

### L_sal-only

Run:

```text
experiments/skyfusion/lsal_only_mse_seed0
```

Key metrics:

- `precision`: `0.70588`
- `recall`: `0.65533`
- `mAP50`: `0.65501`
- `mAP50-95`: `0.35867`
- `train/sal_loss`: `0.00052`
- `val/sal_loss`: `0.00066`

Interpretation:

- `L_sal-only` is essentially on par with the clean baseline
- no clear gain over baseline yet
- this is enough reason to continue to `SRW-only` and then `SRW + L_sal`

### SRW-only

At handoff time:

- train script is implemented and locally smoke-tested
- full Kaggle run has not yet been recorded into the repo metrics

## 7. Commands to run next on Kaggle

### 7.1 SRW-only full run

```bash
python scripts/11_train_srw_only.py \
  --data /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_only_p3_seed0 \
  --target-layers P3 \
  --saliency-provider saliency_head \
  --alpha-init 0.1
```

### 7.2 Optional SRW-only smoke test first

```bash
python scripts/11_train_srw_only.py \
  --data /kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml \
  --model yolov8n.yaml \
  --epochs 1 \
  --imgsz 320 \
  --batch 4 \
  --workers 2 \
  --run-name smoke_srw_only
```

### 7.3 After SRW-only, next target should be Phase 12

Planned next implementation:

- `scripts/12_train_srw_lsal.py`
- joint trainer that:
  - uses SRW on P3
  - predicts `saliency_pred`
  - computes `det_loss + lambda_sal * L_sal`

## 8. Remaining technical backlog

### High priority

- implement `SRW + L_sal` joint trainer
- verify whether SRW actually improves over baseline or just perturbs P3 without gain
- add optional logging of SRW gate statistics into `results.csv`
- add optional visual export of saliency and SRW gate overlays during training

### Medium priority

- support `offline_xai_teacher` for `SRW-only` and later `SRW + L_sal`
- implement lambda scheduling for `L_sal`
- add richer saliency loss options beyond the current `mse/bce/dice` baseline

### Lower priority

- refresh docs that still read like the old Phase-0-only repo
- reconcile older roadmap text with current implemented code
- add an experiment comparison summary script

## 9. Known limitations and caveats

- `GCAME` is still a placeholder
- `gradcam_like_online_debug` is not a full research-grade Grad-CAM for detection
- `offline_xai_teacher` infrastructure exists for debug/precompute, but is not yet integrated into the train loop
- current `SRW-only` trainer only supports `saliency_head`
- current `L_sal-only` trainer uses bbox-derived Gaussian masks only
- no full `SRW + L_sal` trainer exists yet

## 10. State of the worktree

At the time of writing this handoff:

- local `data/` remains untracked
- experiment outputs under `experiments/skyfusion/` include both real Kaggle runs and local smoke/debug artifacts

Before committing or packaging later, decide whether to:

- keep only scripts/config/tests in git
- exclude local smoke outputs
- retain Kaggle metrics JSON files as reproducibility artifacts

## 11. Recommended next-session checklist

When resuming the project:

1. Read this file first.
2. Check `experiments/skyfusion/srw_only_p3_seed0/` if the Kaggle run has already been completed.
3. Compare `baseline`, `tradaug`, `lsal_only`, and `srw_only` metrics.
4. If `SRW-only` is stable, implement `SRW + L_sal`.
5. Keep baseline and tradaug scripts untouched for fair comparison.

## 12. Minimal resume prompt

If another coding session needs to continue from here, this prompt is enough:

```text
Read docs/project_handoff_2026-06-21.md first.
Current completed phases: baseline, tradaug, GT saliency masks, saliency provider debug, SRW module, L_sal-only trainer, SRW-only trainer.
Next target: implement and debug Phase 12 SRW + L_sal joint training on top of the current custom Ultralytics trainer approach, without modifying site-packages.
```
