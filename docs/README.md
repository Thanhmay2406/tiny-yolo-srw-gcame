# Docs Index

- [Roadmap](./roadmap.md)
- [Roadmap Vietnamese](./roadmap_vietnamese.md)
- [Dataset Structure](./skyfusion_dataset_structure.md)
- [Status Summary](./skyfusion_status_summary.md)

## Handoffs

- [2026-06-21](/home/thanhmay/workspace/tiny-yolo-srw-gcame/docs/handoffs/project_handoff_2026-06-21.md:1)
- [2026-06-21 Phase 14](/home/thanhmay/workspace/tiny-yolo-srw-gcame/docs/handoffs/project_handoff_2026-06-21_phase14.md:1)

## Runbooks

- [Offline Teacher Safety](/home/thanhmay/workspace/tiny-yolo-srw-gcame/docs/runbooks/teacher_safe_runs.md:1)

## Common Commands

Traditional augmentation baseline:

```bash
python scripts/06_train_tradaug.py \
  --config configs/train/tradaug_yolov8s.yaml \
  --run-name tradaug_yolov8s_seed0
```

GT saliency debug:

```bash
python scripts/07_debug_gt_saliency.py \
  --data "$SKYFUSION_DATA" \
  --split train \
  --num-samples 16 \
  --output-dir experiments/skyfusion/debug_gt_saliency
```

Offline teacher precompute:

```bash
python scripts/08b_precompute_xai_teacher.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/baseline_yolov8s/weights/best.pt \
  --split train \
  --target-layers P3 \
  --xai-method gradcam_like \
  --output-dir experiments/skyfusion/xai_teacher/baseline_p3_train
```

Teacher-safe `SRW + L_sal`:

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/train/srw_lsal_teacher_safe_yolov8s.yaml \
  --run-name srw_lsal_p3_teacher_safe_seed0
```

Teacher-safe multi-scale `SRW + L_sal`:

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/train/srw_lsal_multiscale_teacher_safe_yolov8s.yaml \
  --run-name srw_lsal_multiscale_teacher_safe_seed0
```
