# Missing Detection Evaluation Commands

Runbook này liệt kê các run chính hiện còn thiếu `detection_eval/metrics.json` hoặc chưa nên hoàn tất trên local CPU.

## Kaggle environment

```bash
export SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
```

## Ghi chú

- `scripts/13_eval_detection.py` hiện đã được vá bug nhỏ để map `--split valid` sang `val` cho `YOLO.val()` của Ultralytics.
- Local CPU có thể chạy, nhưng rất chậm trên full split; vì vậy các lệnh dưới đây nên chạy trên Kaggle GPU/CPU runtime theo source-of-truth dataset path.

## Commands

### `tradaug_yolov8s_seed0`

```bash
python scripts/13_eval_detection.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/tradaug_yolov8s_seed0/weights/best.pt \
  --split valid \
  --run-name tradaug_yolov8s_seed0
```

### `lsal_only_mse_seed0`

```bash
python scripts/13_eval_detection.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/lsal_only_mse_seed0/weights/best.pt \
  --split valid \
  --run-name lsal_only_mse_seed0
```

### `srw_lsal_p3_mse_seed0`

```bash
python scripts/13_eval_detection.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_p3_mse_seed0/weights/best.pt \
  --split valid \
  --run-name srw_lsal_p3_mse_seed0
```

### `srw_lsal_energy_seed0`

```bash
python scripts/13_eval_detection.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_energy_seed0/weights/best.pt \
  --split valid \
  --run-name srw_lsal_energy_seed0
```

### `srw_lsal_energy_bg_sizeaware_seed0`

```bash
python scripts/13_eval_detection.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_energy_bg_sizeaware_seed0/weights/best.pt \
  --split valid \
  --run-name srw_lsal_energy_bg_sizeaware_seed0
```
