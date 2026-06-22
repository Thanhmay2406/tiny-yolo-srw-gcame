# Missing XAI Evaluation Commands

Runbook này liệt kê các run saliency-guided chính hiện còn thiếu `xai_eval/`.

## Kaggle environment

```bash
export SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
```

## Ghi chú

- `scripts/14_eval_xai.py` hiện đã có handling rõ hơn cho:
  - missing checkpoint
  - sample lỗi
  - partial success
- Các lệnh dưới đây nên chạy trên Kaggle thay vì local CPU nếu muốn hoàn tất nhanh trên full validation split.

## Commands

### `lsal_only_mse_seed0`

```bash
python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/lsal_only_mse_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --device cuda:0 \
  --output-dir experiments/skyfusion/lsal_only_mse_seed0/xai_eval
```

### `srw_lsal_p3_mse_seed0`

```bash
python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_p3_mse_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --device cuda:0 \
  --output-dir experiments/skyfusion/srw_lsal_p3_mse_seed0/xai_eval
```

### `srw_lsal_energy_seed0`

```bash
python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_energy_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --device cuda:0 \
  --output-dir experiments/skyfusion/srw_lsal_energy_seed0/xai_eval
```

### `srw_lsal_energy_bg_sizeaware_seed0`

```bash
python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_energy_bg_sizeaware_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --device cuda:0 \
  --output-dir experiments/skyfusion/srw_lsal_energy_bg_sizeaware_seed0/xai_eval
```
