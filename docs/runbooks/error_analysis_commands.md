# Error Analysis Commands

Runbook này chuẩn bị hai cặp error analysis chính giữa baseline và các candidate mạnh nhất hiện tại.

## Kaggle environment

```bash
export SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
```

## Commands

### Baseline vs `srw_lsal_energy_bg_seed0`

```bash
python scripts/16_error_analysis.py \
  --dataset-yaml "$SKYFUSION_DATA" \
  --baseline-run baseline_yolov8s \
  --candidate-run srw_lsal_energy_bg_seed0 \
  --split val \
  --max-samples 128 \
  --device cuda:0 \
  --output-dir experiments/skyfusion/srw_lsal_energy_bg_seed0/error_analysis_vs_baseline
```

### Baseline vs `srw_lsal_p3_warmup_decay_seed0`

```bash
python scripts/16_error_analysis.py \
  --dataset-yaml "$SKYFUSION_DATA" \
  --baseline-run baseline_yolov8s \
  --candidate-run srw_lsal_p3_warmup_decay_seed0 \
  --split val \
  --max-samples 128 \
  --device cuda:0 \
  --output-dir experiments/skyfusion/srw_lsal_p3_warmup_decay_seed0/error_analysis_vs_baseline
```

## Ghi chú

- Error analysis hiện là khung nhẹ nhưng chạy được, ưu tiên CSV/JSON summary và một số image case study.
- Nếu candidate run đã có `xai_eval/` tương ứng, script sẽ cố gắng ghép overlay để tạo ảnh so sánh giàu ngữ cảnh hơn.
