# V2 Tiny-Aware P2/P2+P3 Experiment Runbook

## Goal

Muc tieu cua giai doan V2 la kiem chung gia thuyet rang `P3-only` saliency supervision co the qua tho cho tiny objects, va `P2` hoac `P2+P3` tiny-aware supervision co the phu hop hon.

Runbook nay uu tien:

- tach `architecture effect` va `method effect`
- giu narrative an toan theo huong `saliency-guided SRW + L_sal`
- khong claim `G-CAME` la production implementation
- khong uu tien `teacher-safe` trong vong chay V2 dau tien

Dataset mac dinh:

```bash
export SKYFUSION_DATA=/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
```

## Recommended Run Order

### 1. P2 architecture baseline

```bash
python scripts/05_train_baseline.py \
  --config configs/skyfusion_v2_tiny_aware/baseline_yolov8s_p2_seed0.yaml \
  --run-name baseline_yolov8s_p2_seed0
```

### 2. Main V2 candidate: P2+P3, MSE, size-aware

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_mse_sizeaware_seed0
```

### 3. P2-only ablation

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2_mse_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2_mse_sizeaware_seed0
```

### 4. No-size-aware ablation

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_no_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_mse_no_sizeaware_seed0
```

### 5. Optional later: energy_bg size-aware

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_energy_bg_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_energy_bg_sizeaware_seed0
```

## Evaluation Commands

Detection evaluation:

```bash
python scripts/13_eval_detection.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/<run_name>/weights/best.pt \
  --split valid
```

XAI evaluation:

Single-layer examples:

```bash
python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_p2_mse_sizeaware_seed0/weights/best.pt \
  --split valid \
  --target-layers P2 \
  --output-dir experiments/skyfusion/srw_lsal_p2_mse_sizeaware_seed0/xai_eval

python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_p3_mse_sizeaware_seed0/weights/best.pt \
  --split valid \
  --target-layers P3 \
  --output-dir experiments/skyfusion/srw_lsal_p3_mse_sizeaware_seed0/xai_eval
```

Multi-layer example:

```bash
python scripts/14_eval_xai.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/srw_lsal_p2p3_mse_sizeaware_seed0/weights/best.pt \
  --split valid \
  --target-layers P2 P3 \
  --output-dir experiments/skyfusion/srw_lsal_p2p3_mse_sizeaware_seed0/xai_eval
```

Note:

- `scripts/14_eval_xai.py` hien tai co support `--target-layers P2 P3` o muc hook/eval.
- Cac multi-layer runs se ghi `per_layer` summary rieng trong `xai_metrics.json`.
- Neu checkpoint hoac layer khong hop le voi graph, script se fail sớm; khong nen tu sua training logic trong giai doan prep nay.

## Decision Rule

- Neu `baseline_yolov8s_p2_seed0` cai thien `Recall_tiny` so voi `baseline_yolov8s`, thi `P2` architecture co ich.
- Neu `srw_lsal_p2p3_mse_sizeaware_seed0` cai thien so voi `baseline_yolov8s_p2_seed0`, thi method co ich trong family `P2`.
- Neu `srw_lsal_p2p3_mse_sizeaware_seed0` tot hon `srw_lsal_p2p3_mse_no_sizeaware_seed0`, thi `size_aware` co ich.
- Neu `mAP` tang nhung `Recall_tiny` giam, phai dien giai la `trade-off`, khong claim tiny-object improvement.

## Comparison Structure

So sanh dung:

- old family detection reference:
  - `baseline_yolov8s`
  - `srw_lsal_energy_bg_seed0`
- P2 architecture control:
  - `baseline_yolov8s_p2_seed0`
- V2 method runs:
  - `srw_lsal_p2p3_mse_sizeaware_seed0`
  - `srw_lsal_p2_mse_sizeaware_seed0`
  - `srw_lsal_p2p3_mse_no_sizeaware_seed0`
  - `srw_lsal_p2p3_energy_bg_sizeaware_seed0`

Nguyen tac:

- So sanh V2 methods chu yeu voi `baseline_yolov8s_p2_seed0`.
- Khong dien giai gain cua `P2` candidates nhu the hoan toan do `SRW + L_sal` neu chua tach architecture effect.

## Metrics To Watch First

Thu tu uu tien:

1. `Recall_tiny`
2. `Recall_small`
3. `mAP50-95`
4. `mAP50`
5. XAI metrics theo tung layer

Khong nen ket luan `tiny-object improvement` neu `Recall_tiny` chua vuot baseline dung.

## Useful Helper Commands

Collect current V2-oriented table, cho phep file thieu:

```bash
python scripts/21_collect_v2_tiny_aware_results.py
```

Cap nhat bang model selection tong quat:

```bash
python scripts/18_generate_model_selection_table.py \
  --runs baseline_yolov8s tradaug_yolov8s_seed0 lsal_only_mse_seed0 srw_only_p3_seed0 \
  srw_lsal_p3_mse_seed0 srw_lsal_p3_warmup_decay_seed0 srw_lsal_energy_seed0 \
  srw_lsal_energy_bg_seed0 srw_lsal_energy_bg_sizeaware_seed0 \
  --baseline-run baseline_yolov8s
```

## Expected Outputs

Moi run hop le nen co:

- `experiments/skyfusion/<run>/config.yaml`
- `experiments/skyfusion/<run>/results.csv`
- `experiments/skyfusion/<run>/weights/best.pt`

Sau eval:

- `experiments/skyfusion/<run>/detection_eval/metrics.json`
- `experiments/skyfusion/<run>/xai_eval/xai_metrics.json`

## Narrative Guardrail

Trong giai doan nay, wording uu tien nen la:

- `saliency-guided SRW + L_sal`
- `saliency-guided reweighting and alignment framework`
- `tiny-aware P2/P2+P3 experiment family`

Khong nen viet nhu the `G-CAME` da la production implementation cho cac result hien co.
