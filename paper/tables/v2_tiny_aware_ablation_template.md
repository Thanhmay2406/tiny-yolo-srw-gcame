# V2 Tiny-Aware Ablation Template

| Group | Run | Model | Target layers | Loss | Size-aware | mAP50 | mAP50-95 | Precision | Recall | Recall tiny | Recall small | XAI pointing game | Energy in box | BG energy ratio | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P3 baseline | baseline_yolov8s | yolov8s.pt | default | none | no |  |  |  |  |  |  |  |  |  | Existing seed0 baseline |
| Traditional aug | tradaug_yolov8s_seed0 | yolov8s.pt | default | none | no |  |  |  |  |  |  |  |  |  | Existing seed0 augmentation baseline |
| P3 best detection | srw_lsal_energy_bg_seed0 | yolov8s.pt | P3 | energy_bg | no |  |  |  |  |  |  |  |  |  | Existing best mAP candidate |
| P2 baseline | baseline_yolov8s_p2_seed0 | yolov8s-p2.yaml | default/P2-enabled | none | no |  |  |  |  |  |  |  |  |  | Architecture effect |
| V2 main | srw_lsal_p2p3_mse_sizeaware_seed0 | yolov8s-p2.yaml | P2+P3 | mse | yes |  |  |  |  |  |  |  |  |  | Main tiny-aware candidate |
| V2 layer ablation | srw_lsal_p2_mse_sizeaware_seed0 | yolov8s-p2.yaml | P2 | mse | yes |  |  |  |  |  |  |  |  |  | P2-only effect |
| V2 size ablation | srw_lsal_p2p3_mse_no_sizeaware_seed0 | yolov8s-p2.yaml | P2+P3 | mse | no |  |  |  |  |  |  |  |  |  | Size-aware effect |
| V2 optional | srw_lsal_p2p3_energy_bg_sizeaware_seed0 | yolov8s-p2.yaml | P2+P3 | energy_bg | yes |  |  |  |  |  |  |  |  |  | Optional after main V2 |

## Interpretation Guide

- Do not claim tiny-object improvement unless Recall tiny improves against the correct baseline.
- Compare V2 methods primarily against `baseline_yolov8s_p2_seed0`, not only against old `baseline_yolov8s`.
- Compare P2 architecture against old baseline to separate architecture gain from method gain.
- If detection mAP improves but XAI metrics degrade, frame the result as a detection/explainability trade-off.
