# V2 Tiny-Aware Ablation Results

This table is generated from available local artifacts only. Missing files are marked as `MISSING`.

| Group | Run | Model | Target layers | Loss | Size-aware | mAP50 | mAP50-95 | Precision | Recall | Recall tiny | Recall small | XAI pointing game | Energy in box | BG energy ratio | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P3 baseline | baseline_yolov8s | yolov8s.pt | default | none | no | 0.6567 | 0.3586 | 0.7094 | 0.6669 | 0.7574 | 0.9403 | MISSING | MISSING | MISSING | Existing seed0 baseline |
| Traditional aug | tradaug_yolov8s_seed0 | yolov8s.pt | default | none | no | 0.6683 | 0.3605 | 0.7417 | 0.6480 | 0.7488 | 0.9403 | MISSING | MISSING | MISSING | Existing seed0 augmentation baseline |
| P3 best detection | srw_lsal_energy_bg_seed0 | yolov8s.pt | P3 | energy_bg | False | 0.6739 | 0.3662 | 0.7299 | 0.6705 | 0.7479 | 0.9375 | 0.0111 | 0.0172 | 0.9828 | Existing best mAP candidate |
| P2 baseline | baseline_yolov8s_p2_seed0 | MISSING | default | none | no | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | Architecture effect |
| V2 main | srw_lsal_p2p3_mse_sizeaware_seed0 | MISSING | default | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | Main tiny-aware candidate |
| V2 layer ablation | srw_lsal_p2_mse_sizeaware_seed0 | MISSING | default | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | P2-only effect |
| V2 size ablation | srw_lsal_p2p3_mse_no_sizeaware_seed0 | MISSING | default | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | Size-aware effect |
| V2 optional | srw_lsal_p2p3_energy_bg_sizeaware_seed0 | MISSING | default | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | Optional after main V2 |
