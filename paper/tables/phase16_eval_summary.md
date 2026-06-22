# Phase 16 Evaluation Summary

Primary comparison runs:

| Run | Precision | Recall | F1 | mAP50 | mAP50-95 | Delta vs baseline | Recall_tiny | Recall_small | Recall_medium_large | Pointing Game P3 | Energy-in-Box P3 | BER P3 | Best epoch | Best train mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_yolov8s` | 0.7094 | 0.6669 | 0.6875 | 0.6567 | 0.3586 | 0.0000 | 0.7574 | 0.9403 | 0.9503 | N/A | N/A | N/A | 94 | 0.3588 |
| `srw_only_p3_seed0` | 0.7145 | 0.6655 | 0.6892 | 0.6618 | 0.3618 | +0.0032 | 0.7547 | 0.9431 | 0.9503 | N/A | N/A | N/A | 99 | 0.3632 |
| `srw_lsal_p3_warmup_decay_seed0` | 0.7256 | 0.6695 | 0.6964 | 0.6674 | 0.3639 | +0.0052 | 0.7493 | 0.9331 | 0.9503 | 0.0290 | 0.0190 | 0.9810 | 100 | 0.3631 |
| `srw_lsal_energy_bg_seed0` | 0.7299 | 0.6705 | 0.6989 | 0.6739 | 0.3662 | +0.0076 | 0.7479 | 0.9375 | 0.9503 | 0.0111 | 0.0172 | 0.9828 | 99 | 0.3676 |

## Current reading

- Best detection run: `srw_lsal_energy_bg_seed0`
- Best XAI run among evaluated saliency models: `srw_lsal_p3_warmup_decay_seed0`
- Tiny-object recall did not improve over baseline in the current best detection run.
- Medium/large recall is effectively unchanged across the four runs.
- `srw_lsal_energy_bg_seed0` peaked at epoch 99, so the run looks stable and not obviously undertrained.

## Source paths

- Detection eval: `experiments/skyfusion/<run>/detection_eval/metrics.json`
- XAI eval: `experiments/skyfusion/<run>/xai_eval/xai_metrics.json`
- Convergence eval: `experiments/skyfusion/<run>/convergence_eval/convergence_metrics.json`
