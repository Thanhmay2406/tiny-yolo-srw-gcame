# Balanced Model Selection Summary

This table is a lightweight comparison aid, not a proof of absolute model superiority.

| Run | mAP50 | mAP50-95 | Delta vs baseline | Recall_tiny | Recall_small | Pointing Game | Energy-in-Box | BER | Recommended use | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `baseline_yolov8s` | 0.6567 | 0.3586 | 0.0000 | 0.7574 | 0.9403 | N/A | N/A | N/A | not enough XAI data | best epoch=94 |
| `srw_only_p3_seed0` | 0.6618 | 0.3618 | 0.0032 | 0.7547 | 0.9431 | N/A | N/A | N/A | not enough XAI data | tiny recall below baseline; best epoch=99 |
| `srw_lsal_p3_warmup_decay_seed0` | 0.6674 | 0.3639 | 0.0052 | 0.7493 | 0.9331 | 0.0290 | 0.0190 | 0.9810 | better XAI localization candidate | tiny recall below baseline; best epoch=100 |
| `srw_lsal_energy_bg_seed0` | 0.6739 | 0.3662 | 0.0076 | 0.7479 | 0.9375 | 0.0111 | 0.0172 | 0.9828 | best detection candidate | tiny recall below baseline; best epoch=99 |

## Notes

- The current table only uses runs that already have `detection_eval/metrics.json` available.
- Any run not yet included here should first run `scripts/13_eval_detection.py` to generate `detection_eval` outputs before being compared fairly.
- `recommended_use` is intentionally conservative and does not declare a universally best model.
- Detection ranking and XAI ranking may disagree.
- Multi-seed confirmation is still required before making strong claims.
