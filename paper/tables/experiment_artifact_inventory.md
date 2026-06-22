# Experiment Artifact Inventory

| Run | has_best_pt | has_results_csv | has_detection_eval | has_xai_eval | has_convergence_eval | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline_yolov8s` | True | True | True | False | True |  |
| `dataset_inspect` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `dataset_inspect_local` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `dataset_inspect_local_smoke` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `debug_gt_saliency_local` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `debug_saliency_head_local` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `debug_srw_shapes_local` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `debug_srw_yolo_shapes_local` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `debug_teacher_saliency_local` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `lsal_only_mse_seed0` | True | True | False | False | False | priority detection eval missing; priority xai eval missing; convergence eval missing |
| `smoke_config_support_local` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_lsal_only_local` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_srw_lsal_energy_bg_local` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_srw_lsal_local` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_srw_lsal_multiscale_local` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_srw_lsal_multiscale_local_rerun` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_srw_lsal_schedule_local` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_srw_only_local` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_teacher_disable_ok` | False | True | False | False | False | missing best.pt; auxiliary/local debug run |
| `smoke_teacher_disable_ok_rerun` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_teacher_effective_config_check` | True | True | False | False | False | auxiliary/local debug run |
| `smoke_teacher_guard_error` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |
| `srw_lsal_energy_bg_seed0` | True | True | True | True | True |  |
| `srw_lsal_energy_bg_sizeaware_seed0` | True | True | False | False | False | priority detection eval missing; priority xai eval missing; convergence eval missing |
| `srw_lsal_energy_seed0` | True | True | False | False | False | priority detection eval missing; priority xai eval missing; convergence eval missing |
| `srw_lsal_p3_mse_seed0` | True | True | False | False | False | priority detection eval missing; priority xai eval missing; convergence eval missing |
| `srw_lsal_p3_warmup_decay_seed0` | True | True | True | True | True |  |
| `srw_only_p3_seed0` | True | True | True | False | True |  |
| `tradaug_yolov8s_seed0` | True | True | False | False | False | priority detection eval missing; convergence eval missing |
| `xai_teacher_local_smoke` | False | False | False | False | False | missing best.pt; missing results.csv; auxiliary/local debug run |

Notes:

- `priority detection eval missing`: run chinh co checkpoint nhưng chua co `detection_eval/metrics.json`.
- `priority xai eval missing`: run saliency-guided chinh co checkpoint nhưng chua co `xai_eval/*.json`.
- `auxiliary/local debug run`: khong nam trong nhom evidence chinh cho paper.
