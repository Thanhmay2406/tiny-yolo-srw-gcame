# Regenerate Model Selection Summary

Sau khi bổ sung thêm `detection_eval` và `xai_eval`, chạy lại bảng model selection:

```bash
python scripts/18_generate_model_selection_table.py \
  --runs \
    baseline_yolov8s \
    tradaug_yolov8s_seed0 \
    lsal_only_mse_seed0 \
    srw_only_p3_seed0 \
    srw_lsal_p3_mse_seed0 \
    srw_lsal_p3_warmup_decay_seed0 \
    srw_lsal_energy_seed0 \
    srw_lsal_energy_bg_seed0 \
    srw_lsal_energy_bg_sizeaware_seed0 \
  --baseline-run baseline_yolov8s
```

Output:

- `paper/tables/model_selection_summary.md`
- `paper/tables/model_selection_summary.csv`

Lưu ý:

- Script không tuyên bố model nào tốt nhất tuyệt đối.
- `recommended_use` chỉ là gợi ý thận trọng dựa trên evidence hiện có.
