# V2 Tiny-Aware Config Sanity Report — 2026-06-23

## Summary

Tat ca 6 config V2 tiny-aware duoc yeu cau deu ton tai, doc duoc, va PASS o muc sanity cau truc co ban.

Kiem tra da thuc hien:

- file ton tai va doc duoc
- co `run_name`
- co `model`
- co `data`
- co `imgsz`
- co `batch`
- co `epochs`
- co `seed`
- co `output_root`
- voi config `SRW + L_sal`: co `target_layers`, `loss_type`, `size_aware`

Khong thay loi cau truc ro rang can sua ngay trong cac config nay.

## Configs

### PASS — `configs/skyfusion_v2_tiny_aware/baseline_yolov8s_p2_seed0.yaml`

- run_name: `baseline_yolov8s_p2_seed0`
- model: `yolov8s-p2.yaml`
- data: `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- imgsz: `640`
- batch: `16`
- epochs: `100`
- seed: `0`
- output_root: `experiments/skyfusion`
- issue: none found in sanity check
- train command:

```bash
python scripts/05_train_baseline.py \
  --config configs/skyfusion_v2_tiny_aware/baseline_yolov8s_p2_seed0.yaml \
  --run-name baseline_yolov8s_p2_seed0
```

### PASS — `configs/skyfusion_v2_tiny_aware/srw_lsal_p2_mse_sizeaware_seed0.yaml`

- run_name: `srw_lsal_p2_mse_sizeaware_seed0`
- model: `yolov8s-p2.yaml`
- data: `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- imgsz: `640`
- batch: `16`
- epochs: `100`
- seed: `0`
- output_root: `experiments/skyfusion`
- target_layers: `P2`
- loss_type: `mse`
- size_aware: `true`
- issue: none found in sanity check
- train command:

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2_mse_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2_mse_sizeaware_seed0
```

### PASS — `configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_sizeaware_seed0.yaml`

- run_name: `srw_lsal_p2p3_mse_sizeaware_seed0`
- model: `yolov8s-p2.yaml`
- data: `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- imgsz: `640`
- batch: `16`
- epochs: `100`
- seed: `0`
- output_root: `experiments/skyfusion`
- target_layers: `P2`, `P3`
- loss_type: `mse`
- size_aware: `true`
- issue: none found in sanity check
- train command:

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_mse_sizeaware_seed0
```

### PASS — `configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_no_sizeaware_seed0.yaml`

- run_name: `srw_lsal_p2p3_mse_no_sizeaware_seed0`
- model: `yolov8s-p2.yaml`
- data: `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- imgsz: `640`
- batch: `16`
- epochs: `100`
- seed: `0`
- output_root: `experiments/skyfusion`
- target_layers: `P2`, `P3`
- loss_type: `mse`
- size_aware: `false`
- issue: none found in sanity check
- train command:

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_no_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_mse_no_sizeaware_seed0
```

### PASS — `configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_energy_bg_sizeaware_seed0.yaml`

- run_name: `srw_lsal_p2p3_energy_bg_sizeaware_seed0`
- model: `yolov8s-p2.yaml`
- data: `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- imgsz: `640`
- batch: `16`
- epochs: `100`
- seed: `0`
- output_root: `experiments/skyfusion`
- target_layers: `P2`, `P3`
- loss_type: `energy_bg`
- size_aware: `true`
- issue: none found in sanity check
- train command:

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_energy_bg_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_energy_bg_sizeaware_seed0
```

### PASS — `configs/skyfusion_v2_tiny_aware/srw_lsal_p3_mse_sizeaware_seed0.yaml`

- run_name: `srw_lsal_p3_mse_sizeaware_seed0`
- model: `yolov8s-p2.yaml`
- data: `/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml`
- imgsz: `640`
- batch: `16`
- epochs: `100`
- seed: `0`
- output_root: `experiments/skyfusion`
- target_layers: `P3`
- loss_type: `mse`
- size_aware: `true`
- issue: none found in sanity check
- train command:

```bash
python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p3_mse_sizeaware_seed0.yaml \
  --run-name srw_lsal_p3_mse_sizeaware_seed0
```

## Notes

- Tat ca config V2 deu dung `yolov8s-p2.yaml`, phu hop voi muc tieu kiem chung `P2/P2+P3`.
- Tat ca config V2 dang tro toi Kaggle path tuyet doi cho dataset. Neu chay local, can override `SKYFUSION_DATA` hoac `--data`.
- Khong co config nao trong nhom nay uu tien `teacher-safe`, phu hop voi thu tu uu tien V2 hien tai.
