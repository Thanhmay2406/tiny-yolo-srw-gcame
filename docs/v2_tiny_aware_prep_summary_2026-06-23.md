# V2 Tiny-Aware Prep Summary

## What changed

- Updated:
  - `docs/roadmap.md`
  - `docs/roadmap_vietnamese.md`
  - `docs/runbooks/run_v2_tiny_aware_experiments.md`
- Created:
  - `docs/v2_tiny_aware_config_sanity_2026-06-23.md`
  - `paper/tables/v2_tiny_aware_ablation_template.md`
  - `scripts/21_collect_v2_tiny_aware_results.py`
  - `docs/v2_tiny_aware_prep_summary_2026-06-23.md`

## What was intentionally not changed

- Khong chay train dai.
- Khong implement `G-CAME`.
- Khong them loss moi.
- Khong commit git.
- Khong sua training logic chinh cua `SRW + L_sal` vi chua co bug can fix ngay cho giai doan prep nay.

## Sanity checks run

- Da luu snapshot git:
  - `git_status_before_v2_prep.txt`
  - `git_branch_before_v2_prep.txt`
  - `git_log_before_v2_prep.txt`
- Da kiem tra cau truc 6 config V2 va ghi report:
  - `docs/v2_tiny_aware_config_sanity_2026-06-23.md`
- Da chay:

```bash
python -m pytest tests/test_layer_resolver.py tests/test_multiscale_srw.py tests/test_v2_tiny_aware_configs.py -q
```

Ket qua:

- Khong collect duoc test vi environment hien tai thieu `torch`
- Loi xuat hien ngay luc import module, khong phai fail logic cua thay doi moi

- Da chay:

```bash
python scripts/21_collect_v2_tiny_aware_results.py
```

Ket qua:

- PASS
- Da tao:
  - `paper/tables/v2_tiny_aware_ablation_results.md`
  - `paper/tables/v2_tiny_aware_ablation_results.csv`

## Current recommended next manual commands

```bash
python scripts/05_train_baseline.py \
  --config configs/skyfusion_v2_tiny_aware/baseline_yolov8s_p2_seed0.yaml \
  --run-name baseline_yolov8s_p2_seed0

python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_mse_sizeaware_seed0

python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2_mse_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2_mse_sizeaware_seed0

python scripts/12_train_srw_lsal.py \
  --config configs/skyfusion_v2_tiny_aware/srw_lsal_p2p3_mse_no_sizeaware_seed0.yaml \
  --run-name srw_lsal_p2p3_mse_no_sizeaware_seed0
```

## Risks remaining

- V2 chua co artifact train that.
- Multi-seed chua co cho family V2.
- Teacher-safe khong phai uu tien trong vong chay V2 dau tien.
- XAI metrics van can dien giai can trong.
- Docs cu van co the con mot so cho outdated ngoai cac warning block da them.
