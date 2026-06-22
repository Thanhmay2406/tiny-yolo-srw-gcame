# Kaggle Clone Eval Workflow

Runbook này dùng khi bạn muốn:

- đẩy phần code cần thiết lên GitHub
- mở Kaggle Notebook mới
- `git clone` repo trực tiếp trong `/kaggle/working`
- chạy detection/XAI/error-analysis bằng các script Python của repo
- copy artifact cần giữ ra ngoài repo
- xóa repo trước khi `Save Version` để output nhẹ hơn

## Nguyên tắc

- Không ghi vào `/kaggle/input`.
- Chỉ clone repo vào `/kaggle/working/tiny-yolo-srw-gcame`.
- Chỉ giữ lại artifact cuối cùng dưới `/kaggle/working/final_artifacts/`.
- Sau khi copy xong artifact, xóa repo clone để giảm dung lượng version output.

## Dataset source-of-truth

```text
/kaggle/input/datasets/thanhmay2406/dataset-for-research/SkyFusion_yolo/data.yaml
```

## Notebook template

Notebook template nằm tại:

```text
kaggle/02_eval_from_github_notebook.ipynb
```

Notebook này làm sẵn các bước:

1. clone repo từ GitHub
2. cài dependencies tối thiểu
3. chạy detection eval cho các run còn thiếu
4. chạy XAI eval cho các saliency-guided run còn thiếu
5. chạy error analysis cho 2 cặp chính
6. thử regenerate `paper/tables/model_selection_summary.*`
7. copy artifact cần giữ sang `/kaggle/working/final_artifacts`
8. xóa repo clone
9. in ra tree output cuối để bạn `Save Version`

## Chuẩn bị local trước khi lên Kaggle

1. commit các file cần thiết
2. push lên branch bạn muốn Kaggle clone
3. trên Kaggle, bật Internet
4. dán hoặc upload notebook template
5. sửa `REPO_URL` và `REPO_BRANCH` nếu cần

## Artifact nên giữ

- `experiments/skyfusion/<run>/detection_eval/`
- `experiments/skyfusion/<run>/xai_eval/`
- `experiments/skyfusion/<run>/error_analysis_vs_baseline/`
- `paper/tables/model_selection_summary.md`
- `paper/tables/model_selection_summary.csv`
- `paper/tables/experiment_artifact_inventory.md`
- `paper/tables/experiment_artifact_inventory.csv`

## Sau khi chạy xong

Nếu notebook đã copy artifact sang `/kaggle/working/final_artifacts/` và đã xóa repo clone, bạn chỉ cần:

1. kiểm tra lại file trong `final_artifacts`
2. `Save Version`
3. tải artifact về hoặc import lại vào local repo sau

