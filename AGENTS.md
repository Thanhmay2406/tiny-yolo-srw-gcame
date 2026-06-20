# AGENTS

## Scope

This repository is a Kaggle-first research codebase for G-CAME-guided SRW YOLO on the Kaggle dataset `thanhmay2406/dataset-for-research`.

## Coding Rules

- Do not write to `/kaggle/input`.
- Do not hard-code local machine paths.
- Every runnable script must use `argparse`.
- Every experiment must save its config and metrics.
- Baseline training must not import `SRW` or `L_sal`.
- If dataset conversion is needed, write only to `/kaggle/working/data/SkyFusion_yolo_reconverted`.
- Do not assume the YOLO dataset already exists; inspect first and only reconvert when the YOLO copy is missing or invalid.
- Keep Kaggle outputs under `experiments/skyfusion/<run_name>/` or `paper/`.
- Do not modify Ultralytics source inside `site-packages`.

## Dataset Notes

- The Kaggle dataset currently contains both:
  - the original COCO-style SkyFusion dataset
  - a YOLO-formatted SkyFusion dataset
- Prefer validating the existing YOLO copy before running any reconversion step.
