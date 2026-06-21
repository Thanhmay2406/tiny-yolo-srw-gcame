# tiny-yolo-saliency-align

Kaggle-first research repository for G-CAME-guided SRW YOLO on the dataset `thanhmay2406/dataset-for-research`.

## Current Status

The repository now covers the baseline and several post-baseline debug phases from the roadmap.

- `scripts/05_train_baseline.py`: clean YOLO baseline training.
- `scripts/06_train_tradaug.py`: traditional augmentation baseline.
- `scripts/07_debug_gt_saliency.py`: Gaussian GT saliency mask debug from YOLO boxes.
- `scripts/08_debug_xai_saliency.py`: saliency provider debug on YOLO feature maps.
- `scripts/08b_precompute_xai_teacher.py`: offline teacher saliency precompute.
- `scripts/09_debug_srw_shapes.py`: standalone and YOLO-hook SRW shape debug.
- `scripts/10_train_lsal_only.py`: `L_sal`-only training.
- `scripts/11_train_srw_only.py`: `SRW`-only training.
- `scripts/12_train_srw_lsal.py`: joint `SRW + L_sal` training with optional offline teacher loss.
- Phase 13 lambda scheduling is implemented for `scripts/12_train_srw_lsal.py`.

## Current Repository Structure

```text
.
|-- AGENTS.md
|-- README.md
|-- requirements.txt
|-- scripts/
|   |-- 00_env_check.py
|-- src/
|   |-- utils/
|       |-- device.py
|       |-- io.py
|       |-- logging.py
|       |-- paths.py
|       |-- seed.py
|-- kaggle/
|   |-- 00_bootstrap_kaggle.sh
|-- data/
|   |-- SkyFusion/
|   |-- SkyFusion_yolo/
|   |-- dataset-metadata.json
|-- experiments/
|-- paper/
```

## Dataset Notes

The Kaggle dataset currently includes both:

- the original COCO-style SkyFusion dataset in `data/SkyFusion`
- a YOLO-formatted copy in `data/SkyFusion_yolo`

Phase 0 assumes neither is authoritative until inspected.

Rules:

- never write into `/kaggle/input`
- if reconversion is required on Kaggle, write only to `/kaggle/working/data/SkyFusion_yolo_reconverted`
- keep experiment outputs under `experiments/skyfusion/<run_name>/`
- keep paper artifacts under `paper/`

## Minimal Dependencies

Install the project requirements with:

```bash
pip install -r requirements.txt
```

## Kaggle Quickstart

Inside a Kaggle Notebook, after cloning or copying this repo into `/kaggle/working`, run:

```bash
cd /kaggle/working/tiny-yolo-srw-gcame
bash kaggle/00_bootstrap_kaggle.sh
```

That script will:

- install the minimal dependencies from `requirements.txt`
- run `scripts/00_env_check.py`

## Using Your Kaggle API

If you want to push the project code itself to Kaggle using your current API token, use the helper below on your local machine:

```bash
cd /home/thanhmay/workspace/tiny-yolo-srw-gcame
bash kaggle/01_push_code_dataset.sh
```

This does three things:

- prepares a lightweight code bundle under `/tmp/skyfusion_kaggle_code`
- excludes local-only content such as `.git`, `.venv`, and the dataset files
- creates or versions a Kaggle code dataset with your current Kaggle API credentials

Default code dataset id:

```text
thanhmay2406/tiny-yolo-srw-gcame-code
```

You can override bundle path, dataset id, title, and version message:

```bash
bash kaggle/01_push_code_dataset.sh \
  /tmp/skyfusion_kaggle_code \
  thanhmay2406/tiny-yolo-srw-gcame-code \
  tiny-yolo-srw-gcame-code \
  "phase0 and dataset inspect setup"
```

On Kaggle Notebook, attach:

- dataset: `thanhmay2406/dataset-for-research`
- code dataset: `thanhmay2406/tiny-yolo-srw-gcame-code`

Then run:

```bash
cp -r /kaggle/input/tiny-yolo-srw-gcame-code /kaggle/working/tiny-yolo-srw-gcame
cd /kaggle/working/tiny-yolo-srw-gcame
bash kaggle/00_bootstrap_kaggle.sh
python scripts/01_inspect_kaggle_dataset.py \
  --input-root /kaggle/input/datasets/thanhmay2406/dataset-for-research \
  --max-depth 4 \
  --output-dir experiments/skyfusion/dataset_inspect
```

## Environment Check

You can also run the environment check directly:

```bash
python scripts/00_env_check.py --experiments-dir experiments
```

It prints:

- Python version
- torch version
- CUDA availability
- GPU name if available
- current working directory
- whether `experiments/` is writable

## Kaggle Path Policy

- `/kaggle/input` is read-only and must not be modified
- generated files belong in `/kaggle/working`
- do not hard-code local absolute paths
- every runnable script must use `argparse`
- every experiment must save config and metrics
- baseline training, when added later, must not import SRW or `L_sal`

## Post-Baseline Commands

Traditional augmentation baseline:

```bash
python scripts/06_train_tradaug.py \
  --data "$SKYFUSION_DATA" \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name tradaug_yolov8s_seed0
```

GT saliency debug:

```bash
python scripts/07_debug_gt_saliency.py \
  --data "$SKYFUSION_DATA" \
  --split train \
  --num-samples 16 \
  --output-dir experiments/skyfusion/debug_gt_saliency
```

Saliency provider debug:

```bash
python scripts/08_debug_xai_saliency.py \
  --data "$SKYFUSION_DATA" \
  --split valid \
  --target-layers P3 \
  --saliency-provider saliency_head \
  --weights yolov8s.pt \
  --output-dir experiments/skyfusion/debug_saliency_head
```

Offline teacher precompute:

```bash
python scripts/08b_precompute_xai_teacher.py \
  --data "$SKYFUSION_DATA" \
  --weights experiments/skyfusion/baseline_yolov8s/weights/best.pt \
  --split train \
  --target-layers P3 \
  --xai-method gradcam_like \
  --output-dir experiments/skyfusion/xai_teacher/baseline_p3_train
```

SRW debug:

```bash
python scripts/09_debug_srw_shapes.py --from-yolo --model yolov8s.yaml --target-layers P3
```

Joint `SRW + L_sal`:

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_p3_mse_seed0 \
  --target-layers P3 \
  --loss-type mse \
  --lambda-sal 0.1 \
  --alpha-init 0.1
```

Joint `SRW + L_sal` with lambda scheduling:

```bash
python scripts/12_train_srw_lsal.py \
  --data "$SKYFUSION_DATA" \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --seed 0 \
  --run-name srw_lsal_p3_warmup_decay_seed0 \
  --target-layers P3 \
  --loss-type mse \
  --lambda-schedule warmup_cosine_decay \
  --lambda-max 0.2 \
  --lambda-min 0.01 \
  --warmup-epochs 5 \
  --alpha-init 0.1
```

## Phase 0 Validation Commands

Run these commands on Kaggle to validate the bootstrap:

```bash
cd /kaggle/working/tiny-yolo-srw-gcame
bash kaggle/00_bootstrap_kaggle.sh
python scripts/00_env_check.py --experiments-dir experiments
```
