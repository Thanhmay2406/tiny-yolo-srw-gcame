# tiny-yolo-saliency-align

Kaggle-first research repository for G-CAME-guided SRW YOLO on the dataset `thanhmay2406/dataset-for-research`.

## Phase 0 Status

This phase only prepares the repository for Kaggle execution.

- Training is not implemented in this phase.
- No SRW integration is implemented yet.
- No `L_sal` training path is implemented yet.
- The existing dataset must be inspected first; reconversion to YOLO is only needed when the YOLO copy is missing or invalid.

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

## Phase 0 Validation Commands

Run these commands on Kaggle to validate the bootstrap:

```bash
cd /kaggle/working/tiny-yolo-srw-gcame
bash kaggle/00_bootstrap_kaggle.sh
python scripts/00_env_check.py --experiments-dir experiments
```
