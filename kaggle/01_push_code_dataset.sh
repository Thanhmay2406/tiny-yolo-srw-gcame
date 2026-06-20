#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUNDLE_DIR="${1:-/tmp/skyfusion_kaggle_code}"
DATASET_ID="${2:-thanhmay2406/tiny-yolo-srw-gcame-code}"
TITLE="${3:-tiny-yolo-srw-gcame-code}"
MESSAGE="${4:-update kaggle code bundle}"

if [ -x "$REPO_DIR/.venv/bin/kaggle" ]; then
  KAGGLE_BIN="$REPO_DIR/.venv/bin/kaggle"
else
  KAGGLE_BIN="kaggle"
fi

python "$REPO_DIR/scripts/02_prepare_kaggle_code_bundle.py" \
  --output-dir "$BUNDLE_DIR" \
  --dataset-id "$DATASET_ID" \
  --title "$TITLE" \
  --clean

if "$KAGGLE_BIN" datasets status "$DATASET_ID" >/dev/null 2>&1; then
  echo "Code dataset exists: $DATASET_ID"
  "$KAGGLE_BIN" datasets version -p "$BUNDLE_DIR" -m "$MESSAGE" --dir-mode zip
else
  echo "Code dataset does not exist yet: $DATASET_ID"
  "$KAGGLE_BIN" datasets create -p "$BUNDLE_DIR" --dir-mode zip
fi
