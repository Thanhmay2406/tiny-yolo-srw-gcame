#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/00_env_check.py --experiments-dir "$REPO_DIR/experiments"
