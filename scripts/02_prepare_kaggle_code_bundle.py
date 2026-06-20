#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, save_json


DEFAULT_CODE_DATASET_ID = "thanhmay2406/tiny-yolo-srw-gcame-code"
DEFAULT_CODE_DATASET_TITLE = "tiny-yolo-srw-gcame-code"

FILES_TO_COPY = [
    "README.md",
    "AGENTS.md",
    "requirements.txt",
]

DIRS_TO_COPY = [
    "src",
    "scripts",
    "configs",
    "kaggle",
    "docs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a lightweight code bundle for uploading this repo to Kaggle as a code dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/skyfusion_kaggle_code"),
        help="Destination directory for the Kaggle code bundle.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_CODE_DATASET_ID,
        help="Kaggle dataset id for the code bundle, e.g. username/slug.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=DEFAULT_CODE_DATASET_TITLE,
        help="Kaggle dataset title for the code bundle.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before preparing the bundle.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)


def build_metadata(dataset_id: str, title: str) -> dict:
    return {
        "title": title,
        "id": dataset_id,
        "licenses": [
            {
                "name": "CC0-1.0",
            }
        ],
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    ensure_dir(output_dir)

    for relative_path in FILES_TO_COPY:
        src = PROJECT_ROOT / relative_path
        if src.is_file():
            copy_file(src, output_dir / relative_path)

    for relative_dir in DIRS_TO_COPY:
        src_dir = PROJECT_ROOT / relative_dir
        if src_dir.is_dir():
            copy_tree(src_dir, output_dir / relative_dir)

    metadata = build_metadata(dataset_id=args.dataset_id, title=args.title)
    save_json(output_dir / "dataset-metadata.json", metadata)

    print(f"Kaggle code bundle prepared at: {output_dir}")
    print(f"Dataset id: {args.dataset_id}")


if __name__ == "__main__":
    main()
