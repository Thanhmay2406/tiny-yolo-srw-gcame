#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.inspect_kaggle_dataset import inspect_kaggle_dataset
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Kaggle SkyFusion dataset layout.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input/datasets/thanhmay2406/dataset-for-research"),
        help="Dataset root mounted from Kaggle input.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset subdirectory name under --input-root.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth to render in the text report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/skyfusion/dataset_inspect"),
        help="Writable output directory for inspection artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    try:
        report = inspect_kaggle_dataset(
            input_root=args.input_root,
            dataset_name=args.dataset_name,
            max_depth=args.max_depth,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    tree_parts = []
    for item in report["inspected_roots"]:
        tree_parts.append(f"# {item['root']}")
        tree_parts.append(item["tree"].rstrip())
        tree_parts.append("")
    (output_dir / "tree.txt").write_text("\n".join(tree_parts).rstrip() + "\n", encoding="utf-8")

    file_summary = {
        item["root"]: {
            "summary": item["summary"],
            "format_detection": item["format_detection"],
        }
        for item in report["inspected_roots"]
    }
    save_json(output_dir / "file_summary.json", file_summary)

    candidate_images = {item["root"]: item["candidate_images"] for item in report["inspected_roots"]}
    save_json(output_dir / "candidate_images.json", candidate_images)

    candidate_annotations = {
        item["root"]: item["candidate_annotations"] for item in report["inspected_roots"]
    }
    save_json(output_dir / "candidate_annotations.json", candidate_annotations)

    (output_dir / "recommendation.txt").write_text(report["recommendation"], encoding="utf-8")

    print(f"Inspection complete. Artifacts saved to: {output_dir.resolve()}")
    print(f"Dataset root: {report['dataset_root']}")
    for item in report["inspected_roots"]:
        detection = item["format_detection"]
        print(f"- {item['root']}: format={detection['format']}, files={item['summary']['total_files']}")


if __name__ == "__main__":
    main()
