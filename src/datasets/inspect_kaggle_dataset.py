from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ANNOTATION_EXTENSIONS = {".json", ".txt", ".yaml", ".yml", ".csv"}


def _depth_from_root(path: Path, root: Path) -> int:
    if path == root:
        return 0
    return len(path.relative_to(root).parts)


def walk_limited(root: Path, max_depth: int) -> List[Path]:
    root = root.resolve()
    paths: List[Path] = []

    def _walk(current: Path) -> None:
        paths.append(current)
        if _depth_from_root(current, root) >= max_depth or not current.is_dir():
            return

        for child in sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            _walk(child)

    _walk(root)
    return paths


def render_tree(root: Path, max_depth: int) -> str:
    root = root.resolve()
    lines = [root.name]

    def _walk(current: Path, prefix: str) -> None:
        if _depth_from_root(current, root) >= max_depth or not current.is_dir():
            return

        children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        for index, child in enumerate(children):
            connector = "`-- " if index == len(children) - 1 else "|-- "
            lines.append(f"{prefix}{connector}{child.name}")
            extension = "    " if index == len(children) - 1 else "|   "
            _walk(child, prefix + extension)

    _walk(root, "")
    return "\n".join(lines) + "\n"


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def summarize_files(root: Path) -> Dict[str, Any]:
    extension_counts: Counter[str] = Counter()
    total_files = 0

    for file_path in _iter_files(root):
        total_files += 1
        extension = file_path.suffix.lower() or "<no_ext>"
        extension_counts[extension] += 1

    return {
        "root": str(root),
        "total_files": total_files,
        "extensions": dict(sorted(extension_counts.items())),
    }


def find_candidate_files(root: Path, extensions: set[str]) -> List[str]:
    matches = []
    for file_path in _iter_files(root):
        if file_path.suffix.lower() in extensions:
            matches.append(str(file_path))
    return sorted(matches)


def _is_coco_json(json_path: Path) -> bool:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        return False

    required_keys = {"images", "annotations", "categories"}
    return required_keys.issubset(payload.keys())


def detect_dataset_format(root: Path) -> Dict[str, Any]:
    json_candidates = [Path(path) for path in find_candidate_files(root, {".json"})]
    coco_files = [str(path) for path in json_candidates if _is_coco_json(path)]

    images_dir = root / "images"
    labels_dir = root / "labels"
    label_txt_files = list(labels_dir.rglob("*.txt")) if labels_dir.is_dir() else []
    has_yolo_layout = images_dir.is_dir() and labels_dir.is_dir() and len(label_txt_files) > 0

    if coco_files and has_yolo_layout:
        format_name = "coco_and_yolo"
    elif coco_files:
        format_name = "coco"
    elif has_yolo_layout:
        format_name = "yolo"
    else:
        format_name = "unknown"

    return {
        "format": format_name,
        "coco_json_files": coco_files,
        "yolo_images_dir": str(images_dir) if images_dir.is_dir() else None,
        "yolo_labels_dir": str(labels_dir) if labels_dir.is_dir() else None,
        "yolo_label_file_count": len(label_txt_files),
    }


def inspect_root(root: Path, max_depth: int) -> Dict[str, Any]:
    return {
        "root": str(root),
        "tree": render_tree(root, max_depth=max_depth),
        "summary": summarize_files(root),
        "candidate_images": find_candidate_files(root, IMAGE_EXTENSIONS),
        "candidate_annotations": find_candidate_files(root, ANNOTATION_EXTENSIONS),
        "format_detection": detect_dataset_format(root),
    }


def list_available_input_folders(input_root: Path) -> List[str]:
    if not input_root.exists():
        return []

    candidates = []
    for child in sorted(input_root.iterdir(), key=lambda item: item.name.lower()):
        candidates.append(str(child))
        if child.is_dir():
            for grandchild in sorted(child.iterdir(), key=lambda item: item.name.lower()):
                candidates.append(str(grandchild))
    return candidates


def resolve_dataset_root(input_root: Path, dataset_name: str | None = None) -> Path:
    if dataset_name:
        dataset_root = input_root / dataset_name
    else:
        dataset_root = input_root

    if dataset_root.exists():
        return dataset_root.resolve()

    available = list_available_input_folders(Path("/kaggle/input"))
    available_text = "\n".join(f"- {item}" for item in available) if available else "- <no folders found>"
    raise FileNotFoundError(
        "Dataset root was not found.\n"
        f"Requested path: {dataset_root}\n"
        "Available folders under /kaggle/input:\n"
        f"{available_text}"
    )


def build_recommendation(report: Dict[str, Any]) -> str:
    inspected = report["inspected_roots"]
    lines = [
        f"Dataset root: {report['dataset_root']}",
        "",
    ]

    for item in inspected:
        detection = item["format_detection"]
        lines.append(f"[{Path(item['root']).name}]")
        lines.append(f"format: {detection['format']}")
        lines.append(f"candidate images: {len(item['candidate_images'])}")
        lines.append(f"candidate annotations: {len(item['candidate_annotations'])}")
        if detection["coco_json_files"]:
            lines.append(f"coco json files: {len(detection['coco_json_files'])}")
        if detection["yolo_label_file_count"]:
            lines.append(f"yolo label files: {detection['yolo_label_file_count']}")
        lines.append("")

    skyfusion = next((item for item in inspected if Path(item["root"]).name == "SkyFusion"), None)
    skyfusion_yolo = next((item for item in inspected if Path(item["root"]).name == "SkyFusion_yolo"), None)

    if skyfusion and skyfusion["format_detection"]["format"] in {"coco", "coco_and_yolo"}:
        lines.append("COCO source detected under SkyFusion.")
    else:
        lines.append("COCO source was not clearly detected under SkyFusion.")

    if skyfusion_yolo and skyfusion_yolo["format_detection"]["format"] in {"yolo", "coco_and_yolo"}:
        lines.append("YOLO dataset detected under SkyFusion_yolo.")
        lines.append("Recommendation: validate the existing YOLO dataset before any reconversion.")
    else:
        lines.append("YOLO dataset was not clearly detected under SkyFusion_yolo.")
        lines.append("Recommendation: inspect the YOLO export and reconvert only if validation fails.")

    lines.append("Do not write anything into /kaggle/input.")
    return "\n".join(lines) + "\n"


def inspect_kaggle_dataset(input_root: Path, dataset_name: str | None, max_depth: int) -> Dict[str, Any]:
    dataset_root = resolve_dataset_root(input_root=input_root, dataset_name=dataset_name)
    inspected_roots: List[Dict[str, Any]] = [inspect_root(dataset_root, max_depth=max_depth)]

    for child_name in ("SkyFusion", "SkyFusion_yolo"):
        child_root = dataset_root / child_name
        if child_root.exists():
            inspected_roots.append(inspect_root(child_root, max_depth=max_depth))

    report = {
        "dataset_root": str(dataset_root),
        "max_depth": max_depth,
        "inspected_roots": inspected_roots,
    }
    report["recommendation"] = build_recommendation(report)
    return report
