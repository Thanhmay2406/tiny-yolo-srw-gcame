from __future__ import annotations

from pathlib import Path

from src.utils.io import load_yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_split_sources(data_yaml: str | Path, split: str) -> list[Path]:
    yaml_path = Path(data_yaml).resolve()
    payload = load_yaml(yaml_path)
    raw_value = payload.get(split)
    if raw_value is None and split == "valid":
        raw_value = payload.get("val")
    if raw_value is None and split == "val":
        raw_value = payload.get("valid")
    if raw_value is None:
        raise ValueError(f"Split '{split}' is not defined in dataset yaml: {yaml_path}")

    candidates = raw_value if isinstance(raw_value, list) else [raw_value]
    resolved: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = (yaml_path.parent / path).resolve()
        resolved.append(path)
    return resolved


def collect_image_files(sources: list[Path], limit: int | None = None) -> list[Path]:
    images: list[Path] = []
    for source in sources:
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
                    if limit is not None and len(images) >= limit:
                        return images
        elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(source)
            if limit is not None and len(images) >= limit:
                return images
    return images


def infer_label_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    parts = list(path.parts)
    if "images" in parts:
        image_index = parts.index("images")
        parts[image_index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return path.with_suffix(".txt")


def image_id_from_path(data_yaml: str | Path, image_path: str | Path) -> str:
    yaml_path = Path(data_yaml).resolve()
    root = yaml_path.parent
    return str(Path(image_path).resolve().relative_to(root))


def list_split_samples(data_yaml: str | Path, split: str, limit: int | None = None) -> list[dict[str, Path | str]]:
    sources = resolve_split_sources(data_yaml=data_yaml, split=split)
    images = collect_image_files(sources=sources, limit=limit)
    samples: list[dict[str, Path | str]] = []
    for image_path in images:
        samples.append(
            {
                "image_path": image_path,
                "label_path": infer_label_path(image_path),
                "image_id": image_id_from_path(data_yaml, image_path),
            }
        )
    return samples
