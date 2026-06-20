from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_environment(cache_root: str | Path | None = None) -> Path:
    existing = os.environ.get("MPLCONFIGDIR")
    if existing:
        existing_path = Path(existing)
        existing_path.mkdir(parents=True, exist_ok=True)
        return existing_path

    candidate = Path(cache_root) if cache_root is not None else Path("/tmp/matplotlib")
    candidate.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(candidate)
    return candidate
