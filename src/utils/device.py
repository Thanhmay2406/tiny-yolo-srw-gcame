from __future__ import annotations

from typing import Any, Dict

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def get_device_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_count": 0,
    }

    if torch is None:
        return summary

    summary["torch_version"] = torch.__version__
    summary["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        summary["device"] = "cuda"
        summary["gpu_count"] = torch.cuda.device_count()
        summary["gpu_name"] = torch.cuda.get_device_name(0)

    return summary


def resolve_device(prefer: str = "auto") -> str:
    if torch is None:
        return "cpu"
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"
