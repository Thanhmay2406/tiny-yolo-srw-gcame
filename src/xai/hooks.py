from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForwardCapture:
    module: torch.nn.Module

    def __post_init__(self) -> None:
        self.output: torch.Tensor | None = None
        self.handle = self.module.register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.output = output

    def clear(self) -> None:
        self.output = None

    def remove(self) -> None:
        self.handle.remove()
