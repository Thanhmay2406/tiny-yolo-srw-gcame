from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseSaliencyProvider(nn.Module, ABC):
    mode: str

    @abstractmethod
    def forward(
        self,
        feature_map: torch.Tensor,
        image_ids: list[str] | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
