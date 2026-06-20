from __future__ import annotations

import torch

from src.xai.saliency_head import SaliencyHead


def test_saliency_head_shape_range_and_gradients() -> None:
    feature_map = torch.randn(2, 64, 20, 20, requires_grad=True)
    head = SaliencyHead(in_channels=64)
    saliency = head(feature_map)
    assert saliency.shape == (2, 1, 20, 20)
    assert torch.all(saliency >= 0.0)
    assert torch.all(saliency <= 1.0)
    saliency.mean().backward()
    assert feature_map.grad is not None


def test_saliency_head_handles_small_feature_maps() -> None:
    feature_map = torch.randn(1, 16, 2, 2)
    head = SaliencyHead(in_channels=16)
    saliency = head(feature_map)
    assert saliency.shape == (1, 1, 2, 2)
