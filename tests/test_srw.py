from __future__ import annotations

import torch

from src.models.srw import SRWModule


def test_srw_preserves_shape_and_gate_ranges() -> None:
    feature_map = torch.randn(2, 32, 10, 12, requires_grad=True)
    saliency_map = torch.rand(2, 1, 10, 12)
    module = SRWModule(channels=32)
    output, gate_s, gate_c, alpha = module(feature_map, saliency_map, return_gates=True)
    assert output.shape == feature_map.shape
    assert gate_s.shape == (2, 1, 10, 12)
    assert gate_c.shape == (2, 32, 1, 1)
    assert torch.all(gate_s >= 0.0)
    assert torch.all(gate_s <= 1.0)
    assert torch.all(gate_c >= 0.0)
    assert torch.all(gate_c <= 1.0)
    assert alpha.item() != 0.0


def test_srw_backward_flows_through_inputs_and_params() -> None:
    feature_map = torch.randn(2, 16, 8, 8, requires_grad=True)
    saliency_map = torch.zeros(2, 1, 8, 8)
    module = SRWModule(channels=16)
    loss = module(feature_map, saliency_map).mean()
    loss.backward()
    assert feature_map.grad is not None
    assert module.alpha.grad is not None


def test_srw_one_saliency_does_not_crash() -> None:
    feature_map = torch.randn(1, 8, 4, 4)
    saliency_map = torch.ones(1, 1, 4, 4)
    module = SRWModule(channels=8)
    output = module(feature_map, saliency_map)
    assert output.shape == feature_map.shape
