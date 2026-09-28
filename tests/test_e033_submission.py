"""Unit tests for standalone E033 Champion Pareto Ensemble submission."""

from __future__ import annotations

import numpy as np
import torch

from submission.e033.submission import (
    FiLMResidualConv3dAdapter,
    ResidualConv3dAdapter,
    extract_flow_invariants,
)


def test_extract_flow_invariants_shape_and_range() -> None:
    inputs = torch.randn(2, 20, 32, 64, 3)
    z_flow = extract_flow_invariants(inputs)
    assert z_flow.shape == (2, 4)
    assert not torch.isnan(z_flow).any()
    assert not torch.isinf(z_flow).any()


def test_film_adapter_invariance() -> None:
    adapter = FiLMResidualConv3dAdapter(
        in_channels=6, out_channels=3, flow_dim=4, hidden_channels=16, n_blocks=2, kernel_size=3
    ).eval()
    x = torch.randn(2, 20, 32, 64, 3)
    base = torch.randn(2, 20, 32, 64, 3)

    out_batched = adapter(x, base)
    out_single0 = adapter(x[:1], base[:1])
    out_single1 = adapter(x[1:], base[1:])

    diff0 = (out_batched[:1] - out_single0).abs().max().item()
    diff1 = (out_batched[1:] - out_single1).abs().max().item()

    assert diff0 < 1e-5
    assert diff1 < 1e-5


def test_residual_adapter_invariance() -> None:
    adapter = ResidualConv3dAdapter(
        in_channels=6, out_channels=3, hidden_channels=16, n_blocks=2, kernel_size=3
    ).eval()
    x = torch.randn(2, 20, 32, 64, 3)
    base = torch.randn(2, 20, 32, 64, 3)

    out_batched = adapter(x, base)
    out_single0 = adapter(x[:1], base[:1])
    diff0 = (out_batched[:1] - out_single0).abs().max().item()
    assert diff0 < 1e-6
