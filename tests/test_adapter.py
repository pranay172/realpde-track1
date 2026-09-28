from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
from torch import nn


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.adapter import (  # noqa: E402
    ResidualConv3dAdapter,
    ResidualCNO,
    freeze_module,
    wrap_frozen_cno,
)


class TinyBackbone(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 0.5


class AdapterTests(unittest.TestCase):
    def test_zero_init_adapter_is_identity_on_the_backbone(self) -> None:
        backbone = TinyBackbone()
        model = wrap_frozen_cno(
            backbone,
            {
                "name": "residual_conv3d",
                "input": "concat_input_and_backbone",
                "hidden_channels": 8,
                "n_blocks": 2,
                "kernel_size": 3,
                "zero_init_last": True,
            },
        )
        inputs = torch.randn(2, 4, 6, 8, 3)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), backbone(inputs))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.adapter.parameters()))
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.adapter.training)

    def test_adapter_can_change_the_output_after_a_step(self) -> None:
        backbone = TinyBackbone()
        adapter = ResidualConv3dAdapter(hidden_channels=4, n_blocks=1, zero_init_last=False)
        model = ResidualCNO(backbone, adapter)
        freeze_module(backbone)
        inputs = torch.randn(1, 2, 4, 4, 3)
        before = model(inputs).detach().clone()
        optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.1)
        loss = model(inputs).pow(2).mean()
        loss.backward()
        optimizer.step()
        after = model(inputs)
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(tuple(after.shape), tuple(inputs.shape))

    def test_wrap_rejects_unfreeze_and_unknown_adapters(self) -> None:
        with self.assertRaises(ValueError):
            wrap_frozen_cno(TinyBackbone(), {"name": "lora"})
        with self.assertRaises(ValueError):
            ResidualConv3dAdapter(kernel_size=2)


if __name__ == "__main__":
    unittest.main()
