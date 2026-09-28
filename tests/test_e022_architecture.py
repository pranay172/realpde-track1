import unittest
import torch
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

from realpde_t1.adapter import (
    HeteroskedasticResidualConv3dAdapter,
    HeteroskedasticResidualCNO,
    wrap_frozen_heteroskedastic_cno,
)
from realpde_t1.training import heteroskedastic_residual_loss
from load_baseline import build_model


class TestE022Architecture(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.backbone = build_model("cno", device="cpu")
        self.adapter_config = {
            "name": "heteroskedastic_residual_conv3d",
            "input": "concat_input_and_backbone",
            "in_channels": 6,
            "out_channels": 3,
            "uncertainty_channels": 2,
            "hidden_channels": 32,
            "n_blocks": 2,
            "kernel_size": 3,
            "zero_init_last": True,
            "initial_log_var": -4.0,
        }
        self.model = wrap_frozen_heteroskedastic_cno(self.backbone, self.adapter_config)

    def test_zero_initialization(self):
        """At step 0, the model output should be identical to the backbone."""
        x = torch.randn(2, 20, 32, 64, 3)
        with torch.no_grad():
            backbone_out = self.backbone(x)
            pred, log_var = self.model(x)
        self.assertTrue(torch.allclose(pred, backbone_out, atol=1e-6))
        self.assertEqual(log_var.shape, (2, 20, 32, 64, 2))
        self.assertTrue(torch.allclose(log_var, torch.tensor(-4.0), atol=1e-5))

    def test_trainable_parameters(self):
        """Backbone parameters must be frozen, only adapter parameters trainable."""
        trainable = self.model.trainable_parameters()
        total_trainable = sum(p.numel() for p in trainable)
        # 6->32 (6*32*27+32 = 5216), 32->32 (32*32*27+32 = 27680), 32->3 (32*3*27+3 = 2595), 32->2 (32*2*27+2 = 1730)
        self.assertGreater(total_trainable, 30000)
        self.assertLess(total_trainable, 50000)
        for name, param in self.backbone.named_parameters():
            self.assertFalse(param.requires_grad, f"backbone param {name} is not frozen")

    def test_heteroskedastic_loss(self):
        """Loss computation must produce finite positive total loss."""
        pred = torch.randn(2, 20, 32, 64, 3)
        log_var = torch.full((2, 20, 32, 64, 2), -3.0)
        target = torch.randn(2, 20, 32, 64, 3)
        mean_tg = torch.zeros(3)
        std_tg = torch.ones(3)

        loss, parts = heteroskedastic_residual_loss(
            pred, log_var, target, mean_tg, std_tg, lambda_mean=0.1, lambda_nll=0.05
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("field_rel", parts)
        self.assertIn("mean_wake", parts)
        self.assertIn("nll", parts)


if __name__ == "__main__":
    unittest.main()
