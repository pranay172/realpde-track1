import sys
from pathlib import Path
import unittest
import torch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.training import (
    spatial_power_spectrum_2d,
    log_spectral_loss,
    mfdr_spectral_loss,
)


class TestMFDRLoss(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.H, self.W, self.C = 2, 20, 32, 64, 3
        self.mean_target = torch.tensor([0.15, -0.0004, 0.0], dtype=torch.float32)
        self.std_target = torch.tensor([0.11, 0.015, 1.0], dtype=torch.float32)

    def test_spatial_power_spectrum_shape_and_values(self):
        x = torch.randn(self.B, self.T, self.H, self.W, 2)
        psd = spatial_power_spectrum_2d(x)
        self.assertEqual(psd.shape, (self.B, self.H, self.W // 2 + 1))
        self.assertTrue(torch.all(psd >= 0.0))

    def test_log_spectral_loss_zero_on_identical(self):
        x = torch.randn(self.B, self.T, self.H, self.W, 2)
        loss = log_spectral_loss(x, x)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_mfdr_spectral_loss_finite_and_gradients(self):
        pred_norm = torch.randn(
            self.B, self.T, self.H, self.W, self.C, requires_grad=True
        )
        target_norm = torch.randn(self.B, self.T, self.H, self.W, self.C)

        total, parts = mfdr_spectral_loss(
            pred_norm,
            target_norm,
            self.mean_target,
            self.std_target,
            lambda_mean=0.2,
            lambda_fluct=0.1,
            lambda_tke=0.05,
            lambda_spec=0.02,
        )

        self.assertTrue(torch.isfinite(total))
        for k, v in parts.items():
            self.assertTrue(torch.isfinite(v), f"part {k} is non-finite: {v}")

        total.backward()
        self.assertIsNotNone(pred_norm.grad)
        self.assertTrue(torch.all(torch.isfinite(pred_norm.grad)))


if __name__ == "__main__":
    unittest.main()
