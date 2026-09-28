import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.adapter import ResidualConv3dAdapter, wrap_frozen_cno
from realpde_t1.training import scale_balanced_uv_relative_mse


class TestRolloutCurriculum(unittest.TestCase):
    def test_2step_rollout_gradients(self):
        B = 2
        inputs = torch.randn(B, 20, 32, 64, 3, requires_grad=False)
        target30 = torch.randn(B, 30, 32, 64, 3, requires_grad=False)

        mean_tg = torch.tensor([0.15, -0.0004, 0.0], dtype=torch.float32)
        std_tg = torch.tensor([0.11, 0.015, 1.0], dtype=torch.float32)

        # Mock adapter
        adapter = ResidualConv3dAdapter(
            in_channels=6, out_channels=3, hidden_channels=8, n_blocks=2, zero_init_last=False
        )

        # Step 1
        dummy_base1 = torch.zeros_like(inputs)
        pred1 = dummy_base1 + adapter(inputs, dummy_base1)
        loss1 = scale_balanced_uv_relative_mse(
            pred1, target30[:, :20], mean_tg, std_tg, measured_channels=2
        )

        # Step 2: Rollout context
        rollout_input = torch.cat((inputs[:, 10:20], pred1[:, :10]), dim=1)
        self.assertEqual(rollout_input.shape, (B, 20, 32, 64, 3))

        dummy_base2 = torch.zeros_like(rollout_input)
        pred2 = dummy_base2 + adapter(rollout_input, dummy_base2)
        loss2 = scale_balanced_uv_relative_mse(
            pred2, target30[:, 10:30], mean_tg, std_tg, measured_channels=2
        )

        loss_total = 0.5 * loss1 + 0.5 * loss2
        loss_total.backward()

        self.assertTrue(torch.isfinite(loss_total))
        for param in adapter.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.all(torch.isfinite(param.grad)))


if __name__ == "__main__":
    unittest.main()
