from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.diagnostics import checkpoint_drift, lead_relative_l2  # noqa: E402


class DiagnosticTests(unittest.TestCase):
    def test_lead_relative_l2(self) -> None:
        target = np.ones((2, 3, 1, 1, 3), dtype=np.float32)
        prediction = target.copy()
        prediction[:, 1, ..., :2] = 0.0
        self.assertEqual(lead_relative_l2(prediction, target), [0.0, 1.0, 0.0])

    def test_checkpoint_drift_separates_batchnorm_state(self) -> None:
        initial = {
            "encoder.weight": torch.tensor([1.0, 2.0]),
            "encoder.batch_norm.weight": torch.tensor([1.0, 1.0]),
            "encoder.batch_norm.bias": torch.tensor([0.0, 0.0]),
            "encoder.batch_norm.running_mean": torch.tensor([0.0, 0.0]),
            "encoder.batch_norm.running_var": torch.tensor([1.0, 4.0]),
            "encoder.batch_norm.num_batches_tracked": torch.tensor(2),
        }
        final = {name: value.clone() for name, value in initial.items()}
        final["encoder.weight"] += 0.1
        final["encoder.batch_norm.running_mean"] = torch.tensor([1.0, 1.0])
        final["encoder.batch_norm.running_var"] = torch.tensor([4.0, 1.0])
        final["encoder.batch_norm.num_batches_tracked"] += 3

        result = checkpoint_drift(initial, final)

        self.assertEqual(result["batchnorm_running_state"]["layers"], 1)
        self.assertEqual(
            result["batchnorm_running_state"]["num_batches_increment_unique"], [3]
        )
        self.assertGreater(
            result["batchnorm_running_state"]["running_mean_shift_standardized_rms"], 0.5
        )
        self.assertEqual(result["categories"]["all_trainable"]["elements"], 6)
        self.assertGreater(result["categories"]["all_trainable"]["relative_l2"], 0.0)

    def test_checkpoint_drift_rejects_key_changes(self) -> None:
        with self.assertRaises(ValueError):
            checkpoint_drift({"a": torch.ones(1)}, {"b": torch.ones(1)})


if __name__ == "__main__":
    unittest.main()
