from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.scaling import (  # noqa: E402
    apply_window_scale,
    window_channel_stds,
    window_scale_ratios,
)


class WindowScaleTests(unittest.TestCase):
    def test_identity_when_window_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 0.05, size=(1, 20, 8, 8, 3)).astype(np.float32)
        windows = np.repeat(base, 3, axis=0)
        windows[..., 2] = 0.0
        reference = window_channel_stds(windows)[0]
        ratios = window_scale_ratios(windows, reference)
        np.testing.assert_allclose(ratios[:, :2], 1.0, atol=1e-5)
        restored = apply_window_scale(
            apply_window_scale(windows, ratios, invert=True), ratios, invert=False
        )
        np.testing.assert_allclose(restored[..., :2], windows[..., :2], rtol=1e-5, atol=1e-6)
        self.assertTrue(np.all(restored[..., 2] == 0.0))

    def test_scale_round_trip_changes_std_then_restores(self) -> None:
        rng = np.random.default_rng(1)
        windows = np.zeros((2, 8, 4, 4, 3), dtype=np.float32)
        windows[0, ..., 0] = rng.normal(0.0, 2.0, size=(8, 4, 4)).astype(np.float32)
        windows[0, ..., 1] = rng.normal(0.0, 0.8, size=(8, 4, 4)).astype(np.float32)
        windows[1, ..., 0] = rng.normal(0.0, 0.2, size=(8, 4, 4)).astype(np.float32)
        windows[1, ..., 1] = rng.normal(0.0, 0.1, size=(8, 4, 4)).astype(np.float32)
        windows[:, 0, 0, 0, :2] = 0.0
        reference = np.array([1.0, 1.0], dtype=np.float32)
        ratios = window_scale_ratios(windows, reference)
        self.assertGreater(float(ratios[0, 0]), float(ratios[1, 0]))
        scaled = apply_window_scale(windows, ratios, invert=True)
        stds = window_channel_stds(scaled)
        np.testing.assert_allclose(stds, 1.0, atol=0.05)
        restored = apply_window_scale(scaled, ratios, invert=False)
        np.testing.assert_allclose(restored[..., :2], windows[..., :2], rtol=1e-5, atol=1e-6)

    def test_protocol_is_inference_only_on_fold_a(self) -> None:
        config = json.loads(
            (REPO / "configs" / "e015_window_scale.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["evaluation"]["fold"], "A")
        self.assertEqual(config["persist_first_frames"], 4)
        self.assertEqual(config["point_model"], "artifacts/e005_scale_balanced_uv/final.pth")
        self.assertIn("No training", config["controls"])


if __name__ == "__main__":
    unittest.main()
