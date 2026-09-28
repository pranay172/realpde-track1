from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.training import (  # noqa: E402
    RealWindowDataset,
    WindowGeometry,
    compute_gaussian_stats,
    enumerate_windows,
    fluctuation_relative_mse,
    official_tke_maps,
    scale_balanced_uv_relative_mse,
    split_mean_fluct_tke_loss,
    temporal_mean_relative_mse,
    tke_relative_l2,
    vorticity_dt_weights,
    wake_weighted_relative_mse,
    make_optimizer,
)


def _write(path: Path, *, grouped: bool = False) -> np.ndarray:
    base = np.arange(100 * 4 * 6, dtype=np.float32).reshape(100, 4, 6)
    with h5py.File(path, "w") as handle:
        root = handle.create_group("measured_data") if grouped else handle
        root["u"] = base
        root["v"] = base * np.float32(-0.25)
    return base


class TrainingDataTests(unittest.TestCase):
    def test_loader_supports_both_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write(root / "100_0.h5")
            _write(root / "200_0.h5", grouped=True)
            geometry = WindowGeometry(spatial_stride=2)
            dataset = RealWindowDataset(
                root, ["100_0.h5", "200_0.h5"], geometry, preload=True
            )
            self.assertEqual(len(dataset), 8)
            inputs, targets = dataset[0]
            self.assertEqual(tuple(inputs.shape), (20, 2, 3, 3))
            self.assertEqual(tuple(targets.shape), tuple(inputs.shape))
            self.assertTrue(np.all(inputs.numpy()[..., 2] == 0.0))

    def test_stats_match_brute_force_windows(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write(root / "100_0.h5")
            geometry = WindowGeometry(spatial_stride=2)
            dataset = RealWindowDataset(root, ["100_0.h5"], geometry)
            inputs = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
            targets = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
            stats = compute_gaussian_stats(root, ["100_0.h5"], geometry)
            input64 = inputs.astype(np.float64)
            target64 = targets.astype(np.float64)
            expected_input_mean = input64.mean(axis=(0, 1, 2, 3))
            expected_target_mean = target64.mean(axis=(0, 1, 2, 3))
            expected_input_std = input64.std(axis=(0, 1, 2, 3))
            expected_target_std = target64.std(axis=(0, 1, 2, 3))
            expected_input_std[2] = 1.0
            expected_target_std[2] = 1.0
            np.testing.assert_allclose(stats.mean_input, expected_input_mean, rtol=1e-6)
            np.testing.assert_allclose(stats.std_input, expected_input_std, rtol=1e-6)
            np.testing.assert_allclose(stats.mean_target, expected_target_mean, rtol=1e-6)
            np.testing.assert_allclose(stats.std_target, expected_target_std, rtol=1e-6)

    def test_window_count(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write(root / "100_0.h5")
            entries = enumerate_windows(root, ["100_0.h5"], WindowGeometry())
            self.assertEqual(entries, [("100_0.h5", 0), ("100_0.h5", 20), ("100_0.h5", 40), ("100_0.h5", 60)])

    def test_scale_balanced_loss_is_scale_invariant_and_ignores_pressure(self) -> None:
        mean = torch.zeros(3)
        std = torch.ones(3)
        target = torch.ones((2, 2, 1, 1, 3))
        prediction = target.clone()
        prediction[..., :2] += 0.5
        prediction[..., 2] = 1000.0
        loss = scale_balanced_uv_relative_mse(prediction, target, mean, std)

        scaled_loss = scale_balanced_uv_relative_mse(
            prediction * 4.0, target * 4.0, mean, std
        )
        pressure_reset = prediction.clone()
        pressure_reset[..., 2] = target[..., 2]
        pressure_loss = scale_balanced_uv_relative_mse(
            pressure_reset, target, mean, std
        )

        self.assertAlmostEqual(float(loss), 0.25)
        self.assertAlmostEqual(float(scaled_loss), float(loss))
        self.assertAlmostEqual(float(pressure_loss), float(loss))

    def test_scale_balanced_loss_validates_inputs(self) -> None:
        tensor = torch.zeros((1, 2, 1, 1, 3))
        with self.assertRaises(ValueError):
            scale_balanced_uv_relative_mse(
                tensor, tensor[..., :2], torch.zeros(3), torch.ones(3)
            )
        with self.assertRaises(ValueError):
            scale_balanced_uv_relative_mse(
                tensor, tensor, torch.zeros(3), torch.ones(3), denominator_epsilon=0
            )

    def test_split_loss_separates_mean_and_fluctuation(self) -> None:
        mean = torch.zeros(3)
        std = torch.ones(3)
        time = torch.linspace(0.0, 1.0, 8).view(1, 8, 1, 1, 1)
        target = torch.zeros((1, 8, 2, 2, 3))
        target[..., 0] = 2.0 + time[..., 0]
        target[..., 1] = -0.5
        mean_shift = target.clone()
        mean_shift[..., :2] += 0.4
        mean_shift[..., 2] = 9.0
        time_mean = time.mean(dim=1, keepdim=True)
        fluct_shift = target.clone()
        fluct_shift[..., 0] = 2.0 + time_mean[..., 0] + 2.0 * (time[..., 0] - time_mean[..., 0])
        field, parts = split_mean_fluct_tke_loss(
            mean_shift, target, mean, std, lambda_mean=1.0, lambda_fluct=1.0, lambda_tke=1.0
        )
        self.assertGreater(float(parts["mean"]), 0.0)
        self.assertAlmostEqual(float(parts["fluct"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["tke"]), 0.0, places=6)
        self.assertGreater(float(field), 0.0)
        _, fluct_parts = split_mean_fluct_tke_loss(
            fluct_shift, target, mean, std, lambda_mean=1.0, lambda_fluct=1.0, lambda_tke=1.0
        )
        self.assertAlmostEqual(float(fluct_parts["mean"]), 0.0, places=6)
        self.assertGreater(float(fluct_parts["fluct"]), 0.0)
        self.assertGreater(float(fluct_parts["tke"]), 0.0)

    def test_tke_maps_match_official_formula(self) -> None:
        uv = torch.zeros((1, 4, 1, 1, 2))
        uv[0, :, 0, 0, 0] = torch.tensor([1.0, 3.0, 5.0, 7.0])
        uv[0, :, 0, 0, 1] = torch.tensor([0.0, 2.0, 0.0, 2.0])
        tke = official_tke_maps(uv)
        u_var = float(torch.var(uv[0, :, 0, 0, 0], unbiased=False))
        v_var = float(torch.var(uv[0, :, 0, 0, 1], unbiased=False))
        self.assertAlmostEqual(float(tke), 0.5 * (u_var + v_var), places=6)

    def test_split_loss_rejects_negative_weights_and_spectrum_is_separate(self) -> None:
        tensor = torch.zeros((1, 2, 1, 1, 3))
        with self.assertRaises(ValueError):
            split_mean_fluct_tke_loss(
                tensor, tensor, torch.zeros(3), torch.ones(3), lambda_tke=-0.1
            )
        self.assertAlmostEqual(
            float(temporal_mean_relative_mse(tensor, tensor, torch.zeros(3), torch.ones(3))),
            0.0,
        )
        self.assertAlmostEqual(
            float(fluctuation_relative_mse(tensor, tensor, torch.zeros(3), torch.ones(3))),
            0.0,
        )
        self.assertAlmostEqual(
            float(tke_relative_l2(tensor, tensor, torch.zeros(3), torch.ones(3))),
            0.0,
        )

    def test_wake_weights_recover_e005_when_alpha_is_zero(self) -> None:
        mean = torch.zeros(3)
        std = torch.ones(3)
        target = torch.rand((2, 4, 8, 10, 3))
        target[..., 2] = 7.0
        prediction = target + 0.25
        prediction[..., 2] = -3.0
        baseline = scale_balanced_uv_relative_mse(prediction, target, mean, std)
        weighted, parts = wake_weighted_relative_mse(
            prediction, target, mean, std, alpha=0.0
        )
        self.assertAlmostEqual(float(weighted), float(baseline), places=6)
        self.assertAlmostEqual(float(parts["unweighted"]), float(baseline), places=6)
        self.assertAlmostEqual(float(parts["mean_weight"]), 1.0, places=6)
        scaled, _ = wake_weighted_relative_mse(
            prediction * 3.0, target * 3.0, mean, std, alpha=1.0
        )
        original, _ = wake_weighted_relative_mse(
            prediction, target, mean, std, alpha=1.0
        )
        self.assertAlmostEqual(float(scaled), float(original), places=5)

    def test_wake_weights_concentrate_on_time_varying_vorticity(self) -> None:
        field = torch.zeros((1, 4, 8, 12, 2))
        x = torch.linspace(0.0, 1.0, 12).view(1, 1, 1, 12)
        time = torch.arange(4).float().view(1, 4, 1, 1)
        field[:, :, :, 6:, 1] = time * x[:, :, :, 6:]
        weights = vorticity_dt_weights(field, alpha=1.0)
        left = float(weights[:, :, :, :6].mean())
        right = float(weights[:, :, :, 6:].mean())
        self.assertGreater(right, left)
        self.assertGreater(right, 1.0)
        with self.assertRaises(ValueError):
            vorticity_dt_weights(field, alpha=-1.0)

    def test_make_optimizer_selects_adam_and_adamw(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(3))
        adam = make_optimizer(
            [parameter], name="Adam", learning_rate=1e-3, betas=(0.9, 0.999), weight_decay=0.1
        )
        adamw = make_optimizer(
            [parameter], name="AdamW", learning_rate=1e-3, betas=(0.9, 0.999), weight_decay=0.1
        )
        self.assertIsInstance(adam, torch.optim.Adam)
        self.assertNotIsInstance(adam, torch.optim.AdamW)
        self.assertIsInstance(adamw, torch.optim.AdamW)
        with self.assertRaisesRegex(ValueError, "unsupported optimizer"):
            make_optimizer(
                [parameter], name="SGD", learning_rate=1e-3, betas=(0.9, 0.999), weight_decay=0.0
            )


if __name__ == "__main__":
    unittest.main()
