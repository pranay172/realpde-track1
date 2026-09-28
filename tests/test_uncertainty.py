from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

import scoring  # noqa: E402
from realpde_t1.uncertainty import (  # noqa: E402
    candidate_interval_sums,
    finalize_sps,
    interval_bounds_numpy,
)


class UncertaintyTests(unittest.TestCase):
    def test_streaming_candidate_sps_matches_official_scorer(self) -> None:
        rng = np.random.default_rng(7)
        target = rng.normal(0.1, 0.03, size=(2, 20, 32, 64, 3)).astype(np.float32)
        prediction = target + rng.normal(0, 0.01, size=target.shape).astype(np.float32)
        target[..., 2] = 0.0
        prediction[..., 2] = 0.0
        target[:, :, :2, :3, :2] = 0.0
        errors = np.stack(
            [
                scoring.rel_l2_per_sample(prediction, target, 2),
                scoring.tke_rel_l2_per_sample(prediction, target, 2),
                scoring.mvpe_rel_l2_per_sample(prediction, target),
            ],
            axis=1,
        )
        accuracy = 1.0 - errors / (0.5 + errors)
        alpha = torch.tensor([0.05, 0.025], dtype=torch.float32)
        beta = torch.tensor([0.0, 0.075], dtype=torch.float32)
        branch, coverage_count, scored = candidate_interval_sums(
            torch.from_numpy(prediction),
            torch.from_numpy(target),
            torch.from_numpy(accuracy),
            alpha,
            beta,
            scoring.SIGMA_GLOBAL,
        )
        sps, coverage = finalize_sps(
            branch.numpy(), coverage_count.numpy(), scored
        )
        for index, (a, b) in enumerate(zip(alpha.numpy(), beta.numpy())):
            lower, upper = interval_bounds_numpy(
                prediction, float(a), float(b), scoring.SIGMA_GLOBAL
            )
            expected_sps, expected_coverage = scoring.aggregate_sps(
                prediction, target, 2, lower=lower, upper=upper
            )
            self.assertAlmostEqual(sps[index], 100.0 * expected_sps, places=4)
            self.assertAlmostEqual(coverage[index], expected_coverage, places=6)

    def test_bounds_contract_and_pressure(self) -> None:
        prediction = np.ones((1, 20, 32, 64, 3), dtype=np.float32)
        prediction[..., 2] = 0.0
        lower, upper = interval_bounds_numpy(prediction, 0.05, 0.1, 0.056)
        self.assertEqual(lower.shape, prediction.shape)
        self.assertTrue(np.all(lower <= prediction))
        self.assertTrue(np.all(prediction <= upper))
        self.assertTrue(np.all(lower[..., 2] == 0.0))
        self.assertTrue(np.all(upper[..., 2] == 0.0))


if __name__ == "__main__":
    unittest.main()
