from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.validation import (  # noqa: E402
    files_for_reynolds,
    load_config,
    parse_trajectory_name,
)


class E009ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.e002 = load_config(REPO / "configs" / "e002_split.json")
        self.e009 = json.loads(
            (REPO / "configs" / "e009_oof_uncertainty.json").read_text(encoding="utf-8")
        )
        self.secondary = load_config(REPO / "configs" / "e009_secondary_fold.json")

    def test_files_for_reynolds_requires_exact_group_coverage(self) -> None:
        files = ["5025_0.h5", "5025_5.h5", "12675_0.h5"]
        self.assertEqual(
            files_for_reynolds(files, [5025, 12675]),
            ["5025_0.h5", "5025_5.h5", "12675_0.h5"],
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            files_for_reynolds(files, [5025, 19050])

    def test_oof_partitions_are_disjoint_and_cover_e002_training(self) -> None:
        train = list(self.e002["training_files"])
        oof_train = files_for_reynolds(train, self.e009["training_nominal_re"])
        calibration = files_for_reynolds(
            train, self.e009["training_side_calibration"]["calibration_nominal_re"]
        )
        self.assertEqual(len(oof_train), 45)
        self.assertEqual(len(calibration), 19)
        self.assertEqual(set(oof_train) & set(calibration), set())
        self.assertEqual(set(oof_train) | set(calibration), set(train))
        self.assertEqual(
            {parse_trajectory_name(name)[0] for name in oof_train},
            set(self.e009["training_nominal_re"]),
        )
        self.assertEqual(
            {parse_trajectory_name(name)[0] for name in calibration},
            {5025, 12675, 19050, 25425},
        )

    def test_interval_grid_matches_e006_and_contains_e006_selection(self) -> None:
        e006 = json.loads(
            (REPO / "configs" / "e006_uncertainty_calibration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.e009["interval"]["alpha_grid"], e006["interval"]["alpha_grid"])
        self.assertEqual(self.e009["interval"]["beta_grid"], e006["interval"]["beta_grid"])
        self.assertEqual(self.e009["interval"]["sigma_global"], e006["interval"]["sigma_global"])
        self.assertIn(self.e009["interval"]["e006_alpha"], self.e009["interval"]["alpha_grid"])
        self.assertIn(self.e009["interval"]["e006_beta"], self.e009["interval"]["beta_grid"])

    def test_secondary_fold_is_frozen_and_disjoint_from_prior_holdouts(self) -> None:
        e002_val = {parse_trajectory_name(name)[0] for name in self.e002["validation_files"]}
        e009_cal = set(self.e009["training_side_calibration"]["calibration_nominal_re"])
        secondary_val = {
            parse_trajectory_name(name)[0] for name in self.secondary["validation_files"]
        }
        secondary_train = {
            parse_trajectory_name(name)[0] for name in self.secondary["training_files"]
        }
        self.assertEqual(secondary_val, {6300, 13950, 21600, 24150})
        self.assertEqual(len(self.secondary["validation_files"]), 19)
        self.assertEqual(len(self.secondary["training_files"]), 62)
        self.assertEqual(secondary_val & e002_val, set())
        self.assertEqual(secondary_val & e009_cal, set())
        self.assertEqual(secondary_val & secondary_train, set())
        self.assertEqual(self.secondary["excluded_files"], ["7575_0.h5"])
        declared = (
            self.secondary["training_files"]
            + self.secondary["validation_files"]
            + self.secondary["excluded_files"]
        )
        e002_declared = (
            self.e002["training_files"]
            + self.e002["validation_files"]
            + self.e002["excluded_files"]
        )
        self.assertEqual(set(declared), set(e002_declared))
        self.assertEqual(len(declared), 82)


if __name__ == "__main__":
    unittest.main()
