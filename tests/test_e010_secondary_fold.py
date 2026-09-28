from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.validation import load_config, parse_trajectory_name  # noqa: E402


class E010ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.e005 = json.loads(
            (REPO / "configs" / "e005_scale_balanced_uv.json").read_text(encoding="utf-8")
        )
        self.e010 = json.loads(
            (REPO / "configs" / "e010_secondary_fold_robustness.json").read_text(encoding="utf-8")
        )
        self.secondary = load_config(REPO / "configs" / "e009_secondary_fold.json")
        self.e002 = load_config(REPO / "configs" / "e002_split.json")

    def test_training_recipe_matches_e005_except_split(self) -> None:
        for key in ("optimizer", "learning_rate", "betas", "weight_decay",
                    "scheduler", "eta_min", "batch_size", "num_updates",
                    "precision", "gradient_clip_norm"):
            self.assertEqual(self.e010["training"][key], self.e005["training"][key], key)
        self.assertEqual(self.e010["training"]["loss"], self.e005["training"]["loss"])
        self.assertEqual(self.e010["seed"], self.e005["seed"])
        self.assertEqual(self.e010["initial_checkpoint"], self.e005["initial_checkpoint"])
        self.assertEqual(self.e010["split_config"], "configs/e009_secondary_fold.json")
        self.assertNotEqual(self.e010["split_config"], self.e005["split_config"])

    def test_training_files_are_the_reserved_complementary_set(self) -> None:
        self.assertEqual(self.e010["split_config"], "configs/e009_secondary_fold.json")
        self.assertEqual(len(self.secondary["training_files"]), 62)
        self.assertEqual(len(self.secondary["validation_files"]), 19)
        train_re = {parse_trajectory_name(name)[0] for name in self.secondary["training_files"]}
        val_re = {parse_trajectory_name(name)[0] for name in self.secondary["validation_files"]}
        self.assertEqual(val_re, {6300, 13950, 21600, 24150})
        self.assertTrue({3750, 11400, 20325, 26700}.issubset(train_re))
        self.assertEqual(train_re & val_re, set())

    def test_e002_holdout_is_not_the_e010_holdout(self) -> None:
        e002_val = set(self.e002["validation_files"])
        e010_val = set(self.secondary["validation_files"])
        e010_train = set(self.secondary["training_files"])
        self.assertEqual(e002_val & e010_val, set())
        self.assertTrue(e002_val.issubset(e010_train))

    def test_persistence_helper_matches_e002_contract(self) -> None:
        sys.path.insert(0, str(REPO / "scripts"))
        from e010_evaluate_cno import persistence_prediction  # noqa: WPS433

        inputs = np.zeros((2, 20, 4, 6, 3), dtype=np.float32)
        inputs[:, -1, 0, 0, 0] = 3.0
        inputs[:, -1, 0, 0, 1] = -1.5
        inputs[:, -1, 0, 0, 2] = 9.0
        output = persistence_prediction(inputs)
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(np.all(output[:, :, 0, 0, 0] == 3.0))
        self.assertTrue(np.all(output[:, :, 0, 0, 1] == -1.5))
        self.assertTrue(np.all(output[..., 2] == 0.0))


if __name__ == "__main__":
    unittest.main()
