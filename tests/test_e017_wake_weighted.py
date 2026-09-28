from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.validation import load_config  # noqa: E402


class E017ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (REPO / "configs" / "e017_wake_weighted_loss.json").read_text(encoding="utf-8")
        )
        self.split = load_config(REPO / self.config["split_config"])

    def test_protocol_is_e005_recipe_plus_vorticity_weights_on_fold_a(self) -> None:
        loss = self.config["training"]["loss"]
        evaluation = self.config["evaluation"]
        self.assertEqual(self.config["experiment"], "E017")
        self.assertEqual(self.config["seed"], 20260815)
        self.assertEqual(self.config["split_config"], "configs/e002_split.json")
        self.assertEqual(self.split["excluded_files"], ["7575_0.h5"])
        self.assertEqual(len(self.split["training_files"]), 64)
        self.assertEqual(self.config["training"]["num_updates"], 600)
        self.assertEqual(self.config["training"]["batch_size"], 4)
        self.assertEqual(loss["name"], "wake_weighted_relative_mse")
        self.assertEqual(loss["weight"], "target_vorticity_dt")
        self.assertEqual(loss["alpha"], 1.0)
        self.assertNotIn("lambda_tke", loss)
        self.assertEqual(evaluation["fold"], "A")
        self.assertEqual(evaluation["persist_first_frames"], 4)
        self.assertEqual(evaluation["interval"]["alpha"], 0.025)
        self.assertEqual(evaluation["interval"]["beta"], 0.15)
        self.assertEqual(
            evaluation["success_criteria"]["persist_first_4_must_beat_e013"],
            ["rel_l2_score", "mvpe_score"],
        )
        self.assertEqual(
            evaluation["success_criteria"]["persist_first_4_tke_max_regression_vs_e013"],
            1.0,
        )
        self.assertNotIn("Fold C", json.dumps(self.config["evaluation"]))
        self.assertIn("package only if the Fold A persist-first-4 gates pass", self.config["promotion"])

    def test_e013_anchors_match_e012_fold_a_persist_first(self) -> None:
        comparison_path = REPO / "artifacts/e012_multifold/comparison.json"
        if not comparison_path.is_file():
            self.skipTest("local E012 comparison artifact is not present")
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        e013 = comparison["folds"]["A"]["methods"]["e005_persist_first_4"]
        anchors = self.config["evaluation"]["e013_persist_first_4"]
        self.assertAlmostEqual(
            anchors["rel_l2_score"], e013["default"]["scores"]["rel_l2_score"]
        )
        self.assertAlmostEqual(anchors["tke_score"], e013["default"]["scores"]["tke_score"])
        self.assertAlmostEqual(anchors["mvpe_score"], e013["default"]["scores"]["mvpe_score"])
        self.assertAlmostEqual(
            anchors["calibrated_sps_score"], e013["calibrated"]["scores"]["sps_score"]
        )


if __name__ == "__main__":
    unittest.main()
