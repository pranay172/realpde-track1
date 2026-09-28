from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.validation import load_config  # noqa: E402


class E025ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ensemble = json.loads(
            (REPO / "configs" / "e025_fold_b_ensemble.json").read_text(encoding="utf-8")
        )
        self.hetero = json.loads(
            (REPO / "configs" / "e025_fold_b_heteroskedastic.json").read_text(encoding="utf-8")
        )
        self.adapter3 = json.loads(
            (REPO / "configs" / "e025_fold_b_adapter3.json").read_text(encoding="utf-8")
        )
        self.split = load_config(REPO / self.ensemble["split_config"])

    def test_members_use_fold_b_split_and_e010_backbone(self) -> None:
        self.assertEqual(self.ensemble["experiment"], "E025")
        self.assertEqual(self.ensemble["split_config"], "configs/e009_secondary_fold.json")
        self.assertEqual(self.split["excluded_files"], ["7575_0.h5"])
        self.assertEqual(len(self.split["training_files"]), 62)
        self.assertEqual(self.ensemble["persist_first_frames"], 4)
        self.assertEqual(self.ensemble["interval"]["alpha"], 0.025)
        self.assertEqual(self.ensemble["interval"]["beta"], 0.15)
        self.assertEqual(
            self.ensemble["members"][0]["checkpoint"],
            "artifacts/e020_secondary_fold_residual/final.pth",
        )
        self.assertTrue(self.ensemble["members"][0]["already_trained"])
        for cfg in (self.hetero, self.adapter3):
            self.assertEqual(cfg["split_config"], "configs/e009_secondary_fold.json")
            self.assertEqual(
                cfg["initial_checkpoint"], "artifacts/e010_secondary_fold/final.pth"
            )
            self.assertEqual(cfg["evaluation"]["fold"], "B")
        self.assertEqual(self.adapter3["training"]["optimizer"], "AdamW")
        self.assertEqual(self.hetero["training"]["optimizer"], "AdamW")
        self.assertEqual(
            self.ensemble["success_criteria"]["persist_first_4_must_beat_e020"],
            ["rel_l2_score", "mvpe_score"],
        )
        self.assertIn("evidence only", self.ensemble["promotion"])
        self.assertIn("Do not score Fold A", self.ensemble["limitations"][0])


if __name__ == "__main__":
    unittest.main()
