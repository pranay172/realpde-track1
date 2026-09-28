from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]


class E014PreflightTests(unittest.TestCase):
    def test_preflight_uses_sim_pretrain_fno_and_fold_a_only(self) -> None:
        config = json.loads(
            (REPO / "configs" / "e014_fno_preflight.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["experiment"], "E014")
        self.assertEqual(config["phase"], "preflight")
        self.assertIn("sim_pretrain/sim_fno.pth", config["checkpoint"])
        self.assertNotIn("sim_real_ft", config["checkpoint"])
        self.assertEqual(config["evaluation"]["fold"], "A")
        self.assertEqual(config["persist_first_frames"], 4)
        self.assertEqual(
            set(config["normalizers"]),
            {"official_real_train", "train_sim"},
        )


if __name__ == "__main__":
    unittest.main()
