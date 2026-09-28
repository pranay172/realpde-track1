from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from realpde_t1.validation import (  # noqa: E402
    load_config,
    parse_trajectory_name,
    persist_first_frames,
    training_touches_fold,
)


class E012ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.e002 = load_config(REPO / "configs" / "e002_split.json")
        self.fold_b = load_config(REPO / "configs" / "e009_secondary_fold.json")
        self.fold_c = load_config(REPO / "configs" / "e012_fold_c.json")
        self.protocol = json.loads(
            (REPO / "configs" / "e012_multifold_comparison.json").read_text(encoding="utf-8")
        )

    def _re(self, names: list[str]) -> set[int]:
        return {parse_trajectory_name(name)[0] for name in names}

    def test_three_folds_are_re_disjoint_and_cover_public_files(self) -> None:
        a = self._re(self.e002["validation_files"])
        b = self._re(self.fold_b["validation_files"])
        c = self._re(self.fold_c["validation_files"])
        self.assertEqual(a, {3750, 11400, 20325, 26700})
        self.assertEqual(b, {6300, 13950, 21600, 24150})
        self.assertEqual(c, {5025, 16500, 22875, 25425})
        self.assertEqual(a & b, set())
        self.assertEqual(a & c, set())
        self.assertEqual(b & c, set())
        public = set(self.e002["training_files"]) | set(self.e002["validation_files"])
        self.assertEqual(len(self.fold_c["validation_files"]), 17)
        self.assertEqual(len(self.fold_c["training_files"]), 64)
        self.assertEqual(set(self.fold_c["training_files"]) | set(self.fold_c["validation_files"]), public)
        self.assertEqual(self.fold_c["excluded_files"], ["7575_0.h5"])

    def test_contamination_rule_matches_known_checkpoints(self) -> None:
        self.assertFalse(
            training_touches_fold(self.e002["training_files"], self.e002["validation_files"])
        )
        self.assertTrue(
            training_touches_fold(self.e002["training_files"], self.fold_b["validation_files"])
        )
        self.assertTrue(
            training_touches_fold(self.e002["training_files"], self.fold_c["validation_files"])
        )
        self.assertFalse(
            training_touches_fold(self.fold_b["training_files"], self.fold_b["validation_files"])
        )
        self.assertTrue(
            training_touches_fold(self.fold_b["training_files"], self.e002["validation_files"])
        )
        self.assertFalse(training_touches_fold([], self.fold_c["validation_files"]))

    def test_persist_first_overwrites_only_the_requested_prefix(self) -> None:
        inputs = np.zeros((2, 20, 4, 6, 3), dtype=np.float32)
        inputs[:, -1, 0, 0, 0] = 2.0
        prediction = np.ones((2, 20, 4, 6, 3), dtype=np.float32)
        prediction[..., 2] = 3.0
        hybrid = persist_first_frames(prediction, inputs, 4)
        self.assertTrue(np.all(hybrid[:, :4, 0, 0, 0] == 2.0))
        self.assertTrue(np.all(hybrid[:, 4:, 0, 0, 0] == 1.0))
        self.assertTrue(np.all(hybrid[..., 2] == 0.0))
        with self.assertRaisesRegex(ValueError, "invalid persist-first"):
            persist_first_frames(prediction, inputs, 21)

    def test_protocol_freezes_k_and_withholds_fold_c_from_learned_ranking(self) -> None:
        self.assertEqual(self.protocol["methods"]["e005_persist_first_4"]["persist_first_frames"], 4)
        self.assertEqual(self.protocol["methods"]["e010_persist_first_4"]["persist_first_frames"], 4)
        self.assertTrue(self.protocol["ranking"]["use_only_clean_folds"])
        self.assertTrue(self.protocol["ranking"]["do_not_invent_final_score"])
        self.assertEqual(
            self.protocol["success_criteria"]["no_clean_learned_row_on_fold_c"],
            True,
        )


if __name__ == "__main__":
    unittest.main()
