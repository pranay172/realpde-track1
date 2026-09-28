"""Guard costly experiments against accidental split/control/gate changes."""
import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO / "scripts"))
from evaluate_clean_candidate import assess


def config(name):
    return json.loads((REPO / "configs" / name).read_text())


class Score80ProtocolTests(unittest.TestCase):
    def test_e038_changes_only_budget_on_fold_b(self):
        old = config("e010_secondary_fold_robustness.json")
        new = config("e038_fold_b_long_backbone.json")
        for key in ("seed","split_config","initial_checkpoint","data"):
            self.assertEqual(old[key],new[key])
        expected = dict(old["training"],num_updates=3000,maximum_wall_minutes=260)
        self.assertEqual(expected,new["training"])

    def test_e040_restarts_with_only_budget_change(self):
        old = config("e034_long_backbone.json")
        new = config("e040_fold_a_6000_backbone.json")
        for key in ("seed","split_config","initial_checkpoint","data"):
            self.assertEqual(old[key],new[key])
        self.assertEqual(dict(old["training"],num_updates=6000,maximum_wall_minutes=520),new["training"])

    def test_clean_adapters_preserve_rollout_recipe_and_match_parent_split(self):
        old = config("e027_residual_rollout_curriculum.json")
        for fold,parent in (("a","e034_long_backbone.json"),("b","e038_fold_b_long_backbone.json")):
            new = config(f"e039{fold}_long_backbone_rollout.json")
            self.assertEqual(new["training"],old["training"])
            self.assertEqual(new["model"],old["model"])
            self.assertEqual(new["split_config"],config(parent)["split_config"])
            self.assertTrue(new["require_clean_backbone_provenance"])

    def test_slice_regression_blocks_aggregate_win(self):
        ev = {"minimum_deltas":{"rel_l2_score":0},"slice_max_regression":{"rel_l2_score":.25}}
        reference = {"rel_l2_score":94.0}
        baseline = {"6300":{"calibrated":{"scores":reference}}}
        candidate = {"6300":{"calibrated":{"scores":{"rel_l2_score":93.0}}}}
        result = assess({"rel_l2_score":95},reference,candidate,baseline,ev)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["Re6300_rel_l2_score"]["passed"])
