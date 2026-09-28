from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

from load_baseline import load_baseline  # noqa: E402
from realpde_t1.adapter import wrap_frozen_cno  # noqa: E402
from realpde_t1.validation import load_config  # noqa: E402


class E020ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (REPO / "configs" / "e020_secondary_fold_residual.json").read_text(encoding="utf-8")
        )
        self.split = load_config(REPO / self.config["split_config"])

    def test_protocol_is_frozen_e010_cno_plus_e005_loss_on_fold_b(self) -> None:
        model = self.config["model"]
        loss = self.config["training"]["loss"]
        evaluation = self.config["evaluation"]
        self.assertEqual(self.config["experiment"], "E020")
        self.assertEqual(self.config["seed"], 20260815)
        self.assertEqual(self.config["split_config"], "configs/e009_secondary_fold.json")
        self.assertEqual(
            self.config["initial_checkpoint"], "artifacts/e010_secondary_fold/final.pth"
        )
        self.assertEqual(self.split["excluded_files"], ["7575_0.h5"])
        self.assertEqual(len(self.split["training_files"]), 62)
        self.assertEqual(len(self.split["validation_files"]), 19)
        self.assertEqual(self.config["training"]["num_updates"], 600)
        self.assertEqual(self.config["training"]["batch_size"], 4)
        self.assertEqual(self.config["training"]["learning_rate"], 0.0003)
        self.assertEqual(loss["name"], "per_window_physical_relative_mse")
        self.assertEqual(model["wrapper"], "residual_cno")
        self.assertEqual(model["base_model_type"], "cno")
        self.assertEqual(model["trainable"], "adapter_only")
        self.assertFalse(model["last_block_unfreeze"])
        self.assertEqual(model["adapter"]["hidden_channels"], 16)
        self.assertEqual(model["adapter"]["n_blocks"], 2)
        self.assertTrue(model["adapter"]["zero_init_last"])
        self.assertEqual(evaluation["fold"], "B")
        self.assertEqual(evaluation["persist_first_frames"], 4)
        self.assertEqual(evaluation["interval"]["alpha"], 0.025)
        self.assertEqual(evaluation["interval"]["beta"], 0.15)
        self.assertEqual(
            evaluation["success_criteria"]["persist_first_4_must_beat_e010"],
            ["rel_l2_score", "mvpe_score"],
        )

    def test_e010_backbone_can_be_loaded_and_wrapped(self) -> None:
        e010_path = REPO / self.config["initial_checkpoint"]
        if not e010_path.is_file():
            self.skipTest(f"E010 checkpoint {e010_path} not found")
        base_model, meta = load_baseline("cno", str(e010_path), device="cpu")
        self.assertEqual(meta["model_type"], "cno")
        self.assertEqual(meta["missing_keys"], [])
        self.assertEqual(meta["unexpected_keys"], [])
        wrapped = wrap_frozen_cno(base_model, self.config["model"]["adapter"])
        trainable = wrapped.trainable_parameters()
        total_trainable = sum(p.numel() for p in trainable)
        self.assertEqual(total_trainable, 10835)
        inputs = torch.randn(1, 20, 32, 64, 3)
        with torch.no_grad():
            out_base = base_model(inputs)
            out_wrapped = wrapped(inputs)
            torch.testing.assert_close(out_wrapped, out_base)


if __name__ == "__main__":
    unittest.main()
