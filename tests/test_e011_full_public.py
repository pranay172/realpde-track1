from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from realpde_t1.validation import audit_split, load_config, parse_trajectory_name  # noqa: E402
from e011_write_submission import render_submission  # noqa: E402


def _write_h5(path: Path, re_value: int, aoa: int) -> None:
    frames = 80
    base = np.arange(frames * 4 * 6, dtype=np.float32).reshape(frames, 4, 6)
    with h5py.File(path, "w") as handle:
        handle["u"] = base
        handle["v"] = -base
        handle["t"] = np.arange(frames, dtype=np.float32)
        handle["re"] = re_value
        handle["aoa"] = aoa


class E011ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.e002 = load_config(REPO / "configs" / "e002_split.json")
        self.e005 = json.loads(
            (REPO / "configs" / "e005_scale_balanced_uv.json").read_text(encoding="utf-8")
        )
        self.e011 = json.loads(
            (REPO / "configs" / "e011_full_public_submission.json").read_text(encoding="utf-8")
        )
        self.split = load_config(REPO / "configs" / "e011_full_public_split.json")

    def test_training_recipe_matches_e005_except_split(self) -> None:
        for key in (
            "optimizer", "learning_rate", "betas", "weight_decay",
            "scheduler", "eta_min", "batch_size", "num_updates",
            "precision", "gradient_clip_norm",
        ):
            self.assertEqual(self.e011["training"][key], self.e005["training"][key], key)
        self.assertEqual(self.e011["training"]["loss"], self.e005["training"]["loss"])
        self.assertEqual(self.e011["seed"], self.e005["seed"])
        self.assertEqual(self.e011["initial_checkpoint"], self.e005["initial_checkpoint"])
        self.assertEqual(self.e011["split_config"], "configs/e011_full_public_split.json")

    def test_split_uses_every_usable_public_file(self) -> None:
        public = set(self.e002["training_files"]) | set(self.e002["validation_files"])
        self.assertEqual(set(self.split["training_files"]), public)
        self.assertEqual(self.split["validation_files"], [])
        self.assertEqual(self.split["excluded_files"], ["7575_0.h5"])
        self.assertEqual(len(self.split["training_files"]), 81)
        self.assertNotIn("7575_0.h5", self.split["training_files"])
        self.assertEqual(
            {parse_trajectory_name(name)[0] for name in self.split["training_files"]},
            {parse_trajectory_name(name)[0] for name in public},
        )

    def test_interval_and_batching_are_frozen_e008_values(self) -> None:
        uncertainty = self.e011["candidate"]["uncertainty"]
        self.assertEqual(uncertainty["alpha"], 0.025)
        self.assertEqual(uncertainty["beta"], 0.15)
        self.assertEqual(uncertainty["sigma_global"], 0.0563870259)
        self.assertEqual(self.e011["candidate"]["point_batch_size"], 1)

    def test_audit_split_allows_empty_validation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_h5(root / "100_0.h5", 101, 0)
            _write_h5(root / "7575_0.h5", 7585, 0)
            config_path = root / "split.json"
            config_path.write_text(json.dumps({
                "experiment": "E011",
                "expected_h5_count": 2,
                "excluded_files": ["7575_0.h5"],
                "validation_groups": {},
                "training_files": ["100_0.h5"],
                "validation_files": [],
                "window": {
                    "input_frames": 20, "output_frames": 20, "temporal_stride": 40,
                    "start_offset": 0, "spatial_stride": 2, "raw_height": 4,
                    "raw_width": 6, "evaluation_height": 2, "evaluation_width": 3,
                },
            }), encoding="utf-8")
            manifest = audit_split(root, load_config(config_path))
            self.assertEqual(manifest["counts"]["train"], 1)
            self.assertEqual(manifest["counts"]["validation"], 0)
            self.assertEqual(manifest["counts"]["validation_windows"], 0)

    def test_wrapper_renderer_embeds_normalizer_and_frozen_interval(self) -> None:
        normalizer = {
            "mean_input": [0.1, -0.2, 0.0],
            "std_input": [1.5, 0.25, 1.0],
            "mean_target": [0.3, 0.4, 0.0],
            "std_target": [2.0, 0.5, 1.0],
        }
        source = render_submission(normalizer, self.e011["candidate"]["uncertainty"])
        namespace: dict[str, object] = {"__file__": str(REPO / "submission" / "e011" / "submission.py")}
        exec(compile(source, "<e011-wrapper>", "exec"), namespace)
        np.testing.assert_array_equal(
            np.asarray(namespace["_MEAN_INPUT"], dtype=np.float32),
            np.asarray(normalizer["mean_input"], dtype=np.float32),
        )
        self.assertEqual(namespace["_BATCH_SIZE"], 1)
        self.assertEqual(namespace["_ALPHA"], np.float32(0.025))


if __name__ == "__main__":
    unittest.main()
