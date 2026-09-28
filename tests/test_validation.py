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
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

from realpde_t1.validation import (  # noqa: E402
    audit_split,
    build_validation_arrays,
    load_config,
    parse_trajectory_name,
    score_prediction,
    window_starts,
)
import scoring  # noqa: E402


def _write_h5(path: Path, re_value: int, aoa: int) -> None:
    frames = 80
    base = np.arange(frames * 4 * 6, dtype=np.float32).reshape(frames, 4, 6)
    with h5py.File(path, "w") as handle:
        handle["u"] = base
        handle["v"] = -base
        handle["t"] = np.arange(frames, dtype=np.float32)
        handle["re"] = re_value
        handle["aoa"] = aoa


class ValidationTests(unittest.TestCase):
    def test_window_policy_and_filename_parser(self) -> None:
        window = {
            "input_frames": 20, "output_frames": 20,
            "temporal_stride": 40, "start_offset": 0,
        }
        self.assertEqual(window_starts(80, window), [0, 40])
        self.assertEqual(window_starts(79, window), [0])
        self.assertEqual(parse_trajectory_name("11400_15.h5"), (11400, 15))
        with self.assertRaisesRegex(ValueError, "prevent overlap"):
            window_starts(80, {**window, "temporal_stride": 20})

    def test_split_audit_and_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            tmp_path = Path(name)
            _write_h5(tmp_path / "100_0.h5", 101, 0)
            _write_h5(tmp_path / "200_0.h5", 202, 0)
            _write_h5(tmp_path / "7575_0.h5", 7585, 0)
            config_path = tmp_path / "split.json"
            config_path.write_text(json.dumps({
                "experiment": "E002",
                "expected_h5_count": 3,
                "excluded_files": ["7575_0.h5"],
                "validation_groups": {"id_interpolation": [200]},
                "training_files": ["100_0.h5"],
                "validation_files": ["200_0.h5"],
                "window": {
                    "input_frames": 20, "output_frames": 20, "temporal_stride": 40,
                    "start_offset": 0, "spatial_stride": 2, "raw_height": 4,
                    "raw_width": 6, "evaluation_height": 2, "evaluation_width": 3,
                },
            }), encoding="utf-8")
            config = load_config(config_path)
            manifest = audit_split(tmp_path, config)
            self.assertEqual(manifest["counts"], {
                "all": 3, "train": 1, "validation": 1,
                "excluded": 1, "validation_windows": 2,
            })
            input_path, target_path, rows = build_validation_arrays(
                tmp_path, manifest, tmp_path / "arrays"
            )
            inputs = np.load(input_path, mmap_mode="r")
            targets = np.load(target_path, mmap_mode="r")
            self.assertEqual(inputs.shape, (2, 20, 2, 3, 3))
            self.assertEqual(targets.shape, inputs.shape)
            self.assertTrue(np.all(inputs[..., 2] == 0.0))
            self.assertTrue(np.all(targets[..., 2] == 0.0))
            self.assertEqual(rows[1]["start"], 40)

    def test_score_helper_matches_official_core(self) -> None:
        rng = np.random.default_rng(7)
        target = rng.normal(size=(2, 20, 32, 64, 3)).astype(np.float32)
        target[..., 2] = 0.0
        prediction = target * np.float32(0.9)
        result = score_prediction(prediction, target, 0.1, scoring)
        channels = scoring.measured_channels(target)
        expected_rel = float(np.mean(scoring.rel_l2_per_sample(prediction, target, channels)))
        self.assertAlmostEqual(result["raw"]["relative_l2"], expected_rel)
        self.assertAlmostEqual(
            result["scores"]["rel_l2_score"], scoring.score_error(expected_rel)
        )


if __name__ == "__main__":
    unittest.main()
