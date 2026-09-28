from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

from realpde_t1.validation import persist_first_frames  # noqa: E402


def load_submission(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E029SubmissionTests(unittest.TestCase):
    def test_e029_constants_and_freezes_k(self) -> None:
        e029 = load_submission(REPO / "submission/e029/submission.py", "e029_test")
        config = json.loads(
            (REPO / "configs" / "e029_full_rollout_curriculum.json").read_text(encoding="utf-8")
        )
        self.assertEqual(e029._BATCH_SIZE, 1)
        self.assertEqual(e029._PERSIST_FIRST_FRAMES, 4)
        self.assertEqual(e029._PERSIST_FIRST_FRAMES, config["candidate"]["persist_first_frames"])
        self.assertEqual(e029._ALPHA, np.float32(0.025))

    def test_e029_persist_first_matches_shared_helper_and_recomputes_bounds(self) -> None:
        e029 = load_submission(REPO / "submission/e029/submission.py", "e029_helper")
        inputs = np.zeros((2, 20, 32, 64, 3), dtype=np.float32)
        inputs[:, -1, 3, 5, 0] = 0.4
        prediction = np.full((2, 20, 32, 64, 3), 0.2, dtype=np.float32)
        prediction[..., 2] = 1.0
        hybrid = e029._persist_first(prediction, inputs)
        expected = persist_first_frames(prediction, inputs, 4)
        np.testing.assert_array_equal(hybrid, expected)
        lower, upper = e029._bounds(hybrid)
        self.assertTrue(np.all(lower <= hybrid))
        self.assertTrue(np.all(hybrid <= upper))
        self.assertTrue(np.all(lower[..., 2] == 0.0))
        self.assertTrue(np.all(upper[..., 2] == 0.0))


if __name__ == "__main__":
    unittest.main()
