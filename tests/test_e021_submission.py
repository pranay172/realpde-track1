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


class E021SubmissionTests(unittest.TestCase):
    def test_e021_keeps_e013_constants_and_freezes_k(self) -> None:
        e013 = load_submission(REPO / "submission/e013/submission.py", "e013_for_e021")
        e021 = load_submission(REPO / "submission/e021/submission.py", "e021_compare")
        config = json.loads(
            (REPO / "configs" / "e021_residual_cno_submission.json").read_text(encoding="utf-8")
        )
        self.assertEqual(e021._BATCH_SIZE, 1)
        self.assertEqual(e021._PERSIST_FIRST_FRAMES, 4)
        self.assertEqual(e021._PERSIST_FIRST_FRAMES, config["candidate"]["persist_first_frames"])
        for name in (
            "_EXPECTED_SAMPLE_SHAPE",
            "_MEAN_INPUT",
            "_STD_INPUT",
            "_MEAN_TARGET",
            "_STD_TARGET",
            "_ALPHA",
            "_ADDITIVE_HALF_WIDTH",
            "_BATCH_SIZE",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(e021, name)), np.asarray(getattr(e013, name))
            )

    def test_e021_persist_first_matches_shared_helper_and_recomputes_bounds(self) -> None:
        e021 = load_submission(REPO / "submission/e021/submission.py", "e021_helper")
        inputs = np.zeros((2, 20, 32, 64, 3), dtype=np.float32)
        inputs[:, -1, 3, 5, 0] = 0.4
        prediction = np.full((2, 20, 32, 64, 3), 0.2, dtype=np.float32)
        prediction[..., 2] = 1.0
        hybrid = e021._persist_first(prediction, inputs)
        expected = persist_first_frames(prediction, inputs, 4)
        np.testing.assert_array_equal(hybrid, expected)
        lower, upper = e021._bounds(hybrid)
        np.testing.assert_array_equal(
            lower[:, :4, 3, 5, 0],
            hybrid[:, :4, 3, 5, 0]
            - (e021._ALPHA * np.abs(hybrid[:, :4, 3, 5, 0]) + e021._ADDITIVE_HALF_WIDTH),
        )
        self.assertTrue(np.all(lower[..., 2] == 0.0))
        self.assertTrue(np.all(upper[..., 2] == 0.0))
        self.assertTrue(np.all(lower <= hybrid))
        self.assertTrue(np.all(hybrid <= upper))

    def test_e021_model_loading_and_forward(self) -> None:
        e021 = load_submission(REPO / "submission/e021/submission.py", "e021_load_test")
        ckpt_path = REPO / "artifacts/e019_residual_adapter_e005/final.pth"
        if not ckpt_path.is_file():
            self.skipTest("E019 checkpoint not found")
        model = e021._load_residual_cno(str(ckpt_path), torch.device("cpu")).eval()
        inputs = torch.randn(1, 20, 32, 64, 3)
        with torch.no_grad():
            out = model(inputs)
        self.assertEqual(out.shape, (1, 20, 32, 64, 3))
        self.assertTrue(torch.all(torch.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
