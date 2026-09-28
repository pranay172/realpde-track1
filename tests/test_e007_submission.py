from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SUBMISSION = REPO / "submission" / "e007" / "submission.py"


def load_submission():
    spec = importlib.util.spec_from_file_location("e007_submission", SUBMISSION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestE007Submission(unittest.TestCase):
    def test_bounds_match_e006_policy_and_zero_pressure(self) -> None:
        module = load_submission()
        prediction = np.array([[[[[-2.0, 1.0, 0.0]]]]], dtype=np.float32)
        lower, upper = module._bounds(prediction)
        floor = np.float32(0.15 * 0.0563870259)
        expected_half = np.array([[[[[0.05 + floor, 0.025 + floor, 0.0]]]]], dtype=np.float32)
        np.testing.assert_array_equal(lower, prediction - expected_half)
        np.testing.assert_array_equal(upper, prediction + expected_half)
        self.assertEqual(lower.dtype, np.float32)
        self.assertEqual(upper.dtype, np.float32)
        self.assertTrue(np.all(lower <= prediction))
        self.assertTrue(np.all(prediction <= upper))

    def test_input_contract_rejects_bad_inputs(self) -> None:
        module = load_submission()
        good = np.zeros((1, 20, 32, 64, 3), dtype=np.float64)
        self.assertEqual(module._validate_input(good).dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "expected input shape"):
            module._validate_input(np.zeros((1, 20, 64, 32, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "at least one"):
            module._validate_input(np.zeros((0, 20, 32, 64, 3), dtype=np.float32))
        good[0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            module._validate_input(good)


if __name__ == "__main__":
    unittest.main()
