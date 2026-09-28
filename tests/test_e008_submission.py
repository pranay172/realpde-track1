from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_submission(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e008_only_changes_inference_batching_constants() -> None:
    e007 = load_submission(REPO / "submission/e007/submission.py", "e007_compare")
    e008 = load_submission(REPO / "submission/e008/submission.py", "e008_compare")
    assert e007._BATCH_SIZE == 4
    assert e008._BATCH_SIZE == 1
    for name in (
        "_EXPECTED_SAMPLE_SHAPE",
        "_MEAN_INPUT",
        "_STD_INPUT",
        "_MEAN_TARGET",
        "_STD_TARGET",
        "_ALPHA",
        "_ADDITIVE_HALF_WIDTH",
    ):
        np.testing.assert_array_equal(np.asarray(getattr(e008, name)), np.asarray(getattr(e007, name)))


def test_e008_bounds_are_identical_to_e007() -> None:
    e007 = load_submission(REPO / "submission/e007/submission.py", "e007_bounds")
    e008 = load_submission(REPO / "submission/e008/submission.py", "e008_bounds")
    prediction = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(1, 1, 2, 4, 3)
    prediction[..., 2] = 0.0
    for actual, expected in zip(e008._bounds(prediction), e007._bounds(prediction)):
        np.testing.assert_array_equal(actual, expected)
