#!/usr/bin/env python3
"""Validate one extracted Track 1 submission inside the evaluator container."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load submission from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "predict", None)):
        raise AttributeError("submission.py must expose a callable predict")
    return module


def prediction_from(result):
    if isinstance(result, dict):
        if "prediction" not in result:
            raise KeyError("dict output is missing prediction")
        if ("lower" in result) != ("upper" in result):
            raise ValueError("bounds must contain both lower and upper")
        return np.asarray(result["prediction"]), sorted(result)
    return np.asarray(result), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=Path("/submission/submission.py"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    input_array = np.load(args.input, allow_pickle=False)
    if input_array.ndim != 5 or tuple(input_array.shape[1:]) != (20, 32, 64, 3):
        raise ValueError(f"unexpected input shape: {input_array.shape}")

    submission = load_submission(args.submission)
    start = time.perf_counter()
    first_raw = submission.predict(input_array, metadata={})
    first_seconds = time.perf_counter() - start
    first, output_keys = prediction_from(first_raw)

    start = time.perf_counter()
    second_raw = submission.predict(input_array, metadata={})
    second_seconds = time.perf_counter() - start
    second, second_keys = prediction_from(second_raw)

    expected_shape = (input_array.shape[0], 20, 32, 64, 3)
    if tuple(first.shape) != expected_shape or tuple(second.shape) != expected_shape:
        raise ValueError(
            f"prediction shapes {first.shape}/{second.shape} != {expected_shape}"
        )
    if first.dtype != np.float32 or second.dtype != np.float32:
        raise TypeError(f"predictions must be float32, got {first.dtype}/{second.dtype}")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise FloatingPointError("prediction contains NaN or Inf")
    if not np.array_equal(first, second):
        raise AssertionError(
            f"repeated predict is not exact; max_abs={np.max(np.abs(first - second))}"
        )
    if output_keys != second_keys:
        raise AssertionError("predict changed its output contract between calls")
    if not np.array_equal(first[..., 2], np.zeros_like(first[..., 2])):
        raise AssertionError("real-flow pressure prediction must be zero")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, first, allow_pickle=False)
    report = {
        "status": "passed",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "submission_path": str(args.submission.resolve()),
        "input_shape": list(input_array.shape),
        "input_dtype": str(input_array.dtype),
        "output_shape": list(first.shape),
        "output_dtype": str(first.dtype),
        "output_keys": output_keys,
        "output_min": float(first.min()),
        "output_max": float(first.max()),
        "pressure_exactly_zero": True,
        "same_process_exact_repeat": True,
        "first_predict_seconds": first_seconds,
        "second_predict_seconds": second_seconds,
        "mean_first_predict_seconds_per_sample": first_seconds / input_array.shape[0],
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_sha256": sha256_array(first),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
