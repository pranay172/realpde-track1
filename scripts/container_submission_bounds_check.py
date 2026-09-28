#!/usr/bin/env python3
"""Validate one extracted Track 1 bounded submission in an isolated process."""

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


REQUIRED_KEYS = {"lower", "prediction", "upper"}


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


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def validate_result(raw, expected_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict) or set(raw) != REQUIRED_KEYS:
        keys = sorted(raw) if isinstance(raw, dict) else None
        raise TypeError(f"predict must return exactly {sorted(REQUIRED_KEYS)}, got {keys}")
    result = {key: np.asarray(raw[key]) for key in REQUIRED_KEYS}
    for key, value in result.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{key} shape {value.shape} != {expected_shape}")
        if value.dtype != np.float32:
            raise TypeError(f"{key} must be float32, got {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"{key} contains NaN or Inf")
        if not np.array_equal(value[..., 2], np.zeros_like(value[..., 2])):
            raise AssertionError(f"{key} pressure channel is not exactly zero")
    if np.any(result["lower"] > result["prediction"]):
        raise AssertionError("lower bound exceeds prediction")
    if np.any(result["prediction"] > result["upper"]):
        raise AssertionError("prediction exceeds upper bound")
    return result


def main() -> None:
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=Path("/submission/submission.py"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    input_array = np.load(args.input, allow_pickle=False)
    if input_array.dtype != np.float32:
        raise TypeError(f"input must be float32, got {input_array.dtype}")
    if input_array.ndim != 5 or tuple(input_array.shape[1:]) != (20, 32, 64, 3):
        raise ValueError(f"unexpected input shape: {input_array.shape}")
    expected_shape = tuple(input_array.shape)

    import_start = time.perf_counter()
    submission = load_submission(args.submission)
    import_seconds = time.perf_counter() - import_start

    synchronize()
    start = time.perf_counter()
    first = validate_result(
        submission.predict(input_array, metadata={"contract_check": True}), expected_shape
    )
    synchronize()
    first_seconds = time.perf_counter() - start

    synchronize()
    start = time.perf_counter()
    second = validate_result(
        submission.predict(input_array, metadata={"contract_check": True}), expected_shape
    )
    synchronize()
    second_seconds = time.perf_counter() - start

    repeat_max_abs = {}
    for key in sorted(REQUIRED_KEYS):
        repeat_max_abs[key] = float(np.max(np.abs(first[key] - second[key])))
        if not np.array_equal(first[key], second[key]):
            raise AssertionError(
                f"repeated {key} is not exact; max_abs={repeat_max_abs[key]}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **first)
    report = {
        "status": "passed",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "submission_path": str(args.submission.resolve()),
        "input_shape": list(input_array.shape),
        "input_dtype": str(input_array.dtype),
        "output_keys": sorted(first),
        "output": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
                "sha256": sha256_array(value),
                "pressure_exactly_zero": True,
            }
            for key, value in sorted(first.items())
        },
        "bounds_enclose_prediction": True,
        "same_process_exact_repeat": True,
        "same_process_max_abs": repeat_max_abs,
        "import_seconds": import_seconds,
        "first_predict_seconds": first_seconds,
        "second_predict_seconds": second_seconds,
        "mean_second_predict_seconds_per_sample": second_seconds / input_array.shape[0],
        "whole_runner_seconds": time.perf_counter() - process_start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
