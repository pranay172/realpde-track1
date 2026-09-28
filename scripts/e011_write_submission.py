#!/usr/bin/env python3
"""Write the E011 wrapper from the trained all-public-data normalizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZER = REPO / "artifacts" / "e011_full_public" / "normalizer.json"
DEFAULT_OUTPUT = REPO / "submission" / "e011" / "submission.py"
DEFAULT_CONFIG = REPO / "configs" / "e011_full_public_submission.json"


TEMPLATE = '''"""E011 Track 1 submission: all-public E005-recipe CNO with frozen E008 bounds."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_CHECKPOINT = os.path.join(_HERE, "model.pth")
_EXPECTED_SAMPLE_SHAPE = (20, 32, 64, 3)

# Statistics fitted only on the 81 usable public real trajectories.
_MEAN_INPUT = np.array(
    {mean_input}, dtype=np.float32
)
_STD_INPUT = np.array(
    {std_input}, dtype=np.float32
)
_MEAN_TARGET = np.array(
    {mean_target}, dtype=np.float32
)
_STD_TARGET = np.array(
    {std_target}, dtype=np.float32
)

# Frozen E006/E008 interval: half_width = alpha * abs(prediction) + beta * sigma_global.
_ALPHA = np.float32({alpha})
_ADDITIVE_HALF_WIDTH = np.float32({beta} * {sigma})

# Evaluate every sample independently so its floating-point result cannot depend
# on the evaluator's batch composition.
_BATCH_SIZE = 1

_STATE = {{"model": None, "device": None}}


def _get_model():
    if _STATE["model"] is None:
        from load_baseline import load_baseline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _metadata = load_baseline("cno", _CHECKPOINT, device=device)
        model.eval()
        _STATE["model"] = model
        _STATE["device"] = device
    return _STATE["model"], _STATE["device"]


def _validate_input(input_array) -> np.ndarray:
    value = np.asarray(input_array, dtype=np.float32)
    if value.ndim != 5 or tuple(value.shape[1:]) != _EXPECTED_SAMPLE_SHAPE:
        raise ValueError(
            "expected input shape (N,20,32,64,3), got %s" % (value.shape,)
        )
    if value.shape[0] == 0:
        raise ValueError("input batch must contain at least one sample")
    if not np.all(np.isfinite(value)):
        raise ValueError("input contains NaN or Inf")
    return value


def _bounds(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half_width = _ALPHA * np.abs(prediction) + _ADDITIVE_HALF_WIDTH
    half_width[..., 2] = 0.0
    lower = prediction - half_width
    upper = prediction + half_width
    lower[..., 2] = 0.0
    upper[..., 2] = 0.0
    return lower.astype(np.float32, copy=False), upper.astype(np.float32, copy=False)


def predict(input_array, metadata=None):
    """Forecast 20 frames and return calibrated per-element bounds."""
    del metadata
    value = _validate_input(input_array)
    model, device = _get_model()

    mean_input = torch.as_tensor(_MEAN_INPUT, device=device)
    std_input = torch.as_tensor(_STD_INPUT, device=device)
    mean_target = torch.as_tensor(_MEAN_TARGET, device=device)
    std_target = torch.as_tensor(_STD_TARGET, device=device)

    outputs = []
    with torch.inference_mode():
        for start in range(0, value.shape[0], _BATCH_SIZE):
            batch = torch.from_numpy(
                np.ascontiguousarray(value[start:start + _BATCH_SIZE])
            ).to(device)
            batch = (batch - mean_input) / std_input
            result = model(batch)
            result = result * std_target + mean_target
            outputs.append(result.float().cpu().numpy())

    prediction = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    prediction[..., 2] = 0.0
    lower, upper = _bounds(prediction)
    return {{"prediction": prediction, "lower": lower, "upper": upper}}
'''


def format_vector(values: list[float]) -> str:
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def render_submission(normalizer: dict, uncertainty: dict) -> str:
    return TEMPLATE.format(
        mean_input=format_vector(normalizer["mean_input"]),
        std_input=format_vector(normalizer["std_input"]),
        mean_target=format_vector(normalizer["mean_target"]),
        std_target=format_vector(normalizer["std_target"]),
        alpha=repr(float(uncertainty["alpha"])),
        beta=repr(float(uncertainty["beta"])),
        sigma=repr(float(uncertainty["sigma_global"])),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    normalizer = json.loads(args.normalizer.read_text(encoding="utf-8"))
    if args.output.exists():
        raise FileExistsError(f"refusing to replace E011 wrapper: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_submission(normalizer, config["candidate"]["uncertainty"]),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
