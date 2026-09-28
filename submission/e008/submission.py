"""E008 Track 1 submission: batch-invariant E005 CNO with E006 bounds."""

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

# E005 statistics fitted only on the 64 trajectories in the E002 training side.
_MEAN_INPUT = np.array(
    [0.15338869392871857, -0.0004647469031624496, 0.0], dtype=np.float32
)
_STD_INPUT = np.array(
    [0.11143741756677628, 0.015735402703285217, 1.0], dtype=np.float32
)
_MEAN_TARGET = np.array(
    [0.15338662266731262, -0.00046777844545431435, 0.0], dtype=np.float32
)
_STD_TARGET = np.array(
    [0.11143583804368973, 0.015741463750600815, 1.0], dtype=np.float32
)

# E006 interval: half_width = alpha * abs(prediction) + beta * sigma_global.
_ALPHA = np.float32(0.025)
_ADDITIVE_HALF_WIDTH = np.float32(0.15 * 0.0563870259)

# Evaluate every sample independently so its floating-point result cannot depend
# on the evaluator's batch composition. E007 used 4 and failed this invariant.
_BATCH_SIZE = 1

_STATE = {"model": None, "device": None}


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
    return {"prediction": prediction, "lower": lower, "upper": upper}
