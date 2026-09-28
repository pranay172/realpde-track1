"""Per-window measured-channel scale transfer for Track 1 inference."""

from __future__ import annotations

import numpy as np


def window_channel_stds(
    windows: np.ndarray,
    *,
    measured_channels: int = 2,
    min_std: float = 1e-6,
) -> np.ndarray:
    """Return per-sample scored-element standard deviations for measured channels."""
    if windows.ndim != 5:
        raise ValueError(f"expected (N,T,H,W,C), got {windows.shape}")
    if not 0 < measured_channels <= windows.shape[-1]:
        raise ValueError(f"invalid measured-channel count: {measured_channels}")
    if min_std <= 0:
        raise ValueError("min_std must be positive")
    values = np.asarray(windows[..., :measured_channels], dtype=np.float64)
    scored = values != 0.0
    counts = scored.sum(axis=(1, 2, 3)).astype(np.float64)
    safe = np.maximum(counts, 1.0)
    totals = np.where(scored, values, 0.0).sum(axis=(1, 2, 3))
    means = totals / safe
    centered = values - means[:, None, None, None, :]
    squares = np.where(scored, centered * centered, 0.0).sum(axis=(1, 2, 3))
    stds = np.sqrt(np.maximum(squares / safe, 0.0))
    stds = np.maximum(stds, min_std)
    stds[counts < 2] = min_std
    return stds.astype(np.float32)


def window_scale_ratios(
    windows: np.ndarray,
    reference_std: np.ndarray,
    *,
    measured_channels: int = 2,
    min_std: float = 1e-6,
) -> np.ndarray:
    """Return per-sample scale ratios that map window stds onto ``reference_std``."""
    reference = np.asarray(reference_std[:measured_channels], dtype=np.float32)
    if reference.shape != (measured_channels,) or np.any(reference <= 0):
        raise ValueError(f"invalid reference std: {reference}")
    stds = window_channel_stds(
        windows, measured_channels=measured_channels, min_std=min_std
    )
    ratios = np.ones(windows.shape[:1] + (windows.shape[-1],), dtype=np.float32)
    ratios[:, :measured_channels] = stds / np.maximum(reference, np.float32(min_std))
    return ratios


def apply_window_scale(windows: np.ndarray, ratios: np.ndarray, invert: bool = False) -> np.ndarray:
    """Scale or unscale measured channels; pressure stays zero."""
    if windows.ndim != 5:
        raise ValueError(f"expected (N,T,H,W,C), got {windows.shape}")
    if ratios.shape != (windows.shape[0], windows.shape[-1]):
        raise ValueError(f"ratio shape {ratios.shape} does not match {windows.shape}")
    scale = np.asarray(ratios, dtype=np.float32)
    if invert:
        scale = np.where(scale == 0, 1.0, 1.0 / scale)
    output = np.asarray(windows, dtype=np.float32) * scale[:, None, None, None, :]
    output[..., 2] = 0.0
    return output
