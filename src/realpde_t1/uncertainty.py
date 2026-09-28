"""Uncertainty-interval utilities matching the Track 1 SPS definition."""

from __future__ import annotations

import numpy as np
import torch


def interval_bounds_numpy(
    prediction: np.ndarray,
    alpha: float,
    beta: float,
    sigma_global: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite symmetric bounds and keep pressure bounds exactly zero."""
    if prediction.ndim != 5 or prediction.shape[-1] != 3:
        raise ValueError(f"expected (N,T,H,W,3), got {prediction.shape}")
    if min(alpha, beta, sigma_global) < 0 or sigma_global == 0:
        raise ValueError("alpha/beta must be non-negative and sigma must be positive")
    point = np.asarray(prediction, dtype=np.float32)
    half = np.float32(alpha) * np.abs(point) + np.float32(beta * sigma_global)
    half[..., 2] = 0.0
    lower = point - half
    upper = point + half
    lower[..., 2] = 0.0
    upper[..., 2] = 0.0
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("constructed interval contains non-finite values")
    if np.any(lower > upper):
        raise ValueError("constructed lower bound exceeds upper bound")
    return lower, upper


def candidate_interval_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    accuracy_factors: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    sigma_global: float,
    measured_channels: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate SPS branch numerators for every interval candidate in parallel."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(f"invalid prediction/target shapes: {prediction.shape}/{target.shape}")
    if accuracy_factors.shape != (prediction.shape[0], 3):
        raise ValueError(f"invalid accuracy-factor shape: {accuracy_factors.shape}")
    if alpha.ndim != 1 or beta.shape != alpha.shape:
        raise ValueError("alpha and beta must be equal-length vectors")
    if sigma_global <= 0 or measured_channels <= 0:
        raise ValueError("invalid sigma or measured-channel count")
    point = prediction[..., :measured_channels]
    truth = target[..., :measured_channels]
    candidate_shape = (len(alpha),) + (1,) * point.ndim
    alpha_view = alpha.to(point).view(candidate_shape)
    beta_view = beta.to(point).view(candidate_shape)
    normalized_half = (
        alpha_view * torch.abs(point).unsqueeze(0) / sigma_global + beta_view
    )
    normalized_error = torch.abs(truth - point).unsqueeze(0) / sigma_global
    scored = truth != 0.0
    inside = normalized_error <= normalized_half
    element = torch.where(
        inside & scored.unsqueeze(0), torch.exp(-2.0 * normalized_half), 0.0
    )
    per_sample = torch.sum(element, dim=(2, 3, 4, 5), dtype=torch.float64)
    accuracy = torch.where(
        torch.isfinite(accuracy_factors), accuracy_factors, 0.0
    ).to(dtype=torch.float64)
    branch_sums = per_sample @ accuracy
    coverage_counts = torch.sum(
        inside & scored.unsqueeze(0), dim=(1, 2, 3, 4, 5), dtype=torch.int64
    )
    return branch_sums, coverage_counts, int(torch.count_nonzero(scored))


def finalize_sps(
    branch_sums: np.ndarray,
    coverage_counts: np.ndarray,
    scored_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert accumulated branch sums/counts into SPS and coverage percentages."""
    if branch_sums.ndim != 2 or branch_sums.shape[1] != 3:
        raise ValueError(f"invalid branch-sum shape: {branch_sums.shape}")
    if coverage_counts.shape != (branch_sums.shape[0],) or scored_count <= 0:
        raise ValueError("invalid coverage/scored counts")
    branch = branch_sums / float(scored_count)
    sps = 100.0 * (0.5 * branch[:, 0] + 0.3 * branch[:, 1] + 0.2 * branch[:, 2])
    coverage = coverage_counts.astype(np.float64) / float(scored_count)
    return sps, coverage
