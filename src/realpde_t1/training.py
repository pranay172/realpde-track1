"""Train-only window loading and Gaussian statistics for Track 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def make_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str,
    learning_rate: float,
    betas: Sequence[float],
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build the named optimizer. Only Adam and AdamW are supported."""
    options = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in options:
        raise ValueError(f"unsupported optimizer: {name}")
    return options[name](
        parameters,
        lr=float(learning_rate),
        betas=tuple(float(value) for value in betas),
        weight_decay=float(weight_decay),
    )


def _field(handle: h5py.File, name: str) -> h5py.Dataset:
    if name in handle:
        return handle[name]
    if "measured_data" in handle and name in handle["measured_data"]:
        return handle["measured_data"][name]
    raise KeyError(f"missing field {name!r} in {handle.filename}")


@dataclass(frozen=True)
class WindowGeometry:
    input_frames: int = 20
    output_frames: int = 20
    temporal_stride: int = 20
    spatial_stride: int = 2

    @property
    def horizon(self) -> int:
        return self.input_frames + self.output_frames


def enumerate_windows(
    data_root: Path,
    files: Sequence[str],
    geometry: WindowGeometry,
) -> list[tuple[str, int]]:
    """Enumerate complete windows from only the explicitly supplied files."""
    entries: list[tuple[str, int]] = []
    for name in files:
        path = data_root / name
        with h5py.File(path, "r") as handle:
            u = _field(handle, "u")
            v = _field(handle, "v")
            if u.shape != v.shape or u.ndim != 3:
                raise ValueError(f"invalid fields in {path}: {u.shape}/{v.shape}")
            n_frames = int(u.shape[0])
        entries.extend(
            (name, start)
            for start in range(0, n_frames - geometry.horizon + 1, geometry.temporal_stride)
        )
    if not entries:
        raise ValueError("no complete training windows were found")
    return entries


class RealWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy real-PIV window loader with explicit immutable file membership."""

    def __init__(
        self,
        data_root: Path,
        files: Sequence[str],
        geometry: WindowGeometry,
        preload: bool = False,
    ) -> None:
        self.data_root = data_root
        self.files = tuple(files)
        self.geometry = geometry
        self.entries = enumerate_windows(data_root, self.files, geometry)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if preload:
            for name in self.files:
                self._cache[name] = self._load_fields(name)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_fields(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        spatial = self.geometry.spatial_stride
        with h5py.File(self.data_root / name, "r") as handle:
            u = np.asarray(_field(handle, "u")[:, ::spatial, ::spatial], dtype=np.float32)
            v = np.asarray(_field(handle, "v")[:, ::spatial, ::spatial], dtype=np.float32)
        return u, v

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name, start = self.entries[index]
        stop = start + self.geometry.horizon
        if name in self._cache:
            full_u, full_v = self._cache[name]
            u, v = full_u[start:stop], full_v[start:stop]
        else:
            u, v = self._load_fields(name)
            u, v = u[start:stop], v[start:stop]
        data = np.stack((u, v, np.zeros_like(u)), axis=-1)
        middle = self.geometry.input_frames
        input_array = np.ascontiguousarray(data[:middle])
        target_array = np.ascontiguousarray(data[middle:])
        return torch.from_numpy(input_array), torch.from_numpy(target_array)


@dataclass(frozen=True)
class GaussianStats:
    mean_input: np.ndarray
    std_input: np.ndarray
    mean_target: np.ndarray
    std_target: np.ndarray
    input_elements_per_channel: int
    target_elements_per_channel: int

    def as_json(self) -> dict:
        return {
            "mean_input": self.mean_input.tolist(),
            "std_input": self.std_input.tolist(),
            "mean_target": self.mean_target.tolist(),
            "std_target": self.std_target.tolist(),
            "input_elements_per_channel": self.input_elements_per_channel,
            "target_elements_per_channel": self.target_elements_per_channel,
        }


def _weighted_moments(
    data: np.ndarray,
    frame_weights: np.ndarray,
) -> tuple[float, float, int]:
    if data.shape[0] != frame_weights.shape[0]:
        raise ValueError("frame-weight length mismatch")
    active = frame_weights > 0
    selected = np.asarray(data[active], dtype=np.float64)
    weights = frame_weights[active].astype(np.float64, copy=False)
    weighted = weights[:, None, None]
    total = float(np.sum(selected * weighted, dtype=np.float64))
    square = float(np.sum(selected * selected * weighted, dtype=np.float64))
    count = int(np.sum(weights, dtype=np.float64) * data.shape[1] * data.shape[2])
    return total, square, count


def compute_gaussian_stats(
    data_root: Path,
    files: Sequence[str],
    geometry: WindowGeometry,
) -> GaussianStats:
    """Compute exact global moments over the explicit training-window elements."""
    input_sum = np.zeros(3, dtype=np.float64)
    input_square = np.zeros(3, dtype=np.float64)
    target_sum = np.zeros(3, dtype=np.float64)
    target_square = np.zeros(3, dtype=np.float64)
    input_count = 0
    target_count = 0

    entries = enumerate_windows(data_root, files, geometry)
    starts_by_file: dict[str, list[int]] = {name: [] for name in files}
    for name, start in entries:
        starts_by_file[name].append(start)

    for name in files:
        path = data_root / name
        spatial = geometry.spatial_stride
        with h5py.File(path, "r") as handle:
            u = np.asarray(_field(handle, "u")[:, ::spatial, ::spatial], dtype=np.float32)
            v = np.asarray(_field(handle, "v")[:, ::spatial, ::spatial], dtype=np.float32)
        input_weights = np.zeros(u.shape[0], dtype=np.int64)
        target_weights = np.zeros(u.shape[0], dtype=np.int64)
        for start in starts_by_file[name]:
            middle = start + geometry.input_frames
            stop = middle + geometry.output_frames
            input_weights[start:middle] += 1
            target_weights[middle:stop] += 1
        for channel, values in enumerate((u, v)):
            total, square, count = _weighted_moments(values, input_weights)
            input_sum[channel] += total
            input_square[channel] += square
            if channel == 0:
                input_count += count
            total, square, count = _weighted_moments(values, target_weights)
            target_sum[channel] += total
            target_square[channel] += square
            if channel == 0:
                target_count += count

    def finish(total: np.ndarray, square: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        mean = total / count
        variance = np.maximum(square / count - mean * mean, 0.0)
        std = np.sqrt(variance)
        mean[2] = 0.0
        std[2] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    mean_input, std_input = finish(input_sum, input_square, input_count)
    mean_target, std_target = finish(target_sum, target_square, target_count)
    return GaussianStats(
        mean_input=mean_input,
        std_input=std_input,
        mean_target=mean_target,
        std_target=std_target,
        input_elements_per_channel=input_count,
        target_elements_per_channel=target_count,
    )


def scale_balanced_uv_relative_mse(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Mean per-window physical relative squared error over measured channels."""
    if normalized_prediction.shape != normalized_target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: "
            f"{normalized_prediction.shape}/{normalized_target.shape}"
        )
    if normalized_prediction.ndim != 5:
        raise ValueError("expected tensors with (N,T,H,W,C) layout")
    if not 0 < measured_channels <= normalized_prediction.shape[-1]:
        raise ValueError(f"invalid measured-channel count: {measured_channels}")
    if denominator_epsilon <= 0:
        raise ValueError("denominator epsilon must be positive")
    prediction = (
        normalized_prediction[..., :measured_channels]
        * std_target[:measured_channels]
        + mean_target[:measured_channels]
    )
    target = (
        normalized_target[..., :measured_channels]
        * std_target[:measured_channels]
        + mean_target[:measured_channels]
    )
    reduction_axes = tuple(range(1, prediction.ndim))
    numerator = torch.sum((prediction - target) ** 2, dim=reduction_axes)
    denominator = torch.sum(target ** 2, dim=reduction_axes).clamp_min(
        denominator_epsilon
    )
    return torch.mean(numerator / denominator)


def _physical_measured(
    normalized: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    measured_channels: int,
) -> torch.Tensor:
    return (
        normalized[..., :measured_channels]
        * std_target[:measured_channels]
        + mean_target[:measured_channels]
    )


def _validate_normalized_pair(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    measured_channels: int,
    denominator_epsilon: float,
) -> None:
    if normalized_prediction.shape != normalized_target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: "
            f"{normalized_prediction.shape}/{normalized_target.shape}"
        )
    if normalized_prediction.ndim != 5:
        raise ValueError("expected tensors with (N,T,H,W,C) layout")
    if not 0 < measured_channels <= normalized_prediction.shape[-1]:
        raise ValueError(f"invalid measured-channel count: {measured_channels}")
    if denominator_epsilon <= 0:
        raise ValueError("denominator epsilon must be positive")


def _per_window_relative_squared(
    prediction: torch.Tensor,
    target: torch.Tensor,
    denominator_epsilon: float,
) -> torch.Tensor:
    reduction_axes = tuple(range(1, prediction.ndim))
    numerator = torch.sum((prediction - target) ** 2, dim=reduction_axes)
    denominator = torch.sum(target ** 2, dim=reduction_axes).clamp_min(denominator_epsilon)
    return numerator / denominator


def temporal_mean_relative_mse(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Mean per-window relative squared error of the temporal-mean u/v field."""
    _validate_normalized_pair(
        normalized_prediction, normalized_target, measured_channels, denominator_epsilon
    )
    prediction = _physical_measured(
        normalized_prediction, mean_target, std_target, measured_channels
    )
    target = _physical_measured(
        normalized_target, mean_target, std_target, measured_channels
    )
    return torch.mean(
        _per_window_relative_squared(
            prediction.mean(dim=1), target.mean(dim=1), denominator_epsilon
        )
    )


def fluctuation_relative_mse(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Mean per-window relative squared error of the zero-mean u/v fluctuation."""
    _validate_normalized_pair(
        normalized_prediction, normalized_target, measured_channels, denominator_epsilon
    )
    prediction = _physical_measured(
        normalized_prediction, mean_target, std_target, measured_channels
    )
    target = _physical_measured(
        normalized_target, mean_target, std_target, measured_channels
    )
    return torch.mean(
        _per_window_relative_squared(
            prediction - prediction.mean(dim=1, keepdim=True),
            target - target.mean(dim=1, keepdim=True),
            denominator_epsilon,
        )
    )


def official_tke_maps(physical_uv: torch.Tensor) -> torch.Tensor:
    """Official per-location TKE: 0.5 (Var_t u + Var_t v), shape (N,H,W)."""
    if physical_uv.ndim != 5 or physical_uv.shape[-1] < 2:
        raise ValueError("expected physical (N,T,H,W,C>=2) velocity tensors")
    u = physical_uv[..., 0]
    v = physical_uv[..., 1]
    u_var = torch.mean((u - u.mean(dim=1, keepdim=True)) ** 2, dim=1)
    v_var = torch.mean((v - v.mean(dim=1, keepdim=True)) ** 2, dim=1)
    return 0.5 * (u_var + v_var)


def tke_relative_l2(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Mean per-window official TKE relative L2 over measured u/v."""
    _validate_normalized_pair(
        normalized_prediction, normalized_target, measured_channels, denominator_epsilon
    )
    if measured_channels < 2:
        raise ValueError("TKE requires both measured velocity channels")
    prediction = _physical_measured(
        normalized_prediction, mean_target, std_target, measured_channels
    )
    target = _physical_measured(
        normalized_target, mean_target, std_target, measured_channels
    )
    pred_tke = official_tke_maps(prediction).flatten(1)
    target_tke = official_tke_maps(target).flatten(1)
    numerator = torch.linalg.vector_norm(pred_tke - target_tke, dim=1)
    denominator = torch.linalg.vector_norm(target_tke, dim=1).clamp_min(denominator_epsilon)
    return torch.mean(numerator / denominator)


def split_mean_fluct_tke_loss(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
    tke_denominator_epsilon: float = 1e-8,
    lambda_mean: float = 0.3,
    lambda_fluct: float = 0.3,
    lambda_tke: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """E005 field loss plus mean, fluctuation, and official TKE terms."""
    if min(lambda_mean, lambda_fluct, lambda_tke) < 0:
        raise ValueError("split-loss weights must be non-negative")
    field = scale_balanced_uv_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    mean = temporal_mean_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    fluct = fluctuation_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    tke = tke_relative_l2(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=tke_denominator_epsilon,
    )
    total = field + lambda_mean * mean + lambda_fluct * fluct + lambda_tke * tke
    return total, {"field": field, "mean": mean, "fluct": fluct, "tke": tke}


def _central_diff(field: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == -1:
        padded = torch.cat((field[..., :1], field, field[..., -1:]), dim=-1)
        return 0.5 * (padded[..., 2:] - padded[..., :-2])
    if dim == -2:
        padded = torch.cat((field[..., :1, :], field, field[..., -1:, :]), dim=-2)
        return 0.5 * (padded[..., 2:, :] - padded[..., :-2, :])
    raise ValueError(f"central differences are only defined on spatial dims, got {dim}")


def vorticity_index_grid(physical_uv: torch.Tensor) -> torch.Tensor:
    """Index-space vorticity ∂v/∂x − ∂u/∂y on a (N,T,H,W,C>=2) field."""
    if physical_uv.ndim != 5 or physical_uv.shape[-1] < 2:
        raise ValueError("expected physical (N,T,H,W,C>=2) velocity tensors")
    dv_dx = _central_diff(physical_uv[..., 1], dim=-1)
    du_dy = _central_diff(physical_uv[..., 0], dim=-2)
    return dv_dx - du_dy


def vorticity_dt_weights(
    physical_uv: torch.Tensor,
    *,
    alpha: float,
    weight_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Per-window weights 1 + α |∂t ω| / mean(|∂t ω|), shape (N,T,H,W)."""
    if alpha < 0:
        raise ValueError("wake-weight alpha must be non-negative")
    if weight_epsilon <= 0:
        raise ValueError("weight epsilon must be positive")
    omega = vorticity_index_grid(physical_uv)
    delta = torch.zeros_like(omega)
    if omega.shape[1] > 1:
        delta[:, 1:] = omega[:, 1:] - omega[:, :-1]
        delta[:, 0] = delta[:, 1]
    score = torch.abs(delta)
    scale = torch.mean(score, dim=(1, 2, 3), keepdim=True).clamp_min(weight_epsilon)
    return 1.0 + alpha * (score / scale)


def wake_weighted_relative_mse(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
    weight_epsilon: float = 1e-12,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """E005 relative u/v loss with target vorticity-change space-time weights."""
    _validate_normalized_pair(
        normalized_prediction, normalized_target, measured_channels, denominator_epsilon
    )
    if measured_channels < 2:
        raise ValueError("vorticity weights require both measured velocity channels")
    prediction = _physical_measured(
        normalized_prediction, mean_target, std_target, measured_channels
    )
    target = _physical_measured(
        normalized_target, mean_target, std_target, measured_channels
    )
    with torch.no_grad():
        weights = vorticity_dt_weights(
            target, alpha=alpha, weight_epsilon=weight_epsilon
        )
    weight = weights.unsqueeze(-1)
    reduction = (1, 2, 3, 4)
    numerator = torch.sum(weight * (prediction - target) ** 2, dim=reduction)
    denominator = torch.sum(weight * target ** 2, dim=reduction).clamp_min(
        denominator_epsilon
    )
    loss = torch.mean(numerator / denominator)
    unweighted = torch.mean(
        _per_window_relative_squared(prediction, target, denominator_epsilon)
    )
    return loss, {"unweighted": unweighted, "mean_weight": weights.mean()}


def heteroskedastic_residual_loss(
    normalized_prediction: torch.Tensor,
    predicted_log_var: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
    lambda_mean: float = 0.1,
    lambda_nll: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined scale-balanced relative L2 loss, decoupled temporal mean wake loss, and heteroskedastic NLL."""
    field_rel = scale_balanced_uv_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    mean_wake = temporal_mean_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    diff_sq = (
        normalized_prediction[..., :measured_channels]
        - normalized_target[..., :measured_channels]
    ) ** 2
    var = torch.exp(predicted_log_var)
    nll = 0.5 * torch.mean(diff_sq / var + predicted_log_var)

    total = field_rel + lambda_mean * mean_wake + lambda_nll * nll
    return total, {
        "field_rel": field_rel,
        "mean_wake": mean_wake,
        "nll": nll,
        "mean_predicted_std": torch.mean(torch.sqrt(var)),
    }


def spatial_power_spectrum_2d(
    fluctuations_uv: torch.Tensor,
) -> torch.Tensor:
    """Mean spatial 2D rFFT power spectrum across time and channels: (N, H, W//2 + 1)."""
    if fluctuations_uv.ndim != 5:
        raise ValueError("expected 5D tensor (N, T, H, W, C)")
    fft = torch.fft.rfft2(fluctuations_uv, dim=(-3, -2))
    psd = torch.mean(torch.abs(fft) ** 2, dim=(1, -1))
    return psd


def log_spectral_loss(
    pred_fluctuations: torch.Tensor,
    target_fluctuations: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean absolute error between log power spectra of predicted and target fluctuations."""
    psd_pred = spatial_power_spectrum_2d(pred_fluctuations)
    psd_target = spatial_power_spectrum_2d(target_fluctuations)
    return torch.mean(torch.abs(torch.log(psd_pred + eps) - torch.log(psd_target + eps)))


def mfdr_spectral_loss(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
    *,
    measured_channels: int = 2,
    denominator_epsilon: float = 1e-12,
    tke_denominator_epsilon: float = 1e-8,
    spectral_epsilon: float = 1e-6,
    lambda_mean: float = 0.20,
    lambda_fluct: float = 0.10,
    lambda_tke: float = 0.05,
    lambda_spec: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decoupled Mean-Fluctuation Regularization (MFDR) with log-spectral penalty."""
    if min(lambda_mean, lambda_fluct, lambda_tke, lambda_spec) < 0:
        raise ValueError("MFDR loss weights must be non-negative")

    field = scale_balanced_uv_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    mean = temporal_mean_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    fluct = fluctuation_relative_mse(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=denominator_epsilon,
    )
    tke = tke_relative_l2(
        normalized_prediction,
        normalized_target,
        mean_target,
        std_target,
        measured_channels=measured_channels,
        denominator_epsilon=tke_denominator_epsilon,
    )

    pred_phys = _physical_measured(
        normalized_prediction, mean_target, std_target, measured_channels
    )
    target_phys = _physical_measured(
        normalized_target, mean_target, std_target, measured_channels
    )
    pred_fluct = pred_phys - pred_phys.mean(dim=1, keepdim=True)
    target_fluct = target_phys - target_phys.mean(dim=1, keepdim=True)

    spec = log_spectral_loss(pred_fluct, target_fluct, eps=spectral_epsilon)

    total = (
        field
        + lambda_mean * mean
        + lambda_fluct * fluct
        + lambda_tke * tke
        + lambda_spec * spec
    )
    return total, {
        "field": field,
        "mean": mean,
        "fluct": fluct,
        "tke": tke,
        "spec": spec,
    }
