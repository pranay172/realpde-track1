"""Saved-prediction and checkpoint-drift diagnostics for Track 1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import torch


_BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def quality_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    scoring: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return official quality-only metrics and their per-window raw errors."""
    scoring.validate_shapes(prediction, target)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction contains non-finite values")
    channels = scoring.measured_channels(target)
    per_window = {
        "relative_l2": scoring.rel_l2_per_sample(prediction, target, channels),
        "tke_relative_l2": scoring.tke_rel_l2_per_sample(prediction, target, channels),
        "mvpe_relative_l2": scoring.mvpe_rel_l2_per_sample(prediction, target),
    }
    sps, coverage = scoring.aggregate_sps(prediction, target, channels)
    raw = {name: float(np.mean(values)) for name, values in per_window.items()}
    raw.update({"sps_aggregate": float(sps), "interval_coverage": float(coverage)})
    scores = {
        "rel_l2_score": scoring.score_error(raw["relative_l2"]),
        "tke_score": scoring.score_error(raw["tke_relative_l2"]),
        "mvpe_score": scoring.score_error(raw["mvpe_relative_l2"]),
        "sps_score": scoring.score_sps(raw["sps_aggregate"]),
    }
    return {"n_samples": int(prediction.shape[0]), "raw": raw, "scores": scores}, per_window


def compare_predictions(
    prediction: np.ndarray,
    persistence: np.ndarray,
    target: np.ndarray,
    scoring: Any,
) -> dict[str, Any]:
    """Compare a learned prediction with persistence on one exact slice."""
    learned, learned_each = quality_metrics(prediction, target, scoring)
    baseline, baseline_each = quality_metrics(persistence, target, scoring)
    score_delta = {
        name: learned["scores"][name] - baseline["scores"][name]
        for name in learned["scores"]
    }
    raw_delta = {
        name: learned["raw"][name] - baseline["raw"][name]
        for name in learned["raw"]
    }
    win_fraction = {
        name: float(np.mean(learned_each[name] < baseline_each[name]))
        for name in learned_each
    }
    return {
        "e003": learned,
        "persistence": baseline,
        "e003_minus_persistence": {"raw": raw_delta, "scores": score_delta},
        "e003_better_window_fraction": win_fraction,
    }


def lead_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    measured_channels: int = 2,
) -> list[float]:
    """Return mean per-window relative L2 independently at each lead time."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(f"invalid prediction/target shapes: {prediction.shape}/{target.shape}")
    result = []
    for lead in range(prediction.shape[1]):
        pred = prediction[:, lead, ..., :measured_channels].reshape(prediction.shape[0], -1)
        truth = target[:, lead, ..., :measured_channels].reshape(target.shape[0], -1)
        denom = np.linalg.norm(truth, axis=1).clip(min=1e-8)
        result.append(float(np.mean(np.linalg.norm(pred - truth, axis=1) / denom)))
    return result


def field_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    normalizer: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Summarize measured-channel scale, masking, bias, and normalized shift."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    channels: dict[str, Any] = {}
    target_mean_reference = normalizer["mean_target"]
    target_std_reference = normalizer["std_target"]
    for index, name in enumerate(("u", "v")):
        truth = np.asarray(target[..., index])
        pred = np.asarray(prediction[..., index])
        scored = truth != 0.0
        if not np.any(scored):
            raise ValueError(f"channel {name} has no non-zero target elements")
        truth_scored = truth[scored]
        pred_scored = pred[scored]
        full_mean = float(np.mean(truth, dtype=np.float64))
        full_std = float(np.std(truth, dtype=np.float64))
        train_mean = float(target_mean_reference[index])
        train_std = float(target_std_reference[index])
        channels[name] = {
            "target_zero_fraction": float(1.0 - np.mean(scored)),
            "target_mean_all": full_mean,
            "target_std_all": full_std,
            "target_mean_scored": float(np.mean(truth_scored, dtype=np.float64)),
            "target_std_scored": float(np.std(truth_scored, dtype=np.float64)),
            "prediction_mean_scored": float(np.mean(pred_scored, dtype=np.float64)),
            "prediction_std_scored": float(np.std(pred_scored, dtype=np.float64)),
            "prediction_bias_scored": float(np.mean(pred_scored - truth_scored, dtype=np.float64)),
            "target_full_mean_z_vs_train": (full_mean - train_mean) / train_std,
            "target_full_std_ratio_vs_train": full_std / train_std,
        }
    return channels


def _aggregate_drift(
    initial: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> dict[str, Any]:
    initial_sq = 0.0
    final_sq = 0.0
    delta_sq = 0.0
    dot = 0.0
    elements = 0
    changed = 0
    for name in names:
        before = initial[name].detach().to(dtype=torch.float64, device="cpu")
        after = final[name].detach().to(dtype=torch.float64, device="cpu")
        if before.shape != after.shape:
            raise ValueError(f"state shape changed for {name}: {before.shape} != {after.shape}")
        delta = after - before
        initial_sq += float(torch.sum(before * before))
        final_sq += float(torch.sum(after * after))
        delta_sq += float(torch.sum(delta * delta))
        dot += float(torch.sum(before * after))
        elements += before.numel()
        changed += int(torch.count_nonzero(delta))
    denom = math.sqrt(initial_sq * final_sq)
    return {
        "tensors": len(names),
        "elements": elements,
        "changed_elements": changed,
        "changed_fraction": changed / elements if elements else 0.0,
        "initial_l2": math.sqrt(initial_sq),
        "final_l2": math.sqrt(final_sq),
        "delta_l2": math.sqrt(delta_sq),
        "relative_l2": math.sqrt(delta_sq / initial_sq) if initial_sq > 0 else None,
        "cosine_similarity": dot / denom if denom > 0 else None,
        "delta_rms": math.sqrt(delta_sq / elements) if elements else 0.0,
    }


def checkpoint_drift(
    initial: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Separate trainable-state drift from BatchNorm running-buffer drift."""
    if set(initial) != set(final):
        missing = sorted(set(initial) - set(final))
        extra = sorted(set(final) - set(initial))
        raise ValueError(f"checkpoint state keys differ; missing={missing}, extra={extra}")
    floating = [name for name, value in initial.items() if torch.is_floating_point(value)]
    parameter_names = [
        name for name in floating if not name.endswith(("running_mean", "running_var"))
    ]
    bn_affine = [name for name in parameter_names if ".batch_norm." in name]
    non_bn = [name for name in parameter_names if name not in bn_affine]
    categories = {
        "all_trainable": _aggregate_drift(initial, final, parameter_names),
        "batchnorm_affine": _aggregate_drift(initial, final, bn_affine),
        "non_batchnorm_trainable": _aggregate_drift(initial, final, non_bn),
    }

    components: dict[str, Any] = {}
    for component in sorted({name.split(".", 1)[0] for name in parameter_names}):
        names = [name for name in parameter_names if name.split(".", 1)[0] == component]
        components[component] = _aggregate_drift(initial, final, names)

    tensor_rows = []
    for name in parameter_names:
        stats = _aggregate_drift(initial, final, [name])
        stats["name"] = name
        tensor_rows.append(stats)
    tensor_rows.sort(key=lambda row: row["delta_l2"], reverse=True)

    layer_rows = []
    all_standardized_mean_shifts: list[np.ndarray] = []
    all_std_ratios: list[np.ndarray] = []
    increments = []
    for mean_name in sorted(name for name in floating if name.endswith("running_mean")):
        prefix = mean_name[: -len("running_mean")]
        variance_name = prefix + "running_var"
        counter_name = prefix + "num_batches_tracked"
        before_mean = initial[mean_name].detach().cpu().numpy().astype(np.float64)
        after_mean = final[mean_name].detach().cpu().numpy().astype(np.float64)
        before_var = initial[variance_name].detach().cpu().numpy().astype(np.float64)
        after_var = final[variance_name].detach().cpu().numpy().astype(np.float64)
        standardized = (after_mean - before_mean) / np.sqrt(np.maximum(before_var, 1e-12))
        std_ratio = np.sqrt(np.maximum(after_var, 1e-12) / np.maximum(before_var, 1e-12))
        increment = int(final[counter_name].item() - initial[counter_name].item())
        increments.append(increment)
        all_standardized_mean_shifts.append(standardized.reshape(-1))
        all_std_ratios.append(std_ratio.reshape(-1))
        layer_rows.append({
            "layer": prefix.rstrip("."),
            "channels": int(before_mean.size),
            "mean_shift_standardized_rms": float(np.sqrt(np.mean(standardized ** 2))),
            "mean_shift_standardized_max_abs": float(np.max(np.abs(standardized))),
            "variance_relative_l2": float(
                np.linalg.norm(after_var - before_var) / max(np.linalg.norm(before_var), 1e-12)
            ),
            "median_std_ratio": float(np.median(std_ratio)),
            "num_batches_increment": increment,
        })
    if not layer_rows:
        raise ValueError("no BatchNorm running-state tensors found")
    shifts = np.concatenate(all_standardized_mean_shifts)
    std_ratios = np.concatenate(all_std_ratios)
    layer_rows.sort(key=lambda row: row["mean_shift_standardized_rms"], reverse=True)
    running_mean_names = [name for name in floating if name.endswith("running_mean")]
    running_var_names = [name for name in floating if name.endswith("running_var")]
    batchnorm = {
        "layers": len(layer_rows),
        "channels": int(shifts.size),
        "running_mean_relative_l2": _aggregate_drift(initial, final, running_mean_names)["relative_l2"],
        "running_variance_relative_l2": _aggregate_drift(initial, final, running_var_names)["relative_l2"],
        "running_mean_shift_standardized_rms": float(np.sqrt(np.mean(shifts ** 2))),
        "running_mean_shift_gt_half_std_fraction": float(np.mean(np.abs(shifts) > 0.5)),
        "running_mean_shift_gt_one_std_fraction": float(np.mean(np.abs(shifts) > 1.0)),
        "running_std_ratio_median": float(np.median(std_ratios)),
        "running_std_ratio_p05": float(np.quantile(std_ratios, 0.05)),
        "running_std_ratio_p95": float(np.quantile(std_ratios, 0.95)),
        "num_batches_increment_unique": sorted(set(increments)),
        "top_layers_by_standardized_mean_shift": layer_rows[:10],
    }
    return {
        "state_tensors": len(initial),
        "floating_state_tensors": len(floating),
        "categories": categories,
        "components": components,
        "top_trainable_tensors_by_delta_l2": tensor_rows[:15],
        "batchnorm_running_state": batchnorm,
    }
