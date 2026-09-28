"""Exact SPS decomposition; target-informed ceilings are diagnostic only."""
from __future__ import annotations

import numpy as np


def sps_components(prediction, target, lower, upper, scoring):
    """Return additive per-window/horizon sums, retaining full-window errors.

    Symmetric oracle width |target-prediction| is the narrowest symmetric
    interval covering the target. It is an upper bound for symmetric intervals
    at fixed predictions, not a learnable calibration target or deployable rule.
    """
    p, t = np.asarray(prediction, dtype=np.float32), np.asarray(target, dtype=np.float32)
    lo, hi = np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)
    if p.shape != t.shape or lo.shape != p.shape or hi.shape != p.shape:
        raise ValueError("prediction, target and bounds must have identical shapes")
    if p.ndim != 5 or p.shape[-1] != 3:
        raise ValueError("expected N,T,H,W,3 fields")
    if not all(np.isfinite(a).all() for a in (p, t, lo, hi)) or np.any(lo > hi):
        raise ValueError("nonfinite fields or reversed bounds")
    errors = np.stack((scoring.rel_l2_per_sample(p, t, 2),
                       scoring.tke_rel_l2_per_sample(p, t, 2),
                       scoring.mvpe_rel_l2_per_sample(p, t)), axis=1)
    accuracy = 1.0 - errors / (0.5 + errors)
    accuracy = np.where(np.isfinite(accuracy), accuracy, 0.0)
    q = accuracy @ np.array([0.5, 0.3, 0.2])
    p, t, lo, hi = (a[..., :2] for a in (p, t, lo, hi))
    scored = t != 0.0
    inside = (t >= lo) & (t <= hi) & scored
    reward = np.where(inside, np.exp(-(hi - lo) / scoring.SIGMA_GLOBAL), 0.0)
    oracle = np.where(scored, np.exp(-2 * np.abs(t - p) / scoring.SIGMA_GLOBAL), 0.0)
    axes = (2, 3, 4)
    counts = scored.sum(axis=axes)
    return {
        "count": counts,
        "covered": inside.sum(axis=axes),
        "interval_reward": reward.sum(axis=axes, dtype=np.float64),
        "sps_reward": reward.sum(axis=axes, dtype=np.float64) * q[:, None],
        "symmetric_oracle_reward": oracle.sum(axis=axes, dtype=np.float64) * q[:, None],
        "accuracy_ceiling_reward": counts * q[:, None],
        "errors": errors,
    }


def summarize_sps(parts, indices=None, horizon=slice(None)):
    selection = slice(None) if indices is None else indices
    count = float(parts["count"][selection, horizon].sum())
    if count == 0:
        return {"scored_elements": 0}
    result = {"scored_elements": int(count)}
    for source, name in (("covered", "coverage_percent"),
                         ("interval_reward", "interval_only_score"),
                         ("sps_reward", "sps_score"),
                         ("symmetric_oracle_reward", "symmetric_oracle_sps"),
                         ("accuracy_ceiling_reward", "accuracy_ceiling_sps")):
        result[name] = 100 * float(parts[source][selection, horizon].sum()) / count
    result["symmetric_oracle_headroom"] = result["symmetric_oracle_sps"] - result["sps_score"]
    return result
