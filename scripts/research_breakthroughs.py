#!/usr/bin/env python3
"""Deep diagnostic and breakthrough research for RealPDE Track 1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

from scoring import score_sps, score_error, SIGMA_GLOBAL, rel_l2_per_sample, tke_rel_l2_per_sample, mvpe_rel_l2_per_sample
from realpde_t1.validation import persist_first_frames
from realpde_t1.adapter import load_residual_cno


def fast_sps_eval(pred: np.ndarray, target: np.ndarray, dm_weight: float, tke_weight: float, mvpe_weight: float,
                  dm: np.ndarray, tke: np.ndarray, mvpe: np.ndarray, scored: np.ndarray, n_scored: int,
                  half_width: np.ndarray) -> tuple[float, float]:
    p = pred[..., :2]
    t = target[..., :2]
    hw = half_width[..., :2]
    lower = p - hw
    upper = p + hw
    inside = (t >= lower) & (t <= upper)
    nil = (2.0 * hw) / SIGMA_GLOBAL
    exp_nil = np.exp(-nil)

    def branch(pm: np.ndarray) -> float:
        shaped_pm = pm.reshape((pm.shape[0], 1, 1, 1, 1))
        elem = (1.0 - shaped_pm) * exp_nil
        elem = np.where(inside, elem, 0.0)
        return float(np.sum(elem, where=scored, dtype=np.float64) / n_scored)

    sps_dm = branch(dm)
    sps_tke = branch(tke)
    sps_mvpe = branch(mvpe)
    weighted = dm_weight * sps_dm + tke_weight * sps_tke + mvpe_weight * sps_mvpe
    coverage = float(np.count_nonzero(inside & scored) / n_scored)
    return weighted, coverage


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running diagnostics on {device}...")

    # Load Fold A validation
    inputs_a = np.load(REPO / "artifacts/e002_validation/arrays/inputs.npy")
    targets_a = np.load(REPO / "artifacts/e002_validation/arrays/targets.npy")

    # Load Fold B validation
    inputs_b = np.load(REPO / "artifacts/e010_secondary_fold/arrays/inputs.npy")
    targets_b = np.load(REPO / "artifacts/e010_secondary_fold/arrays/targets.npy")

    # Load Fold A model (E019)
    ckpt_a = torch.load(REPO / "artifacts/e019_residual_adapter_e005/final.pth", map_location=device, weights_only=False)
    model_a = load_residual_cno(ckpt_a, device).eval()
    norm_a = json.loads((REPO / "artifacts/e005_scale_balanced_uv/normalizer.json").read_text())

    # Load Fold B model (E020)
    ckpt_b = torch.load(REPO / "artifacts/e020_secondary_fold_residual/final.pth", map_location=device, weights_only=False)
    model_b = load_residual_cno(ckpt_b, device).eval()
    norm_b = json.loads((REPO / "artifacts/e010_secondary_fold/normalizer.json").read_text())

    # Predict Fold A
    mean_in_a = torch.tensor(norm_a["mean_input"], device=device, dtype=torch.float32)
    std_in_a = torch.tensor(norm_a["std_input"], device=device, dtype=torch.float32)
    mean_tg_a = torch.tensor(norm_a["mean_target"], device=device, dtype=torch.float32)
    std_tg_a = torch.tensor(norm_a["std_target"], device=device, dtype=torch.float32)

    preds_a = []
    with torch.no_grad():
        for i in range(0, len(inputs_a), 32):
            batch = torch.tensor(inputs_a[i:i+32], device=device, dtype=torch.float32)
            out = model_a((batch - mean_in_a) / std_in_a) * std_tg_a + mean_tg_a
            out[..., 2] = 0.0
            preds_a.append(out.cpu().numpy())
    preds_a = np.concatenate(preds_a, axis=0)
    hybrid_a = persist_first_frames(preds_a, inputs_a, 4)

    # Predict Fold B
    mean_in_b = torch.tensor(norm_b["mean_input"], device=device, dtype=torch.float32)
    std_in_b = torch.tensor(norm_b["std_input"], device=device, dtype=torch.float32)
    mean_tg_b = torch.tensor(norm_b["mean_target"], device=device, dtype=torch.float32)
    std_tg_b = torch.tensor(norm_b["std_target"], device=device, dtype=torch.float32)

    preds_b = []
    with torch.no_grad():
        for i in range(0, len(inputs_b), 32):
            batch = torch.tensor(inputs_b[i:i+32], device=device, dtype=torch.float32)
            out = model_b((batch - mean_in_b) / std_in_b) * std_tg_b + mean_tg_b
            out[..., 2] = 0.0
            preds_b.append(out.cpu().numpy())
    preds_b = np.concatenate(preds_b, axis=0)
    hybrid_b = persist_first_frames(preds_b, inputs_b, 4)

    # Precompute Fold A metrics
    c = 2
    dm_a = rel_l2_per_sample(hybrid_a, targets_a, c)
    tke_a = tke_rel_l2_per_sample(hybrid_a, targets_a, c)
    mvpe_a = mvpe_rel_l2_per_sample(hybrid_a, targets_a)
    dm_a_norm = dm_a / (0.5 + dm_a)
    tke_a_norm = tke_a / (0.5 + tke_a)
    mvpe_a_norm = mvpe_a / (0.5 + mvpe_a)
    scored_a = targets_a[..., :2] != 0.0
    n_scored_a = int(np.count_nonzero(scored_a))

    # Precompute Fold B metrics
    dm_b = rel_l2_per_sample(hybrid_b, targets_b, c)
    tke_b = tke_rel_l2_per_sample(hybrid_b, targets_b, c)
    mvpe_b = mvpe_rel_l2_per_sample(hybrid_b, targets_b)
    dm_b_norm = dm_b / (0.5 + dm_b)
    tke_b_norm = tke_b / (0.5 + tke_b)
    mvpe_b_norm = mvpe_b / (0.5 + mvpe_b)
    scored_b = targets_b[..., :2] != 0.0
    n_scored_b = int(np.count_nonzero(scored_b))

    # Evaluate Fold A Baseline
    hw_a = 0.025 * np.abs(hybrid_a) + 0.15 * SIGMA_GLOBAL
    hw_a[..., 2] = 0.0
    sps_a, cov_a = fast_sps_eval(hybrid_a, targets_a, 0.5, 0.3, 0.2, dm_a_norm, tke_a_norm, mvpe_a_norm, scored_a, n_scored_a, hw_a)
    print(f"\n--- FOLD A BASELINE ---")
    print(f"Rel-L2: {score_error(np.mean(dm_a)):.3f}")
    print(f"MVPE:   {score_error(np.mean(mvpe_a)):.3f}")
    print(f"TKE:    {score_error(np.mean(tke_a)):.3f}")
    print(f"SPS:    {score_sps(sps_a):.3f} (Coverage: {cov_a*100:.2f}%)")

    # Evaluate Fold B Baseline
    hw_b = 0.025 * np.abs(hybrid_b) + 0.15 * SIGMA_GLOBAL
    hw_b[..., 2] = 0.0
    sps_b, cov_b = fast_sps_eval(hybrid_b, targets_b, 0.5, 0.3, 0.2, dm_b_norm, tke_b_norm, mvpe_b_norm, scored_b, n_scored_b, hw_b)
    print(f"\n--- FOLD B BASELINE ---")
    print(f"Rel-L2: {score_error(np.mean(dm_b)):.3f}")
    print(f"MVPE:   {score_error(np.mean(mvpe_b)):.3f}")
    print(f"TKE:    {score_error(np.mean(tke_b)):.3f}")
    print(f"SPS:    {score_sps(sps_b):.3f} (Coverage: {cov_b*100:.2f}%)")

    # 1. Grid search: Time-dependent uncertainty growth
    t_lin = np.linspace(0.05, 1.0, 20, dtype=np.float32).reshape(1, 20, 1, 1, 1)

    best_combo = None
    best_a_sps = score_sps(sps_a)

    for a0 in [0.005, 0.01, 0.015, 0.02, 0.025]:
        for a1 in [0.0, 0.01, 0.02, 0.03, 0.04, 0.06]:
            for b0 in [0.02, 0.04, 0.06, 0.08, 0.10]:
                for b1 in [0.04, 0.08, 0.12, 0.16, 0.20, 0.25]:
                    hw = (a0 + a1 * t_lin) * np.abs(hybrid_a) + (b0 + b1 * t_lin) * SIGMA_GLOBAL
                    hw[..., 2] = 0.0
                    s_val, c_val = fast_sps_eval(hybrid_a, targets_a, 0.5, 0.3, 0.2, dm_a_norm, tke_a_norm, mvpe_a_norm, scored_a, n_scored_a, hw)
                    s_score = score_sps(s_val)
                    if s_score > best_a_sps:
                        best_a_sps = s_score
                        best_combo = (a0, a1, b0, b1, c_val)

    print("\n--- TIME-DEPENDENT BOUNDS RESULTS ---")
    if best_combo:
        a0, a1, b0, b1, c_val = best_combo
        print(f"Fold A SPS: {score_sps(sps_a):.3f} -> {best_a_sps:.3f} (+{best_a_sps - score_sps(sps_a):.3f}) [Cov: {c_val*100:.2f}%]")
        print(f"Optimal parameters: a0={a0}, a1={a1}, b0={b0}, b1={b1}")

        # Test generalization on Fold B
        hw_b_opt = (a0 + a1 * t_lin) * np.abs(hybrid_b) + (b0 + b1 * t_lin) * SIGMA_GLOBAL
        hw_b_opt[..., 2] = 0.0
        sps_b_opt, cov_b_opt = fast_sps_eval(hybrid_b, targets_b, 0.5, 0.3, 0.2, dm_b_norm, tke_b_norm, mvpe_b_norm, scored_b, n_scored_b, hw_b_opt)
        print(f"Fold B Generalization: {score_sps(sps_b):.3f} -> {score_sps(sps_b_opt):.3f} (+{score_sps(sps_b_opt) - score_sps(sps_b):.3f}) [Cov: {cov_b_opt*100:.2f}%]")

    # 2. Channel-Separated (u and v) + Time-dependent Grid Search
    best_chan_combo = None
    best_chan_sps = best_a_sps

    for a_u0 in [0.01, 0.02]:
        for a_u1 in [0.02, 0.04]:
            for a_v0 in [0.002, 0.005, 0.01]:
                for a_v1 in [0.005, 0.01, 0.02]:
                    for bu_0 in [0.04, 0.06, 0.08]:
                        for bu_1 in [0.08, 0.12, 0.16]:
                            for bv_0 in [0.01, 0.02, 0.03]:
                                for bv_1 in [0.01, 0.02, 0.04]:
                                    hw = np.zeros_like(hybrid_a)
                                    hw[..., 0] = (a_u0 + a_u1 * t_lin.squeeze(-1)) * np.abs(hybrid_a[..., 0]) + (bu_0 + bu_1 * t_lin.squeeze(-1)) * SIGMA_GLOBAL
                                    hw[..., 1] = (a_v0 + a_v1 * t_lin.squeeze(-1)) * np.abs(hybrid_a[..., 1]) + (bv_0 + bv_1 * t_lin.squeeze(-1)) * SIGMA_GLOBAL
                                    s_val, c_val = fast_sps_eval(hybrid_a, targets_a, 0.5, 0.3, 0.2, dm_a_norm, tke_a_norm, mvpe_a_norm, scored_a, n_scored_a, hw)
                                    s_score = score_sps(s_val)
                                    if s_score > best_chan_sps:
                                        best_chan_sps = s_score
                                        best_chan_combo = (a_u0, a_u1, a_v0, a_v1, bu_0, bu_1, bv_0, bv_1, c_val)

    print("\n--- CHANNEL + TIME DEPENDENT BOUNDS RESULTS ---")
    if best_chan_combo:
        a_u0, a_u1, a_v0, a_v1, bu_0, bu_1, bv_0, bv_1, c_val = best_chan_combo
        print(f"Fold A SPS: {score_sps(sps_a):.3f} -> {best_chan_sps:.3f} (+{best_chan_sps - score_sps(sps_a):.3f}) [Cov: {c_val*100:.2f}%]")
        print(f"Optimal parameters: u=(a0={a_u0}, a1={a_u1}, b0={bu_0}, b1={bu_1}), v=(a0={a_v0}, a1={a_v1}, b0={bv_0}, b1={bv_1})")

        # Test generalization on Fold B
        hw_b_chan = np.zeros_like(hybrid_b)
        hw_b_chan[..., 0] = (a_u0 + a_u1 * t_lin.squeeze(-1)) * np.abs(hybrid_b[..., 0]) + (bu_0 + bu_1 * t_lin.squeeze(-1)) * SIGMA_GLOBAL
        hw_b_chan[..., 1] = (a_v0 + a_v1 * t_lin.squeeze(-1)) * np.abs(hybrid_b[..., 1]) + (bv_0 + bv_1 * t_lin.squeeze(-1)) * SIGMA_GLOBAL
        sps_b_chan, cov_b_chan = fast_sps_eval(hybrid_b, targets_b, 0.5, 0.3, 0.2, dm_b_norm, tke_b_norm, mvpe_b_norm, scored_b, n_scored_b, hw_b_chan)
        print(f"Fold B Generalization: {score_sps(sps_b):.3f} -> {score_sps(sps_b_chan):.3f} (+{score_sps(sps_b_chan) - score_sps(sps_b):.3f}) [Cov: {cov_b_chan*100:.2f}%]")


if __name__ == "__main__":
    main()
