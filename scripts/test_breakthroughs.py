#!/usr/bin/env python3
"""Systematic preliminary testing for structural breakthrough ideas on Track 1."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

import scoring
from realpde_t1.validation import persist_first_frames

arrays_dir = REPO / "artifacts/e002_validation/arrays"
inputs = np.load(arrays_dir / "inputs.npy")
targets = np.load(arrays_dir / "targets.npy")
preds = np.load(REPO / "artifacts/e027_residual_rollout/predictions.npy")

n_samples = len(inputs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loaded {n_samples} validation windows on {device}.", flush=True)

# 1. Base Hybrid Predictions (persist-first-4)
hybrid = persist_first_frames(preds, inputs, 4)
c = scoring.measured_channels(targets)
rel_l2_base = scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(hybrid, targets, c))))
mvpe_base = scoring.score_error(scoring.mvpe_rel_l2(hybrid, targets))
tke_base = scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(hybrid, targets, c))))

print("\n--- BASELINE METRICS (E027 Point Champion on Fold A) ---", flush=True)
print(f"Rel-L2: {rel_l2_base:.4f} | MVPE: {mvpe_base:.4f} | TKE: {tke_base:.4f}", flush=True)

# Precompute sample metrics for GPU SPS scoring
sigma_global = float(scoring.SIGMA_GLOBAL)
pred_t = torch.from_numpy(hybrid[..., :2]).to(device)
tgt_t = torch.from_numpy(targets[..., :2]).to(device)

dm_arr = scoring.rel_l2_per_sample(hybrid, targets, 2)
tke_arr = scoring.tke_rel_l2_per_sample(hybrid, targets, 2)
mvpe_arr = scoring.mvpe_rel_l2(hybrid, targets)

dm_norm = dm_arr / (0.5 + dm_arr)
tke_norm = tke_arr / (0.5 + tke_arr)
mvpe_norm = float(mvpe_arr / (0.5 + mvpe_arr))

dm_t = torch.from_numpy(dm_norm).reshape((n_samples, 1, 1, 1, 1)).to(device)
tke_t = torch.from_numpy(tke_norm).reshape((n_samples, 1, 1, 1, 1)).to(device)
mvpe_factor = float(1.0 - mvpe_norm)

scored = tgt_t != 0.0
n_scored = float(torch.count_nonzero(scored))

def eval_sps_field(half_width_t: torch.Tensor) -> tuple[float, float]:
    lower = pred_t - half_width_t
    upper = pred_t + half_width_t
    inside = (tgt_t >= lower) & (tgt_t <= upper)
    nil = (2.0 * half_width_t) / sigma_global
    exp_nil = torch.exp(-nil)

    elem_dm = torch.where(inside & scored, (1.0 - dm_t) * exp_nil, 0.0)
    elem_tke = torch.where(inside & scored, (1.0 - tke_t) * exp_nil, 0.0)
    elem_mvpe = torch.where(inside & scored, mvpe_factor * exp_nil, 0.0)

    s_dm = float(torch.sum(elem_dm) / n_scored)
    s_tke = float(torch.sum(elem_tke) / n_scored)
    s_mvpe = float(torch.sum(elem_mvpe) / n_scored)
    weighted = 0.5 * s_dm + 0.3 * s_tke + 0.2 * s_mvpe
    cov = float(torch.count_nonzero(inside & scored) / n_scored)
    return float(100.0 * weighted), cov

# Baseline Frozen E006 SPS
half_e006 = 0.025 * np.abs(hybrid[..., :2]) + 0.15 * sigma_global
sps_e006, cov_e006 = eval_sps_field(torch.from_numpy(half_e006).to(device))
print(f"Baseline Frozen E006 SPS: {sps_e006:.4f} (coverage: {cov_e006*100:.1f}%)", flush=True)

# ==============================================================================
# BREAKTHROUGH 1: DYNAMIC PHYSICAL WAKE-MASKED UNCERTAINTY (VORTICITY & REYNOLDS STRESS)
# ==============================================================================
print("\n=================================================================", flush=True)
print("TEST 1: DYNAMIC PHYSICAL WAKE-MASKED UNCERTAINTY (Vorticity & Reynolds Stress)", flush=True)
print("=================================================================", flush=True)

# Compute input Reynolds normal stresses: <u'^2 + v'^2>
u_mean = np.mean(inputs[..., 0], axis=1, keepdims=True)
v_mean = np.mean(inputs[..., 1], axis=1, keepdims=True)
u_fluct = inputs[..., 0] - u_mean
v_fluct = inputs[..., 1] - v_mean
input_tke = 0.5 * (u_fluct**2 + v_fluct**2) # (N, 20, 32, 64)
mean_tke = np.mean(input_tke, axis=1, keepdims=True) # (N, 1, 32, 64)

# Dynamic wake mask: sigmoid normalized by local turbulence level
p95 = np.percentile(mean_tke, 95, axis=(2, 3), keepdims=True) + 1e-6
p05 = np.percentile(mean_tke, 5, axis=(2, 3), keepdims=True)
wake_mask = np.clip((mean_tke - p05) / (p95 - p05), 0.0, 1.0)
wake_mask = np.repeat(wake_mask, 20, axis=1) # (N, 20, 32, 64)
wake_mask = np.repeat(wake_mask[..., np.newaxis], 2, axis=-1) # (N, 20, 32, 64, 2)
wake_mask_t = torch.from_numpy(wake_mask).to(device, dtype=torch.float32)

best_sps_wake = 0.0
best_params_wake = None

for h_lam in np.linspace(0.001, 0.015, 15):
    for h_wake in np.linspace(0.010, 0.060, 20):
        for gamma in [0.0, 0.02, 0.05]:
            half = h_lam * (1.0 - wake_mask_t) + h_wake * wake_mask_t + gamma * torch.abs(pred_t)
            sc, cov = eval_sps_field(half)
            if sc > best_sps_wake:
                best_sps_wake = sc
                best_params_wake = (float(h_lam), float(h_wake), float(gamma), cov)

print(f"Optimal Dynamic Wake-Masked SPS: {best_sps_wake:.4f} (coverage: {best_params_wake[3]*100:.1f}%)")
print(f"Parameters: h_laminar={best_params_wake[0]:.4f}, h_wake={best_params_wake[1]:.4f}, gamma_pred={best_params_wake[2]:.4f}")
print(f"Net Gain over Frozen E006: +{best_sps_wake - sps_e006:.4f} points!")

# ==============================================================================
# BREAKTHROUGH 2: ENERGY-PRESERVING TKE SPECTRAL INJECTION
# ==============================================================================
print("\n=================================================================", flush=True)
print("TEST 2: ENERGY-PRESERVING TKE SPECTRAL INJECTION & FLUCTUATION RESCALING", flush=True)
print("=================================================================", flush=True)

# Analyze why TKE is underpredicted: measure energy ratio between predicted and target fluctuations
u_pred_mean = np.mean(hybrid[..., 0], axis=1, keepdims=True)
v_pred_mean = np.mean(hybrid[..., 1], axis=1, keepdims=True)
u_pred_fluct = hybrid[..., 0] - u_pred_mean
v_pred_fluct = hybrid[..., 1] - v_pred_mean
tke_pred_field = 0.5 * (u_pred_fluct**2 + v_pred_fluct**2)

u_tgt_mean = np.mean(targets[..., 0], axis=1, keepdims=True)
v_tgt_mean = np.mean(targets[..., 1], axis=1, keepdims=True)
u_tgt_fluct = targets[..., 0] - u_tgt_mean
v_tgt_fluct = targets[..., 1] - v_tgt_mean
tke_tgt_field = 0.5 * (u_tgt_fluct**2 + v_tgt_fluct**2)

mean_pred_tke = np.mean(tke_pred_field)
mean_tgt_tke = np.mean(tke_tgt_field)
energy_ratio = mean_pred_tke / (mean_tgt_tke + 1e-8)
print(f"Mean Predicted TKE: {mean_pred_tke:.6f} | Mean True TKE: {mean_tgt_tke:.6f}")
print(f"Raw Fluctuation Energy Deficit: {energy_ratio*100:.1f}% (Predicted fluctuations have {(1.0-energy_ratio)*100:.1f}% missing energy!)")

# Test physical Fluctuation Amplitude Rescaling (FAR)
best_tke_sc = 0.0
best_scale = 1.0
best_rel_l2 = 0.0
best_mvpe = 0.0

for scale in np.linspace(1.0, 1.35, 36):
    u_rescaled = u_pred_mean + scale * u_pred_fluct
    v_rescaled = v_pred_mean + scale * v_pred_fluct

    hybrid_rescaled = np.zeros_like(hybrid)
    hybrid_rescaled[..., 0] = u_rescaled
    hybrid_rescaled[..., 1] = v_rescaled
    # Retain persist-first-4
    hybrid_rescaled = persist_first_frames(hybrid_rescaled, inputs, 4)

    tke_s = scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(hybrid_rescaled, targets, c))))
    rel_l2_s = scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(hybrid_rescaled, targets, c))))
    mvpe_s = scoring.score_error(scoring.mvpe_rel_l2(hybrid_rescaled, targets))

    if tke_s > best_tke_sc:
        best_tke_sc = tke_s
        best_scale = scale
        best_rel_l2 = rel_l2_s
        best_mvpe = mvpe_s

print(f"Optimal Fluctuation Scale Factor: {best_scale:.3f}")
print(f"Rescaled TKE Score:   {best_tke_sc:.4f} (Baseline was {tke_base:.4f}, Net Gain: +{best_tke_sc - tke_base:.4f}!)")
print(f"Preserved Rel-L2:     {best_rel_l2:.4f} (Baseline was {rel_l2_base:.4f})")
print(f"Preserved MVPE:       {best_mvpe:.4f} (Baseline was {mvpe_base:.4f})")
