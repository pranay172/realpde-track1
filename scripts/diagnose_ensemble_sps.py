#!/usr/bin/env python3
"""Diagnostic script: Multi-Model Residual Ensemble & Spread-based SPS on Fold A."""

import numpy as np
import torch
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

import scoring
from realpde_t1.adapter import load_residual_cno
from realpde_t1.validation import persist_first_frames

arrays_dir = REPO / "artifacts/e002_validation/arrays"
inputs = np.load(arrays_dir / "inputs.npy")
targets = np.load(arrays_dir / "targets.npy")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loaded {len(inputs)} windows. Evaluating on {device}...", flush=True)

# Member 1: E027 (Rollout Champion)
ckpt1 = torch.load(REPO / "artifacts/e027_residual_rollout/final.pth", map_location=device, weights_only=False)
model1 = load_residual_cno(ckpt1, device=device).eval()

# Member 2: E026 (MFDR + Log-Spectral TKE Champion)
ckpt2 = torch.load(REPO / "artifacts/e026_mfdr_residual/final.pth", map_location=device, weights_only=False)
model2 = load_residual_cno(ckpt2, device=device).eval()

# Member 3: E019 (Relative L2 Baseline)
ckpt3 = torch.load(REPO / "artifacts/e019_residual_adapter_e005/final.pth", map_location=device, weights_only=False)
model3 = load_residual_cno(ckpt3, device=device).eval()

norm = ckpt1["normalizer"]
mean_in = torch.tensor(norm["mean_input"], device=device, dtype=torch.float32)
std_in = torch.tensor(norm["std_input"], device=device, dtype=torch.float32)
mean_tgt = torch.tensor(norm["mean_target"], device=device, dtype=torch.float32)
std_tgt = torch.tensor(norm["std_target"], device=device, dtype=torch.float32)

n_samples = len(inputs)
batch_size = 8
p1_list, p2_list, p3_list = [], [], []

with torch.no_grad():
    for i in range(0, n_samples, batch_size):
        inp_b = torch.from_numpy(inputs[i : i + batch_size]).to(device=device, dtype=torch.float32)
        norm_in = (inp_b - mean_in) / std_in

        out1 = model1(norm_in) * std_tgt + mean_tgt
        out2 = model2(norm_in) * std_tgt + mean_tgt
        out3 = model3(norm_in) * std_tgt + mean_tgt

        out1[..., 2] = 0.0
        out2[..., 2] = 0.0
        out3[..., 2] = 0.0

        p1_list.append(out1.cpu().numpy())
        p2_list.append(out2.cpu().numpy())
        p3_list.append(out3.cpu().numpy())

p1 = np.concatenate(p1_list, axis=0)
p2 = np.concatenate(p2_list, axis=0)
p3 = np.concatenate(p3_list, axis=0)

p_ens = (p1 + p2 + p3) / 3.0
hybrid_ens = persist_first_frames(p_ens, inputs, 4)

stack = np.stack([p1, p2, p3], axis=0)
ens_std = np.std(stack, axis=0)

c = scoring.measured_channels(targets)
rel_l2 = scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(hybrid_ens, targets, c))))
tke = scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(hybrid_ens, targets, c))))
mvpe = scoring.score_error(scoring.mvpe_rel_l2(hybrid_ens, targets))

print("\n=== MULTI-MODEL RESIDUAL ENSEMBLE RESULTS (Fold A) ===", flush=True)
print(f"Rel-L2 Score: {rel_l2:.4f} (E027 was 94.7515, E019 was 94.619)", flush=True)
print(f"MVPE Score:   {mvpe:.4f} (E027 was 95.4379, E019 was 95.302)", flush=True)
print(f"TKE Score:    {tke:.4f} (E027 was 72.3884, E026 was 72.849)", flush=True)

# Vectorized GPU SPS
sigma_global = float(scoring.SIGMA_GLOBAL)
pred_t = torch.from_numpy(hybrid_ens[..., :2]).to(device)
tgt_t = torch.from_numpy(targets[..., :2]).to(device)
std_t = torch.from_numpy(ens_std[..., :2]).to(device)

dm_arr = scoring.rel_l2_per_sample(hybrid_ens, targets, 2)
tke_arr = scoring.tke_rel_l2_per_sample(hybrid_ens, targets, 2)
mvpe_arr = scoring.mvpe_rel_l2(hybrid_ens, targets)

dm_norm = dm_arr / (0.5 + dm_arr)
tke_norm = tke_arr / (0.5 + tke_arr)
mvpe_norm = float(mvpe_arr / (0.5 + mvpe_arr))

dm_t = torch.from_numpy(dm_norm).reshape((n_samples, 1, 1, 1, 1)).to(device)
tke_t = torch.from_numpy(tke_norm).reshape((n_samples, 1, 1, 1, 1)).to(device)
mvpe_factor = float(1.0 - mvpe_norm)

scored = tgt_t != 0.0
n_scored = float(torch.count_nonzero(scored))

best_sps = 0.0
best_param = None
for gamma in np.linspace(0.5, 4.0, 36):
    for beta in np.linspace(0.01, 0.20, 20):
        half = gamma * std_t + beta * sigma_global
        lower = pred_t - half
        upper = pred_t + half
        inside = (tgt_t >= lower) & (tgt_t <= upper)
        nil = (2.0 * half) / sigma_global
        exp_nil = torch.exp(-nil)

        elem_dm = torch.where(inside & scored, (1.0 - dm_t) * exp_nil, 0.0)
        elem_tke = torch.where(inside & scored, (1.0 - tke_t) * exp_nil, 0.0)
        elem_mvpe = torch.where(inside & scored, mvpe_factor * exp_nil, 0.0)

        s_dm = float(torch.sum(elem_dm) / n_scored)
        s_tke = float(torch.sum(elem_tke) / n_scored)
        s_mvpe = float(torch.sum(elem_mvpe) / n_scored)
        weighted = 0.5 * s_dm + 0.3 * s_tke + 0.2 * s_mvpe
        cov = float(torch.count_nonzero(inside & scored) / n_scored)
        sc = float(100.0 * weighted)
        if sc > best_sps:
            best_sps = sc
            best_param = (gamma, beta, cov)

print(f"\nEnsemble-Spread SPS: {best_sps:.4f} (gamma={best_param[0]:.2f}, beta={best_param[1]:.2f}, coverage={best_param[2]*100:.1f}%)", flush=True)

# Also check with E006 bounds
half_e006 = 0.025 * np.abs(hybrid_ens) + 0.15 * sigma_global
half_e006[..., 2] = 0.0
low_e006 = hybrid_ens - half_e006
up_e006 = hybrid_ens + half_e006
low_e006[..., 2] = 0.0
up_e006[..., 2] = 0.0
sps_e006, cov_e006 = scoring.aggregate_sps(hybrid_ens, targets, 2, lower=low_e006, upper=up_e006)
print(f"Ensemble under E006 Frozen Bounds: SPS = {scoring.score_sps(sps_e006):.4f} (coverage={cov_e006*100:.1f}%)", flush=True)
