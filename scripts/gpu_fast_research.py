#!/usr/bin/env python3
"""100% PyTorch GPU accelerated breakthrough diagnostic for RealPDE Track 1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

from scoring import score_error, score_sps, SIGMA_GLOBAL, rel_l2_per_sample, tke_rel_l2_per_sample, mvpe_rel_l2_per_sample
from realpde_t1.validation import persist_first_frames
from realpde_t1.adapter import load_residual_cno


def compute_sps_gpu(pred_t: torch.Tensor, target_t: torch.Tensor, hw_t: torch.Tensor,
                     dm_norm: torch.Tensor, tke_norm: torch.Tensor, mvpe_norm: torch.Tensor,
                     scored_mask: torch.Tensor, n_scored: int):
    # All tensors are on GPU float32
    lower = pred_t - hw_t
    upper = pred_t + hw_t
    inside = (target_t >= lower) & (target_t <= upper)
    nil = (2.0 * hw_t) / SIGMA_GLOBAL
    exp_nil = torch.exp(-nil)

    def branch(pm: torch.Tensor):
        shaped_pm = pm.view(-1, 1, 1, 1, 1)
        elem = (1.0 - shaped_pm) * exp_nil
        elem = torch.where(inside, elem, torch.zeros_like(elem))
        return (elem * scored_mask).sum().item() / n_scored

    sps_dm = branch(dm_norm)
    sps_tke = branch(tke_norm)
    sps_mvpe = branch(mvpe_norm)
    weighted = 0.5 * sps_dm + 0.3 * sps_tke + 0.2 * sps_mvpe
    coverage = (inside & scored_mask).sum().item() / n_scored
    return 100.0 * min(max(weighted, 0.0), 1.0), coverage


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on GPU: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    inputs_a = np.load(REPO / "artifacts/e002_validation/arrays/inputs.npy")
    targets_a = np.load(REPO / "artifacts/e002_validation/arrays/targets.npy")
    inputs_b = np.load(REPO / "artifacts/e010_secondary_fold/arrays/inputs.npy")
    targets_b = np.load(REPO / "artifacts/e010_secondary_fold/arrays/targets.npy")

    ckpt_a = torch.load(REPO / "artifacts/e019_residual_adapter_e005/final.pth", map_location=device, weights_only=False)
    model_a = load_residual_cno(ckpt_a, device).eval()
    norm_a = json.loads((REPO / "artifacts/e005_scale_balanced_uv/normalizer.json").read_text())

    ckpt_b = torch.load(REPO / "artifacts/e020_secondary_fold_residual/final.pth", map_location=device, weights_only=False)
    model_b = load_residual_cno(ckpt_b, device).eval()
    norm_b = json.loads((REPO / "artifacts/e010_secondary_fold/normalizer.json").read_text())

    # Predict A
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

    # Predict B
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

    # Convert to GPU tensors (only u and v)
    pred_a_t = torch.tensor(hybrid_a[..., :2], device=device, dtype=torch.float32)
    target_a_t = torch.tensor(targets_a[..., :2], device=device, dtype=torch.float32)
    scored_a_t = target_a_t != 0.0
    n_scored_a = int(scored_a_t.sum().item())

    pred_b_t = torch.tensor(hybrid_b[..., :2], device=device, dtype=torch.float32)
    target_b_t = torch.tensor(targets_b[..., :2], device=device, dtype=torch.float32)
    scored_b_t = target_b_t != 0.0
    n_scored_b = int(scored_b_t.sum().item())

    # Precompute sample metrics
    dm_a = rel_l2_per_sample(hybrid_a, targets_a, 2)
    tke_a = tke_rel_l2_per_sample(hybrid_a, targets_a, 2)
    mvpe_a = mvpe_rel_l2_per_sample(hybrid_a, targets_a)
    dm_a_t = torch.tensor(dm_a / (0.5 + dm_a), device=device, dtype=torch.float32)
    tke_a_t = torch.tensor(tke_a / (0.5 + tke_a), device=device, dtype=torch.float32)
    mvpe_a_t = torch.tensor(mvpe_a / (0.5 + mvpe_a), device=device, dtype=torch.float32)

    dm_b = rel_l2_per_sample(hybrid_b, targets_b, 2)
    tke_b = tke_rel_l2_per_sample(hybrid_b, targets_b, 2)
    mvpe_b = mvpe_rel_l2_per_sample(hybrid_b, targets_b)
    dm_b_t = torch.tensor(dm_b / (0.5 + dm_b), device=device, dtype=torch.float32)
    tke_b_t = torch.tensor(tke_b / (0.5 + tke_b), device=device, dtype=torch.float32)
    mvpe_b_t = torch.tensor(mvpe_b / (0.5 + mvpe_b), device=device, dtype=torch.float32)

    # 1. Baseline Evaluation
    hw_base_a = 0.025 * torch.abs(pred_a_t) + 0.15 * SIGMA_GLOBAL
    sps_base_a, cov_base_a = compute_sps_gpu(pred_a_t, target_a_t, hw_base_a, dm_a_t, tke_a_t, mvpe_a_t, scored_a_t, n_scored_a)

    hw_base_b = 0.025 * torch.abs(pred_b_t) + 0.15 * SIGMA_GLOBAL
    sps_base_b, cov_base_b = compute_sps_gpu(pred_b_t, target_b_t, hw_base_b, dm_b_t, tke_b_t, mvpe_b_t, scored_b_t, n_scored_b)

    print(f"\n[CURRENT BASELINE]", flush=True)
    print(f"Fold A SPS: {sps_base_a:.3f} (Coverage: {cov_base_a*100:.2f}%)", flush=True)
    print(f"Fold B SPS: {sps_base_b:.3f} (Coverage: {cov_base_b*100:.2f}%)", flush=True)

    # 2. Time-growth grid search
    t_vec = torch.linspace(0.05, 1.0, 20, device=device, dtype=torch.float32).view(1, 20, 1, 1, 1)

    best_sps_a = sps_base_a
    best_params = None

    for a0 in [0.005, 0.01, 0.015, 0.02, 0.025]:
        for a1 in [0.0, 0.01, 0.02, 0.03, 0.04, 0.06]:
            for b0 in [0.02, 0.04, 0.06, 0.08, 0.10]:
                for b1 in [0.04, 0.08, 0.12, 0.16, 0.20, 0.25]:
                    hw = (a0 + a1 * t_vec) * torch.abs(pred_a_t) + (b0 + b1 * t_vec) * SIGMA_GLOBAL
                    score, cov = compute_sps_gpu(pred_a_t, target_a_t, hw, dm_a_t, tke_a_t, mvpe_a_t, scored_a_t, n_scored_a)
                    if score > best_sps_a:
                        best_sps_a = score
                        best_params = (a0, a1, b0, b1, cov)

    print(f"\n[TIME-DEPENDENT UNCERTAINTY GROWTH]", flush=True)
    if best_params:
        a0, a1, b0, b1, cov = best_params
        print(f"Fold A: {sps_base_a:.3f} -> {best_sps_a:.3f} (+{best_sps_a - sps_base_a:.3f}) [Cov: {cov*100:.2f}%]", flush=True)
        print(f"Params: a0={a0}, a1={a1}, b0={b0}, b1={b1}", flush=True)

        hw_opt_b = (a0 + a1 * t_vec) * torch.abs(pred_b_t) + (b0 + b1 * t_vec) * SIGMA_GLOBAL
        sps_opt_b, cov_opt_b = compute_sps_gpu(pred_b_t, target_b_t, hw_opt_b, dm_b_t, tke_b_t, mvpe_b_t, scored_b_t, n_scored_b)
        print(f"Fold B Generalization: {sps_base_b:.3f} -> {sps_opt_b:.3f} (+{sps_opt_b - sps_base_b:.3f}) [Cov: {cov_opt_b*100:.2f}%]", flush=True)

    # 3. Channel-Specific (u and v) + Time-dependent Grid Search
    best_chan_sps = best_sps_a
    best_chan_params = None

    for a_u0 in [0.01, 0.02]:
        for a_u1 in [0.01, 0.02, 0.04]:
            for a_v0 in [0.002, 0.005, 0.01]:
                for a_v1 in [0.005, 0.01, 0.02]:
                    for bu_0 in [0.03, 0.05, 0.08]:
                        for bu_1 in [0.08, 0.12, 0.16]:
                            for bv_0 in [0.01, 0.02, 0.03]:
                                for bv_1 in [0.01, 0.02, 0.04]:
                                    hw_u = (a_u0 + a_u1 * t_vec.squeeze(-1)) * torch.abs(pred_a_t[..., 0:1]) + (bu_0 + bu_1 * t_vec.squeeze(-1)) * SIGMA_GLOBAL
                                    hw_v = (a_v0 + a_v1 * t_vec.squeeze(-1)) * torch.abs(pred_a_t[..., 1:2]) + (bv_0 + bv_1 * t_vec.squeeze(-1)) * SIGMA_GLOBAL
                                    hw = torch.cat([hw_u, hw_v], dim=-1)
                                    score, cov = compute_sps_gpu(pred_a_t, target_a_t, hw, dm_a_t, tke_a_t, mvpe_a_t, scored_a_t, n_scored_a)
                                    if score > best_chan_sps:
                                        best_chan_sps = score
                                        best_chan_params = (a_u0, a_u1, a_v0, a_v1, bu_0, bu_1, bv_0, bv_1, cov)

    print(f"\n[CHANNEL-SEPARATED + TIME-DEPENDENT UNCERTAINTY]", flush=True)
    if best_chan_params:
        a_u0, a_u1, a_v0, a_v1, bu_0, bu_1, bv_0, bv_1, cov = best_chan_params
        print(f"Fold A: {sps_base_a:.3f} -> {best_chan_sps:.3f} (+{best_chan_sps - sps_base_a:.3f}) [Cov: {cov*100:.2f}%]", flush=True)
        print(f"Optimal parameters: u=(a0={a_u0}, a1={a_u1}, b0={bu_0}, b1={bu_1}), v=(a0={a_v0}, a1={a_v1}, b0={bv_0}, b1={bv_1})", flush=True)

        hw_u_b = (a_u0 + a_u1 * t_vec.squeeze(-1)) * torch.abs(pred_b_t[..., 0:1]) + (bu_0 + bu_1 * t_vec.squeeze(-1)) * SIGMA_GLOBAL
        hw_v_b = (a_v0 + a_v1 * t_vec.squeeze(-1)) * torch.abs(pred_b_t[..., 1:2]) + (bv_0 + bv_1 * t_vec.squeeze(-1)) * SIGMA_GLOBAL
        hw_chan_b = torch.cat([hw_u_b, hw_v_b], dim=-1)
        sps_b_c, cov_b_c = compute_sps_gpu(pred_b_t, target_b_t, hw_chan_b, dm_b_t, tke_b_t, mvpe_b_t, scored_b_t, n_scored_b)
        print(f"Fold B Generalization: {sps_base_b:.3f} -> {sps_b_c:.3f} (+{sps_b_c - sps_base_b:.3f}) [Cov: {cov_b_c*100:.2f}%]", flush=True)


if __name__ == "__main__":
    main()
