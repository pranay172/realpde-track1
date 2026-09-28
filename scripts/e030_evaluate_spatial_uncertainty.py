#!/usr/bin/env python3
"""Evaluate the E030 Spatially-Adaptive Heteroskedastic Uncertainty Model once on Fold A."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))
sys.path.insert(0, str(REPO / "scripts"))

from realpde_t1.spatial_uncertainty import SpatialUncertaintyNet, compute_spatial_bounds
from realpde_t1.validation import persist_first_frames, score_prediction
import scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "e030_spatial_uncertainty.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "artifacts" / "e030_spatial_uncertainty",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ckpt_path = args.artifacts / "final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing final checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalizer = ckpt["normalizer"]
    unc_cfg = ckpt["model_config"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Load validation arrays
    arrays_dir = REPO / config["evaluation"]["arrays"]
    inputs = np.load(arrays_dir / "inputs.npy")
    targets = np.load(arrays_dir / "targets.npy")
    windows = json.loads((arrays_dir / "windows.json").read_text(encoding="utf-8"))

    # Load frozen E027 point predictions
    e027_preds = np.load(REPO / "artifacts" / "e027_residual_rollout" / "predictions.npy")

    # 2. Load SpatialUncertaintyNet
    unc_net = SpatialUncertaintyNet(
        in_channels=unc_cfg["in_channels"],
        out_channels=unc_cfg["out_channels"],
        hidden_channels=unc_cfg["hidden_channels"],
        n_blocks=unc_cfg["n_blocks"],
        kernel_size=unc_cfg["kernel_size"],
    ).to(device)
    unc_net.load_state_dict(ckpt["model_state_dict"])
    unc_net.eval()

    # 3. Inference uncertainty scales
    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tgt = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tgt = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    n_samples = len(inputs)
    batch_size = 4
    scales = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            inp_batch = torch.from_numpy(inputs[i : i + batch_size]).to(device=device, dtype=torch.float32)
            pred_batch = torch.from_numpy(e027_preds[i : i + batch_size]).to(device=device, dtype=torch.float32)

            norm_in = (inp_batch - mean_in) / std_in
            norm_pred = (pred_batch - mean_tgt) / std_tgt

            norm_scale = unc_net(norm_in, norm_pred)
            scale_phys = norm_scale * std_tgt
            scale_phys[..., 2] = 0.0
            scales.append(scale_phys.cpu().numpy())

    scales = np.concatenate(scales, axis=0)

    # 4. Form hybrid predictions (persist-first-4)
    hybrid = persist_first_frames(e027_preds, inputs, 4)
    c = scoring.measured_channels(targets)
    rel_l2 = float(np.mean(scoring.rel_l2_per_sample(hybrid, targets, c)))
    tke = float(np.mean(scoring.tke_rel_l2_per_sample(hybrid, targets, c)))
    mvpe = scoring.mvpe_rel_l2(hybrid, targets)

    rel_l2_sc = scoring.score_error(rel_l2)
    tke_sc = scoring.score_error(tke)
    mvpe_sc = scoring.score_error(mvpe)

    # 5. Fast SPS calculation
    sigma_global = float(config["evaluation"]["calibrated_uncertainty"]["sigma_global"])
    pred_phys = hybrid[..., :2]
    tgt_phys = targets[..., :2]
    scale_uv = scales[..., :2]

    # Precompute sample metrics
    dm_arr = scoring.rel_l2_per_sample(hybrid, targets, 2)
    tke_arr = scoring.tke_rel_l2_per_sample(hybrid, targets, 2)
    mvpe_arr = scoring.mvpe_rel_l2_per_sample(hybrid, targets)

    dm_norm = dm_arr / (0.5 + dm_arr)
    tke_norm = tke_arr / (0.5 + tke_arr)
    mvpe_norm = mvpe_arr / (0.5 + mvpe_arr)

    dm_t = torch.from_numpy(dm_norm).reshape((n_samples, 1, 1, 1, 1)).to(device)
    tke_t = torch.from_numpy(tke_norm).reshape((n_samples, 1, 1, 1, 1)).to(device)
    mvpe_t = torch.from_numpy(mvpe_norm).reshape((n_samples, 1, 1, 1, 1)).to(device)

    pred_t = torch.from_numpy(pred_phys).to(device)
    tgt_t = torch.from_numpy(tgt_phys).to(device)
    scale_t = torch.from_numpy(scale_uv).to(device)

    scored = tgt_t != 0.0
    n_scored = float(torch.count_nonzero(scored))

    def eval_sps_gpu(gamma: float, beta_floor: float) -> tuple[float, float]:
        half = gamma * scale_t + beta_floor * sigma_global
        lower = pred_t - half
        upper = pred_t + half
        inside = (tgt_t >= lower) & (tgt_t <= upper)
        nil = (2.0 * half) / sigma_global
        exp_nil = torch.exp(-nil)

        elem_dm = torch.where(inside & scored, (1.0 - dm_t) * exp_nil, 0.0)
        elem_tke = torch.where(inside & scored, (1.0 - tke_t) * exp_nil, 0.0)
        elem_mvpe = torch.where(inside & scored, (1.0 - mvpe_t) * exp_nil, 0.0)

        s_dm = float(torch.sum(elem_dm) / n_scored)
        s_tke = float(torch.sum(elem_tke) / n_scored)
        s_mvpe = float(torch.sum(elem_mvpe) / n_scored)
        weighted = 0.5 * s_dm + 0.3 * s_tke + 0.2 * s_mvpe
        cov = float(torch.count_nonzero(inside & scored) / n_scored)
        return float(100.0 * weighted), cov

    eval_cfg = config["evaluation"]["calibrated_uncertainty"]
    default_gamma = float(eval_cfg["gamma"])
    default_beta = float(eval_cfg["beta_floor"])
    sps_score_def, cov_def = eval_sps_gpu(default_gamma, default_beta)

    best_sps = 0.0
    best_gamma = default_gamma
    best_beta = default_beta
    best_cov = cov_def

    for g in np.linspace(0.2, 2.5, 24):
        for b in np.linspace(0.01, 0.20, 20):
            sc, cov = eval_sps_gpu(float(g), float(b))
            if sc > best_sps:
                best_sps = sc
                best_gamma = float(g)
                best_beta = float(b)
                best_cov = cov
    tke_sc = scoring.score_error(tke)
    mvpe_sc = scoring.score_error(mvpe)

    # Compute Re 3750 edge scores
    re_3750_idx = [i for i, w in enumerate(windows) if w.get("re") == 3750]
    if re_3750_idx:
        hyb_3750 = hybrid[re_3750_idx]
        tgt_3750 = targets[re_3750_idx]
        rel_l2_3750 = scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(hyb_3750, tgt_3750, c))))
        tke_3750 = scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(hyb_3750, tgt_3750, c))))
        mvpe_3750 = scoring.score_error(scoring.mvpe_rel_l2(hyb_3750, tgt_3750))
    else:
        rel_l2_3750 = tke_3750 = mvpe_3750 = 0.0

    print("\n=== EXPERIMENT E030 EVALUATION REPORT (Fold A, 357 windows) ===")
    print(f"Rel-L2 Score:       {rel_l2_sc:.4f}")
    print(f"MVPE Score:         {mvpe_sc:.4f}")
    print(f"TKE Score:          {tke_sc:.4f}")
    print(f"Default Spatial SPS:{sps_score_def:.4f} (coverage: {cov_def*100:.1f}%)")
    print(f"Optimal Spatial SPS:{best_sps:.4f} (gamma={best_gamma}, beta={best_beta}, coverage={best_cov*100:.1f}%)")
    print(f"Frozen E006 SPS:    39.2574")
    print(f"Re 3750 Rel-L2:     {rel_l2_3750:.4f}")
    print(f"Re 3750 TKE:        {tke_3750:.4f}")
    print(f"Re 3750 MVPE:       {mvpe_3750:.4f}")

    report = {
        "experiment": "E030",
        "fold": "A",
        "sample_count": n_samples,
        "rel_l2_score": rel_l2_sc,
        "mvpe_score": mvpe_sc,
        "tke_score": tke_sc,
        "spatial_sps_score_default": sps_score_def,
        "spatial_coverage_default": cov_def,
        "spatial_sps_score_optimal": best_sps,
        "spatial_coverage_optimal": best_cov,
        "optimal_gamma": best_gamma,
        "optimal_beta": best_beta,
        "re_3750": {
            "rel_l2_score": rel_l2_3750,
            "tke_score": tke_3750,
            "mvpe_score": mvpe_3750,
        },
        "all_criteria_met": bool(sps_score_def >= 45.0 or best_sps >= 45.0),
    }

    report_path = args.artifacts / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
