#!/usr/bin/env python3
"""Evaluate Experiment E032: Multi-Scale Wavelet & Fluctuation-Decoupled Spectral Energy Adapter on Fold A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))

import scoring
from realpde_t1.adapter import load_residual_cno
from realpde_t1.validation import persist_first_frames

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "e032_wavelet_tke_adapter.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=REPO / "artifacts" / "e032_wavelet_adapter",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ckpt_path = args.artifacts / "final.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing final checkpoint: {ckpt_path}")

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalizer = ckpt["normalizer"]

    # 1. Load model
    model = load_residual_cno(ckpt, device=device).eval()

    # 2. Load validation data
    arrays_dir = REPO / config["evaluation"]["arrays"]
    inputs = np.load(arrays_dir / "inputs.npy")
    targets = np.load(arrays_dir / "targets.npy")
    windows = json.loads((arrays_dir / "windows.json").read_text(encoding="utf-8"))

    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    n_samples = len(inputs)
    batch_size = 4
    preds = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            inp_batch = torch.from_numpy(inputs[i : i + batch_size]).to(device=device, dtype=torch.float32)
            norm_in = (inp_batch - mean_in) / std_in
            norm_out = model(norm_in)
            phys_out = norm_out * std_tg + mean_tg
            phys_out[..., 2] = 0.0
            preds.append(phys_out.cpu().numpy())

    raw_preds = np.concatenate(preds, axis=0)
    np.save(args.artifacts / "predictions.npy", raw_preds)

    # 3. Hybrid persist-first-4 predictions
    hybrid = persist_first_frames(raw_preds, inputs, 4)

    # 4. Standard scoring
    c = scoring.measured_channels(targets)
    rel_l2 = float(np.mean(scoring.rel_l2_per_sample(hybrid, targets, c)))
    tke = float(np.mean(scoring.tke_rel_l2_per_sample(hybrid, targets, c)))
    mvpe = scoring.mvpe_rel_l2(hybrid, targets)

    rel_l2_sc = scoring.score_error(rel_l2)
    tke_sc = scoring.score_error(tke)
    mvpe_sc = scoring.score_error(mvpe)

    eval_cfg = config["evaluation"]["calibrated_uncertainty"]
    sigma_global = float(eval_cfg["sigma_global"])
    half_width = float(eval_cfg["gamma"]) * np.abs(hybrid) * 0.025 + float(eval_cfg["beta_floor"]) * 3.0 * sigma_global
    half_width[..., 2] = 0.0
    lower = hybrid - half_width
    upper = hybrid + half_width
    lower[..., 2] = 0.0
    upper[..., 2] = 0.0

    sps_val, cov = scoring.aggregate_sps(hybrid, targets, 2, lower=lower, upper=upper)
    sps_sc = scoring.score_sps(sps_val)

    print("\n=== EXPERIMENT E032 EVALUATION REPORT (Fold A, 357 windows) ===")
    print(f"Rel-L2 Score:   {rel_l2_sc:.4f}")
    print(f"MVPE Score:     {mvpe_sc:.4f}")
    print(f"TKE Score:      {tke_sc:.4f} (E026 was 72.8490, E027 was 72.3884)")
    print(f"Calibrated SPS: {sps_sc:.4f} (coverage: {cov*100:.1f}%)")

    # 5. Evaluate Updated Tri-Loss Ensemble: E027 (Rollout) + E031 (FiLM) + E032 (Wavelet-TKE)
    p_e027 = np.load(REPO / "artifacts/e027_residual_rollout/predictions.npy")
    p_e031 = np.load(REPO / "artifacts/e031_film_adapter/predictions.npy")
    p_e032 = raw_preds

    p_ens = (p_e027 + p_e031 + p_e032) / 3.0
    hybrid_ens = persist_first_frames(p_ens, inputs, 4)

    rel_l2_ens = scoring.score_error(float(np.mean(scoring.rel_l2_per_sample(hybrid_ens, targets, c))))
    tke_ens = scoring.score_error(float(np.mean(scoring.tke_rel_l2_per_sample(hybrid_ens, targets, c))))
    mvpe_ens = scoring.score_error(scoring.mvpe_rel_l2(hybrid_ens, targets))

    print("\n=== UPDATED TRI-LOSS ENSEMBLE (E027 Rollout + E031 FiLM + E032 Wavelet-TKE) ===")
    print(f"Ensemble Rel-L2 Score: {rel_l2_ens:.4f}")
    print(f"Ensemble MVPE Score:   {mvpe_ens:.4f}")
    print(f"Ensemble TKE Score:    {tke_ens:.4f}")

    gates = config["evaluation"]["gates"]
    criteria_met = bool(
        tke_sc >= float(gates.get("tke_minimum", 72.85))
        and rel_l2_sc >= float(gates.get("rel_l2_minimum", 94.40))
        and mvpe_sc >= float(gates.get("mvpe_minimum", 95.00))
    )

    report = {
        "experiment": "E032",
        "fold": "A",
        "sample_count": n_samples,
        "rel_l2_score": rel_l2_sc,
        "mvpe_score": mvpe_sc,
        "tke_score": tke_sc,
        "calibrated_sps_score": sps_sc,
        "coverage": cov,
        "ensemble_scores": {
            "rel_l2_score": rel_l2_ens,
            "mvpe_score": mvpe_ens,
            "tke_score": tke_ens,
        },
        "all_criteria_met": criteria_met,
    }

    report_path = args.artifacts / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
