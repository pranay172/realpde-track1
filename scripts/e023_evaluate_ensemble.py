#!/usr/bin/env python3
"""Evaluate the E023 Shared-Backbone Multi-Residual Ensemble on Fold A."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
OUTPUT_DIR = REPO / "artifacts" / "e023_ensemble"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.ensemble import load_shared_backbone_ensemble  # noqa: E402
from realpde_t1.adapter import load_residual_cno, load_heteroskedastic_residual_cno  # noqa: E402
from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402
from realpde_t1.validation import persist_first_frames, score_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_ensemble_inference(
    ensemble: torch.nn.Module,
    inputs: np.ndarray,
    normalizer: dict[str, list[float]],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run batched inference and return (predictions, total_variances, mean_seconds_per_sample)."""
    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    ensemble.eval()
    n_samples = len(inputs)
    predictions = []
    variances = []

    # Warmup
    with torch.no_grad():
        w_in = torch.tensor(inputs[:1], device=device, dtype=torch.float32)
        w_norm = (w_in - mean_in) / std_in
        _ = ensemble(w_norm)
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.tensor(inputs[i:i+batch_size], device=device, dtype=torch.float32)
            b_norm = (batch - mean_in) / std_in
            out_norm, total_var = ensemble(b_norm)
            out_phys = out_norm * std_tg + mean_tg
            out_phys[..., 2] = 0.0
            predictions.append(out_phys.cpu().numpy())
            if total_var is not None:
                # Convert normalized variance to physical variance:
                # var_phys = std_target^2 * var_norm
                var_u = (normalizer["std_target"][0] ** 2) * total_var[..., 0].cpu().numpy()
                var_v = (normalizer["std_target"][1] ** 2) * total_var[..., 1].cpu().numpy()
                variances.append(np.stack([var_u, var_v], axis=-1))
        if device.type == "cuda":
            torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    mean_seconds = total_time / n_samples

    preds = np.concatenate(predictions, axis=0)
    vars_arr = np.concatenate(variances, axis=0) if variances else None

    return preds, vars_arr, mean_seconds


def main():
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Evaluating E023 Ensemble on device {device}...", flush=True)

    # 1. Load checkpoints
    ckpt_paths = [
        REPO / "artifacts" / "e019_residual_adapter_e005" / "final.pth",
        REPO / "artifacts" / "e022_heteroskedastic_residual" / "final.pth",
        REPO / "artifacts" / "e023_adapter3" / "final.pth",
    ]
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required checkpoint: {p}")

    checkpoints = [torch.load(p, map_location=device, weights_only=False) for p in ckpt_paths]
    normalizer = json.loads((REPO / "artifacts" / "e019_residual_adapter_e005" / "normalizer.json").read_text())

    # Build ensemble
    ensemble = load_shared_backbone_ensemble(checkpoints, device)

    # Load Fold A validation arrays
    arrays_dir = REPO / "artifacts" / "e002_validation" / "arrays"
    inputs = np.load(arrays_dir / "inputs.npy")
    targets = np.load(arrays_dir / "targets.npy")
    windows = json.loads((arrays_dir / "windows.json").read_text())

    preds, vars_arr, mean_seconds = run_ensemble_inference(
        ensemble, inputs, normalizer, args.batch_size, device
    )
    hybrid_preds = persist_first_frames(preds, inputs, 4)

    scoring = importlib.import_module("scoring")

    # Score default and frozen E006 intervals. Ensemble variance is computed
    # during inference but is not used for SPS.
    b_l, b_u = interval_bounds_numpy(hybrid_preds, 0.025, 0.15, float(scoring.SIGMA_GLOBAL))
    calibrated = score_prediction(hybrid_preds, targets, mean_seconds, scoring, lower=b_l, upper=b_u)
    default = score_prediction(hybrid_preds, targets, mean_seconds, scoring)

    # By Reynolds number
    by_re = {}
    for re_val in sorted({int(w["nominal_re"]) for w in windows}):
        idx = np.asarray([int(w["index"]) for w in windows if int(w["nominal_re"]) == re_val])
        by_re[str(re_val)] = score_prediction(
            hybrid_preds[idx], targets[idx], mean_seconds, scoring,
            lower=b_l[idx], upper=b_u[idx]
        )

    # Pre-registered criteria
    cal_scores = calibrated["scores"]
    assessment = {
        "rel_l2_score": cal_scores["rel_l2_score"],
        "mvpe_score": cal_scores["mvpe_score"],
        "tke_score": cal_scores["tke_score"],
        "sps_score": cal_scores["sps_score"],
        "rel_l2_pass": bool(cal_scores["rel_l2_score"] >= 94.62),
        "mvpe_pass": bool(cal_scores["mvpe_score"] >= 95.30),
        "sps_pass": bool(cal_scores["sps_score"] >= 38.85),
        "tke_pass": bool(cal_scores["tke_score"] >= 72.20),
        "re_3750_rel_l2": by_re["3750"]["scores"]["rel_l2_score"],
        "re_3750_tke": by_re["3750"]["scores"]["tke_score"],
        "re_3750_mvpe": by_re["3750"]["scores"]["mvpe_score"],
        "all_criteria_met": bool(
            cal_scores["rel_l2_score"] >= 94.62
            and cal_scores["mvpe_score"] >= 95.30
            and cal_scores["sps_score"] >= 38.85
            and cal_scores["tke_score"] >= 72.20
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "E023",
        "n_adapters": len(checkpoints),
        "mean_seconds_per_sample": mean_seconds,
        "calibrated_scores": cal_scores,
        "default_scores": default["scores"],
        "by_nominal_re": by_re,
        "assessment": assessment,
    }
    (OUTPUT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== EXPERIMENT E023 SHARED-BACKBONE ENSEMBLE RESULTS ===")
    print(f"Rel-L2 Score: {cal_scores['rel_l2_score']:.4f} (gate >= 94.62, E019: 94.619, E022: 94.601)")
    print(f"MVPE Score:   {cal_scores['mvpe_score']:.4f} (gate >= 95.30, E019: 95.302, E022: 95.291)")
    print(f"TKE Score:    {cal_scores['tke_score']:.4f} (gate >= 72.20, E019: 72.255, E022: 72.247)")
    print(f"SPS Score:    {cal_scores['sps_score']:.4f} (gate >= 38.85, E019: 38.645, E022: 38.843)")
    print(f"Assessment: {json.dumps(assessment, indent=2)}")


if __name__ == "__main__":
    main()
