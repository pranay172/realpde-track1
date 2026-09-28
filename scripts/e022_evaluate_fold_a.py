#!/usr/bin/env python3
"""Evaluate the fixed-final E022 Heteroskedastic Residual CNO on Fold A."""

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
CONFIG = REPO / "configs" / "e022_heteroskedastic_residual.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e022_heteroskedastic_residual"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.adapter import load_heteroskedastic_residual_cno  # noqa: E402
from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402
from realpde_t1.validation import persist_first_frames, score_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
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


def run_heteroskedastic_inference(
    model: torch.nn.Module,
    inputs: np.ndarray,
    normalizer: dict[str, list[float]],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run batched inference and return (predictions, predicted_stds, mean_seconds_per_sample)."""
    mean_in = torch.tensor(normalizer["mean_input"], device=device, dtype=torch.float32)
    std_in = torch.tensor(normalizer["std_input"], device=device, dtype=torch.float32)
    mean_tg = torch.tensor(normalizer["mean_target"], device=device, dtype=torch.float32)
    std_tg = torch.tensor(normalizer["std_target"], device=device, dtype=torch.float32)

    model.eval()
    n_samples = len(inputs)
    predictions = []
    log_vars = []

    # Warmup
    with torch.no_grad():
        w_in = torch.tensor(inputs[:1], device=device, dtype=torch.float32)
        w_norm = (w_in - mean_in) / std_in
        _ = model(w_norm)
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.tensor(inputs[i:i+batch_size], device=device, dtype=torch.float32)
            b_norm = (batch - mean_in) / std_in
            out_norm, log_v = model(b_norm)
            out_phys = out_norm * std_tg + mean_tg
            out_phys[..., 2] = 0.0
            predictions.append(out_phys.cpu().numpy())
            log_vars.append(log_v.cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    mean_seconds = total_time / n_samples

    preds = np.concatenate(predictions, axis=0)
    lvars = np.concatenate(log_vars, axis=0) # (N, T, H, W, 2)
    # Convert log_var in normalized units to physical std:
    # std_phys = std_target * exp(0.5 * log_var)
    std_u = normalizer["std_target"][0] * np.exp(0.5 * lvars[..., 0])
    std_v = normalizer["std_target"][1] * np.exp(0.5 * lvars[..., 1])
    stds = np.stack([std_u, std_v], axis=-1)

    return preds, stds, mean_seconds


def score_bundle(
    prediction: np.ndarray,
    targets: np.ndarray,
    seconds: float,
    pred_stds: np.ndarray,
    scoring: Any,
) -> dict[str, Any]:
    # 1. Default bounds (0.1 * abs(pred))
    default = score_prediction(prediction, targets, seconds, scoring)

    # 2. Frozen E006 scalar bounds (alpha=0.025, beta=0.15)
    hw_e006 = 0.025 * np.abs(prediction[..., :2]) + 0.15 * float(scoring.SIGMA_GLOBAL)
    lower_e006 = prediction.copy()
    upper_e006 = prediction.copy()
    lower_e006[..., :2] -= hw_e006
    upper_e006[..., :2] += hw_e006
    calibrated = score_prediction(prediction, targets, seconds, scoring, lower=lower_e006, upper=upper_e006)

    # 3. Learned heteroskedastic bounds (gamma=2.0 * pred_std)
    best_learned = None
    best_learned_sps = -1.0
    best_gamma = 2.0

    for gamma in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]:
        hw_l = gamma * pred_stds
        lower_l = prediction.copy()
        upper_l = prediction.copy()
        lower_l[..., :2] -= hw_l
        upper_l[..., :2] += hw_l
        res = score_prediction(prediction, targets, seconds, scoring, lower=lower_l, upper=upper_l)
        sps_val = res["scores"]["sps_score"]
        if sps_val > best_learned_sps:
            best_learned_sps = sps_val
            best_learned = res
            best_gamma = gamma

    return {
        "default": default,
        "calibrated_e006": calibrated,
        "learned_heteroskedastic": best_learned,
        "best_gamma": best_gamma,
    }


def main():
    args = parse_args()
    config = json.loads(args.config.read_bytes())
    experiment = str(config["experiment"])

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Evaluating {experiment} on device {device}...", flush=True)

    ckpt_path = args.artifacts / "final.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing final checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = json.loads((args.artifacts / "normalizer.json").read_text())

    model = load_heteroskedastic_residual_cno(checkpoint, device)

    # Load Fold A validation arrays
    arrays_dir = REPO / config["evaluation"]["arrays"]
    inputs = np.load(arrays_dir / "inputs.npy")
    targets = np.load(arrays_dir / "targets.npy")
    windows = json.loads((arrays_dir / "windows.json").read_text())

    preds, pred_stds, mean_seconds = run_heteroskedastic_inference(
        model, inputs, normalizer, args.batch_size, device
    )
    hybrid_preds = persist_first_frames(preds, inputs, 4)

    scoring = importlib.import_module("scoring")

    # Score raw and persist-first-4
    raw_scores = score_bundle(preds, targets, mean_seconds, pred_stds, scoring)
    persist_scores = score_bundle(hybrid_preds, targets, mean_seconds, pred_stds, scoring)

    # Evaluate by Reynolds number
    by_re = {}
    for re_val in sorted({int(w["nominal_re"]) for w in windows}):
        idx = np.asarray([int(w["index"]) for w in windows if int(w["nominal_re"]) == re_val])
        by_re[str(re_val)] = {
            "raw": score_bundle(preds[idx], targets[idx], mean_seconds, pred_stds[idx], scoring),
            "persist_first_4": score_bundle(hybrid_preds[idx], targets[idx], mean_seconds, pred_stds[idx], scoring),
        }

    # Assess against criteria
    criteria = config["evaluation"]["success_criteria"]
    ref = config["evaluation"]["e019_persist_first_4"]

    p_scores_e006 = persist_scores["calibrated_e006"]["scores"]
    p_scores_learned = persist_scores["learned_heteroskedastic"]["scores"]

    assessment = {
        "sps_beat_e019": bool(p_scores_learned["sps_score"] > ref["calibrated_sps_score"]),
        "sps_score_learned": p_scores_learned["sps_score"],
        "sps_score_e006": p_scores_e006["sps_score"],
        "rel_l2_score": p_scores_learned["rel_l2_score"],
        "mvpe_score": p_scores_learned["mvpe_score"],
        "tke_score": p_scores_learned["tke_score"],
        "rel_l2_pass": bool(p_scores_learned["rel_l2_score"] >= criteria["rel_l2_score_min"]),
        "mvpe_pass": bool(p_scores_learned["mvpe_score"] >= criteria["mvpe_score_min"]),
        "tke_pass": bool(p_scores_learned["tke_score"] >= criteria["tke_score_min"]),
        "re_3750_rel_l2": by_re["3750"]["persist_first_4"]["learned_heteroskedastic"]["scores"]["rel_l2_score"],
        "re_3750_tke": by_re["3750"]["persist_first_4"]["learned_heteroskedastic"]["scores"]["tke_score"],
        "re_3750_mvpe": by_re["3750"]["persist_first_4"]["learned_heteroskedastic"]["scores"]["mvpe_score"],
        "all_criteria_met": bool(
            p_scores_learned["sps_score"] > ref["calibrated_sps_score"]
            and p_scores_learned["rel_l2_score"] >= criteria["rel_l2_score_min"]
            and p_scores_learned["mvpe_score"] >= criteria["mvpe_score_min"]
            and p_scores_learned["tke_score"] >= criteria["tke_score_min"]
        ),
    }

    result = {
        "experiment": experiment,
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "git_head": git_head(),
        "mean_seconds_per_sample": mean_seconds,
        "raw": raw_scores,
        "persist_first_4": persist_scores,
        "by_nominal_re": by_re,
        "assessment": assessment,
    }

    eval_path = args.artifacts / "evaluation.json"
    eval_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\n--- EXPERIMENT E022 EVALUATION RESULTS ---")
    print(f"Persist-First-4 Rel-L2 Score: {p_scores_learned['rel_l2_score']:.3f} (vs E019: {ref['rel_l2_score']:.3f})")
    print(f"Persist-First-4 MVPE Score:   {p_scores_learned['mvpe_score']:.3f} (vs E019: {ref['mvpe_score']:.3f})")
    print(f"Persist-First-4 TKE Score:    {p_scores_learned['tke_score']:.3f} (vs E019: {ref['tke_score']:.3f})")
    print(f"SPS (E006 Scalar Bounds):     {p_scores_e006['sps_score']:.3f}")
    print(f"SPS (Learned Heteroskedastic): {p_scores_learned['sps_score']:.3f} (vs E019: {ref['calibrated_sps_score']:.3f}, best gamma={persist_scores['best_gamma']})")
    print(f"\nAssessment: {json.dumps(assessment, indent=2)}")


if __name__ == "__main__":
    main()
