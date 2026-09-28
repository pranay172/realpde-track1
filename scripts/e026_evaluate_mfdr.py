#!/usr/bin/env python3
"""Evaluate the fixed-final E026 MFDR residual checkpoint once on Fold A."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e026_mfdr_residual_adapter.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e026_mfdr_residual"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from e003_evaluate_cno import timed_prediction  # noqa: E402
from realpde_t1.adapter import load_residual_cno  # noqa: E402
from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402
from realpde_t1.validation import persist_first_frames, score_prediction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def score_pair(
    prediction: np.ndarray,
    targets: np.ndarray,
    seconds: float,
    interval: dict[str, float],
    scoring: Any,
) -> dict[str, Any]:
    default = score_prediction(prediction, targets, seconds, scoring)
    lower, upper = interval_bounds_numpy(
        prediction,
        float(interval["alpha"]),
        float(interval["beta"]),
        float(interval["sigma_global"]),
    )
    calibrated = score_prediction(
        prediction,
        targets,
        seconds,
        scoring,
        lower=lower,
        upper=upper,
    )
    return {
        "raw": default["raw"],
        "default_scores": default["scores"],
        "calibrated_scores": calibrated["scores"],
    }


def by_nominal_re(
    prediction: np.ndarray,
    targets: np.ndarray,
    windows: list[dict[str, Any]],
    seconds: float,
    interval: dict[str, float],
    scoring: Any,
) -> dict[str, Any]:
    metrics = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        indices = np.asarray([
            int(row["index"]) for row in windows if int(row["nominal_re"]) == re_value
        ])
        metrics[str(re_value)] = score_pair(
            prediction[indices], targets[indices], seconds, interval, scoring
        )
    return metrics


def assess_candidate(
    metrics: dict[str, Any],
    re_slices: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, Any]:
    cal = metrics["calibrated_scores"]
    re_3750 = re_slices["3750"]["calibrated_scores"]

    rel_l2_pass = bool(cal["rel_l2_score"] >= gates["rel_l2_score_min"])
    mvpe_pass = bool(cal["mvpe_score"] >= gates["mvpe_score_min"])
    tke_pass = bool(cal["tke_score"] >= gates["tke_score_min"])
    sps_pass = bool(cal["sps_score"] >= gates["sps_score_min"])

    re_3750_rel_l2_pass = bool(re_3750["rel_l2_score"] >= gates["re_3750_rel_l2_min"])
    re_3750_tke_pass = bool(re_3750["tke_score"] >= gates["re_3750_tke_min"])
    re_3750_mvpe_pass = bool(re_3750["mvpe_score"] >= gates["re_3750_mvpe_min"])

    all_pass = (
        rel_l2_pass
        and mvpe_pass
        and tke_pass
        and sps_pass
        and re_3750_rel_l2_pass
        and re_3750_tke_pass
        and re_3750_mvpe_pass
    )

    return {
        "rel_l2_score": cal["rel_l2_score"],
        "mvpe_score": cal["mvpe_score"],
        "tke_score": cal["tke_score"],
        "sps_score": cal["sps_score"],
        "rel_l2_pass": rel_l2_pass,
        "mvpe_pass": mvpe_pass,
        "tke_pass": tke_pass,
        "sps_pass": sps_pass,
        "re_3750_rel_l2": re_3750["rel_l2_score"],
        "re_3750_tke": re_3750["tke_score"],
        "re_3750_mvpe": re_3750["mvpe_score"],
        "re_3750_pass": re_3750_rel_l2_pass and re_3750_tke_pass and re_3750_mvpe_pass,
        "all_criteria_met": all_pass,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checkpoint_path = args.artifacts / "final.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing final checkpoint: {checkpoint_path}")

    arrays_dir = REPO / config["evaluation"]["arrays"]
    inputs = np.load(arrays_dir / "inputs.npy")
    targets = np.load(arrays_dir / "targets.npy")
    windows = json.loads((arrays_dir / "windows.json").read_text(encoding="utf-8"))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalizer = checkpoint["normalizer"]
    scoring = importlib.import_module("scoring")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Evaluating E026 MFDR Residual CNO on {device}...")
    model = load_residual_cno(checkpoint, device=device)
    model.eval()

    prediction_path = args.artifacts / "predictions.npy"
    pred_seconds, _ = timed_prediction(
        model,
        device,
        inputs,
        prediction_path,
        args.batch_size,
        normalizer,
    )
    raw_pred = np.load(prediction_path)
    raw_pred[..., 2] = 0.0

    k = int(config["evaluation"]["persist_first_frames"])
    hybrid_pred = persist_first_frames(raw_pred, inputs, k)

    interval = config["evaluation"]["uncertainty"]
    raw_results = score_pair(raw_pred, targets, pred_seconds, interval, scoring)
    hybrid_results = score_pair(hybrid_pred, targets, pred_seconds, interval, scoring)
    re_slices = by_nominal_re(hybrid_pred, targets, windows, pred_seconds, interval, scoring)

    assessment = assess_candidate(
        hybrid_results,
        re_slices,
        config["evaluation"]["acceptance_gates"],
    )

    report = {
        "experiment": "E026",
        "git_commit": git_head(),
        "config_sha256": sha256_file(CONFIG),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "raw_prediction_results": raw_results,
        "persist_first_results": hybrid_results,
        "reynolds_slices": re_slices,
        "assessment": assessment,
    }

    out_path = args.artifacts / "evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== EXPERIMENT E026 MFDR RESIDUAL CNO RESULTS ===")
    print(f"Rel-L2 Score: {assessment['rel_l2_score']:.4f} (min {config['evaluation']['acceptance_gates']['rel_l2_score_min']})")
    print(f"MVPE Score:   {assessment['mvpe_score']:.4f} (min {config['evaluation']['acceptance_gates']['mvpe_score_min']})")
    print(f"TKE Score:    {assessment['tke_score']:.4f} (min {config['evaluation']['acceptance_gates']['tke_score_min']})")
    print(f"SPS Score:    {assessment['sps_score']:.4f} (min {config['evaluation']['acceptance_gates']['sps_score_min']})")
    print(f"Assessment: {json.dumps(assessment, indent=2)}")


if __name__ == "__main__":
    main()
