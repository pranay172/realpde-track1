#!/usr/bin/env python3
"""Evaluate the E025 Fold B shared-backbone ensemble once."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e025_fold_b_ensemble.json"
OUTPUT_DIR = REPO / "artifacts" / "e025_ensemble"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from e023_evaluate_ensemble import run_ensemble_inference  # noqa: E402
from realpde_t1.ensemble import (  # noqa: E402
    build_ensemble_bundle,
    load_shared_backbone_ensemble,
)
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


def assess(config: dict, persist_scores: dict, by_re: dict) -> dict:
    criteria = config["success_criteria"]
    reference = config["e020_persist_first_4"]
    beat_checks = {}
    beat_deltas = {}
    for metric in criteria["persist_first_4_must_beat_e020"]:
        delta = persist_scores[metric] - float(reference[metric])
        beat_deltas[metric] = delta
        beat_checks[metric] = delta > 0
    tke_delta = persist_scores["tke_score"] - float(reference["tke_score"])
    tke_check = tke_delta >= -float(criteria["persist_first_4_tke_max_regression_vs_e020"])
    re6300 = by_re["6300"]["scores"]
    re6300_ref = reference["by_nominal_re"]["6300"]
    re6300_checks = {}
    re6300_deltas = {}
    for metric, maximum_regression in criteria[
        "re_6300_persist_first_4_max_regression_vs_e020"
    ].items():
        delta = re6300[metric] - float(re6300_ref[metric])
        re6300_deltas[metric] = delta
        re6300_checks[metric] = delta >= -float(maximum_regression)
    accepted = all(beat_checks.values()) and tke_check and all(re6300_checks.values())
    return {
        "strict_hypothesis_accepted": accepted,
        "persist_first_4_deltas_vs_e020": {
            **beat_deltas,
            "tke_score": tke_delta,
            "sps_score": persist_scores["sps_score"]
            - float(reference["calibrated_sps_score"]),
        },
        "persist_first_4_beat_checks": beat_checks,
        "persist_first_4_tke_check": tke_check,
        "re_6300_deltas_vs_e020": re6300_deltas,
        "re_6300_checks": re6300_checks,
    }


def main() -> None:
    args = parse_args()
    evaluation_path = OUTPUT_DIR / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"E025 final validation was already evaluated: {evaluation_path}; refusing a repeated look"
        )
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    member_paths = [REPO / row["checkpoint"] for row in config["members"]]
    missing = [str(path) for path in member_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing E025 member checkpoints:\n" + "\n".join(missing))

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False) for path in member_paths
    ]
    bundle = build_ensemble_bundle(
        checkpoints, source_sha256=[sha256_file(path) for path in member_paths]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = OUTPUT_DIR / "ensemble_bundle.pth"
    torch.save(bundle, bundle_path)

    ensemble = load_shared_backbone_ensemble(checkpoints, device).eval()
    normalizer = json.loads((REPO / config["normalizer"]).read_text(encoding="utf-8"))
    arrays = REPO / config["arrays"]
    inputs = np.load(arrays / "inputs.npy")
    targets = np.load(arrays / "targets.npy")
    windows = json.loads((arrays / "windows.json").read_text(encoding="utf-8"))
    if inputs.shape != (376, 20, 32, 64, 3) or targets.shape != inputs.shape:
        raise ValueError(f"unexpected Fold B arrays: {inputs.shape}/{targets.shape}")

    preds, _, mean_seconds = run_ensemble_inference(
        ensemble, inputs, normalizer, args.batch_size, device
    )
    scoring = importlib.import_module("scoring")
    interval = config["interval"]
    default = score_prediction(preds, targets, mean_seconds, scoring)
    hybrid = persist_first_frames(
        preds, inputs, int(config["persist_first_frames"])
    )
    lower, upper = interval_bounds_numpy(
        hybrid,
        float(interval["alpha"]),
        float(interval["beta"]),
        float(interval["sigma_global"]),
    )
    calibrated = score_prediction(
        hybrid, targets, mean_seconds, scoring, lower=lower, upper=upper
    )
    by_re = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        indices = np.asarray(
            [int(row["index"]) for row in windows if int(row["nominal_re"]) == re_value]
        )
        by_re[str(re_value)] = score_prediction(
            hybrid[indices],
            targets[indices],
            mean_seconds,
            scoring,
            lower=lower[indices],
            upper=upper[indices],
        )
    assessment = assess(config, calibrated["scores"], by_re)
    report = {
        "experiment": "E025",
        "status": "complete",
        "evidence_status": "leakage_safe_fold_b",
        "promotion": "evidence_only",
        "strict_hypothesis_accepted": assessment["strict_hypothesis_accepted"],
        "git_commit": git_head(),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "member_checkpoint_sha256": [sha256_file(path) for path in member_paths],
        "bundle_sha256": sha256_file(bundle_path),
        "device": str(device),
        "mean_seconds_per_sample": mean_seconds,
        "n_windows": int(len(inputs)),
        "raw": default,
        "persist_first_4": {"overall": calibrated, "by_nominal_re": by_re},
        "assessment": assessment,
        "uncertainty": "frozen_e006_interval_not_learned_log_variance",
        "limitations": config["limitations"],
    }
    evaluation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "strict_hypothesis_accepted": report["strict_hypothesis_accepted"],
                "persist_first_4_overall": calibrated["scores"],
                "assessment": assessment,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
