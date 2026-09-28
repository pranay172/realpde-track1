#!/usr/bin/env python3
"""Evaluate the selected E006 bounds once on saved E005 validation predictions."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e006_uncertainty_calibration.json"
CALIBRATION = REPO / "artifacts" / "e006_uncertainty" / "calibration.json"
E005_ROOT = REPO / "artifacts" / "e005_scale_balanced_uv"
ARRAYS = REPO / "artifacts" / "e002_validation" / "arrays"
E002_REPORT = REPO / "artifacts" / "e002_validation" / "report.json"
OUTPUT = REPO / "artifacts" / "e006_uncertainty" / "evaluation.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_with_bounds(
    prediction: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seconds_per_sample: float,
    scoring: Any,
) -> dict[str, Any]:
    channels = scoring.measured_channels(target)
    rel = float(np.mean(scoring.rel_l2_per_sample(prediction, target, channels)))
    tke = float(np.mean(scoring.tke_rel_l2_per_sample(prediction, target, channels)))
    mvpe = float(np.mean(scoring.mvpe_rel_l2_per_sample(prediction, target)))
    sps, coverage = scoring.aggregate_sps(
        prediction, target, channels, lower=lower, upper=upper
    )
    return {
        "n_samples": int(len(prediction)),
        "raw": {
            "relative_l2": rel,
            "tke_relative_l2": tke,
            "mvpe_relative_l2": mvpe,
            "sps_aggregate": float(sps),
            "interval_coverage": float(coverage),
        },
        "scores": {
            "rel_l2_score": scoring.score_error(rel),
            "tke_score": scoring.score_error(tke),
            "mvpe_score": scoring.score_error(mvpe),
            "time_score": scoring.score_time(seconds_per_sample),
            "sps_score": scoring.score_sps(sps),
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"E006 was already evaluated: {OUTPUT}")
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if calibration["status"] != "calibration_complete_not_yet_validated":
        raise ValueError("E006 calibration is not a fixed complete run")
    if calibration["config_sha256"] != hashlib.sha256(config_raw).hexdigest():
        raise ValueError("E006 calibration/config hash mismatch")
    if not all(calibration["training_side_acceptance"].values()):
        raise ValueError("E006 failed its training-side acceptance gate")
    e005 = json.loads((E005_ROOT / "evaluation.json").read_text(encoding="utf-8"))
    e002 = json.loads(E002_REPORT.read_text(encoding="utf-8"))
    prediction = np.load(E005_ROOT / "predictions.npy", mmap_mode="r")
    target = np.load(ARRAYS / "targets.npy", mmap_mode="r")
    windows = json.loads((ARRAYS / "windows.json").read_text(encoding="utf-8"))
    if sha256_file(E005_ROOT / "predictions.npy") != e005["prediction_sha256"]:
        raise ValueError("E005 saved prediction hash mismatch")
    selected = calibration["selected"]
    started = time.perf_counter()
    lower, upper = interval_bounds_numpy(
        prediction,
        float(selected["alpha"]),
        float(selected["beta"]),
        float(config["interval"]["sigma_global"]),
    )
    bounds_seconds = time.perf_counter() - started
    bounds_seconds_per_sample = bounds_seconds / len(prediction)
    total_seconds_per_sample = e005["mean_seconds_per_sample"] + bounds_seconds_per_sample
    scoring = importlib.import_module("scoring")
    overall = score_with_bounds(
        prediction, target, lower, upper, total_seconds_per_sample, scoring
    )
    by_stratum = {}
    for group in sorted({row["group"] for row in windows}):
        indices = np.asarray([row["index"] for row in windows if row["group"] == group])
        by_stratum[group] = score_with_bounds(
            prediction[indices], target[indices], lower[indices], upper[indices],
            total_seconds_per_sample, scoring
        )
    by_re = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        indices = np.asarray([
            row["index"] for row in windows if int(row["nominal_re"]) == re_value
        ])
        by_re[str(re_value)] = score_with_bounds(
            prediction[indices], target[indices], lower[indices], upper[indices],
            total_seconds_per_sample, scoring
        )
    for name in ("relative_l2", "tke_relative_l2", "mvpe_relative_l2"):
        if not np.isclose(
            overall["raw"][name], e005["metrics"]["overall"]["raw"][name], atol=0, rtol=0
        ):
            raise ValueError(f"E006 changed E005 point metric {name}")
    point_hash_unchanged = sha256_file(E005_ROOT / "predictions.npy") == e005["prediction_sha256"]
    persistence_sps = e002["models"]["persistence"]["metrics"]["overall"]["scores"]["sps_score"]
    time_regression = e005["metrics"]["overall"]["scores"]["time_score"] - overall["scores"]["time_score"]
    checks = {
        "calibration_gain_pass": calibration["training_side_acceptance"]["calibration_gain_pass"],
        "audit_gain_pass": calibration["training_side_acceptance"]["audit_gain_pass"],
        "validation_sps_exceeds_persistence": overall["scores"]["sps_score"] > persistence_sps,
        "local_time_regression_within_limit": time_regression <= float(
            config["final_evaluation"]["maximum_local_time_score_regression"]
        ),
        "point_prediction_hash_unchanged": point_hash_unchanged,
    }
    report = {
        "experiment": "E006",
        "status": "complete",
        "strict_hypothesis_accepted": all(checks.values()),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "calibration_sha256": sha256_file(CALIBRATION),
        "e005_prediction_sha256": e005["prediction_sha256"],
        "selected": selected,
        "bounds_seconds": bounds_seconds,
        "bounds_seconds_per_sample": bounds_seconds_per_sample,
        "point_seconds_per_sample": e005["mean_seconds_per_sample"],
        "estimated_total_seconds_per_sample": total_seconds_per_sample,
        "time_score_regression_vs_e005": time_regression,
        "overall": overall,
        "by_stratum": by_stratum,
        "by_nominal_re": by_re,
        "checks": checks,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "timing_note": "Point timing is the saved E005 RTX 5050 measurement; bound-construction timing is measured separately on the saved predictions and added. Exact integrated submission timing remains for the container gate.",
        "limitations": config["limitations"],
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
