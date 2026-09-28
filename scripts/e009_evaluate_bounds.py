#!/usr/bin/env python3
"""Evaluate the selected E009 bounds once on saved E005 validation predictions."""

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
CONFIG = REPO / "configs" / "e009_oof_uncertainty.json"
CALIBRATION = REPO / "artifacts" / "e009_oof_uncertainty" / "calibration.json"
E005_ROOT = REPO / "artifacts" / "e005_scale_balanced_uv"
ARRAYS = REPO / "artifacts" / "e002_validation" / "arrays"
E002_REPORT = REPO / "artifacts" / "e002_validation" / "report.json"
E006_EVALUATION = REPO / "artifacts" / "e006_uncertainty" / "evaluation.json"
OUTPUT = REPO / "artifacts" / "e009_oof_uncertainty" / "evaluation.json"
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
    rel = float(np.mean(scoring.rel_l2_per_sample(prediction, target, 2)))
    tke = float(np.mean(scoring.tke_rel_l2_per_sample(prediction, target, 2)))
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
        raise FileExistsError(f"E009 was already evaluated: {OUTPUT}")
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if calibration["status"] != "calibration_complete_not_yet_validated":
        raise ValueError("E009 calibration is not a fixed complete run")
    if calibration["config_sha256"] != hashlib.sha256(config_raw).hexdigest():
        raise ValueError("E009 calibration/config hash mismatch")
    if not calibration["training_side_acceptance"]["oof_calibration_gain_pass"]:
        raise ValueError("E009 failed its training-side OOF gain gate")
    e005 = json.loads((E005_ROOT / "evaluation.json").read_text(encoding="utf-8"))
    e002 = json.loads(E002_REPORT.read_text(encoding="utf-8"))
    e006 = json.loads(E006_EVALUATION.read_text(encoding="utf-8"))
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
            raise ValueError(f"E009 changed E005 point metric {name}")
    point_hash_unchanged = sha256_file(E005_ROOT / "predictions.npy") == e005["prediction_sha256"]
    persistence_sps = e002["models"]["persistence"]["metrics"]["overall"]["scores"]["sps_score"]
    e006_sps = e006["overall"]["scores"]["sps_score"]
    checks = {
        "oof_calibration_gain_pass": calibration["training_side_acceptance"]["oof_calibration_gain_pass"],
        "point_prediction_hash_unchanged": point_hash_unchanged,
        "selected_differs_from_e006": calibration["training_side_acceptance"]["selected_differs_from_e006"],
        "e006_interval_oof_drop_pass": calibration["training_side_acceptance"]["e006_interval_oof_drop_pass"],
    }
    checks["strict_hypothesis_accepted"] = bool(
        checks["selected_differs_from_e006"] or checks["e006_interval_oof_drop_pass"]
    )
    report = {
        "experiment": "E009",
        "status": "complete",
        "strict_hypothesis_accepted": checks["strict_hypothesis_accepted"],
        "promotion": "evidence_only",
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "calibration_sha256": sha256_file(CALIBRATION),
        "e005_prediction_sha256": e005["prediction_sha256"],
        "selected": selected,
        "e006_selected": e006["selected"],
        "bounds_seconds": bounds_seconds,
        "bounds_seconds_per_sample": bounds_seconds_per_sample,
        "point_seconds_per_sample": e005["mean_seconds_per_sample"],
        "estimated_total_seconds_per_sample": total_seconds_per_sample,
        "overall": overall,
        "by_stratum": by_stratum,
        "by_nominal_re": by_re,
        "comparators": {
            "persistence_sps": persistence_sps,
            "e006_validation_sps": e006_sps,
            "e009_minus_e006_sps": overall["scores"]["sps_score"] - e006_sps,
        },
        "checks": checks,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "timing_note": "Point timing is the saved E005 RTX 5050 measurement; bound-construction timing is measured separately on the saved predictions and added.",
        "limitations": config["limitations"],
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "strict_hypothesis_accepted": report["strict_hypothesis_accepted"],
        "overall_scores": overall["scores"],
        "comparators": report["comparators"],
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
