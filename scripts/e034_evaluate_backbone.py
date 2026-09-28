#!/usr/bin/env python3
"""Evaluate the fixed-final E034 long-budget backbone once on frozen Fold A.

One-shot: refuses a repeated look. Computes raw and persist-first-4 rows with
default and frozen (0.025, 0.15) bounds, Reynolds slices on the hybrid, and
assesses the pre-registered gates from configs/e034_long_backbone.json.
"""

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
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e034_long_backbone.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e034_long_backbone"
E002_ARRAYS = REPO / "artifacts" / "e002_validation" / "arrays"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from e003_evaluate_cno import timed_prediction  # noqa: E402
from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402
from realpde_t1.validation import persist_first_frames, score_prediction  # noqa: E402
from load_baseline import load_baseline  # noqa: E402


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
        prediction, targets, seconds, scoring, lower=lower, upper=upper
    )
    return {"default": default, "calibrated": calibrated}


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


def assess(config: dict[str, Any], hybrid_metrics: dict[str, Any]) -> dict[str, Any]:
    criteria = config["evaluation"]["success_criteria"]
    overall = hybrid_metrics["overall"]["calibrated"]["scores"]
    reference = config["evaluation"]["reference_rows_persist_first_4_calibrated"]

    beat_e005 = {}
    for metric, floor in criteria["persist_first_4_calibrated_must_beat_e005"].items():
        beat_e005[metric] = {
            "value": overall[metric],
            "reference": floor,
            "delta": overall[metric] - float(floor),
            "passed": bool(overall[metric] > float(floor)),
        }

    nonreg_e027 = {}
    for metric, allowance in criteria[
        "persist_first_4_calibrated_max_regression_vs_e027"
    ].items():
        delta = overall[metric] - float(reference["e027_overall"][metric])
        nonreg_e027[metric] = {
            "value": overall[metric],
            "reference": reference["e027_overall"][metric],
            "delta": delta,
            "allowance": -float(allowance),
            "passed": bool(delta >= -float(allowance)),
        }

    slice_26700 = hybrid_metrics["by_nominal_re"]["26700"]["calibrated"]["scores"]
    re26700 = {}
    for metric, floor in criteria[
        "re_26700_persist_first_4_calibrated_must_beat_e027_slice"
    ].items():
        re26700[metric] = {
            "value": slice_26700[metric],
            "reference": floor,
            "delta": slice_26700[metric] - float(floor),
            "passed": bool(slice_26700[metric] > float(floor)),
        }
    tke_allowance = float(criteria["re_26700_tke_max_regression_vs_e027_slice"])
    delta = slice_26700["tke_score"] - float(reference["e027_re_26700"]["tke_score"])
    re26700["tke_score_nonregression"] = {
        "value": slice_26700["tke_score"],
        "reference": reference["e027_re_26700"]["tke_score"],
        "delta": delta,
        "allowance": -tke_allowance,
        "passed": bool(delta >= -tke_allowance),
    }

    accepted = (
        all(item["passed"] for item in beat_e005.values())
        and all(item["passed"] for item in nonreg_e027.values())
        and all(item["passed"] for item in re26700.values())
    )
    return {
        "strict_hypothesis_accepted": accepted,
        "beat_e005_checks": beat_e005,
        "non_regression_vs_e027_checks": nonreg_e027,
        "re_26700_checks": re26700,
    }


def main() -> None:
    args = parse_args()
    evaluation_path = args.artifacts / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"E034 final validation was already evaluated: {evaluation_path}; refusing a repeated look"
        )
    checkpoint_path = args.artifacts / "final.pth"
    train_report_path = args.artifacts / "train_report.json"
    if not checkpoint_path.is_file() or not train_report_path.is_file():
        raise FileNotFoundError("E034 training artifacts are incomplete")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != "fixed_budget_final":
        raise ValueError("E034 evaluator accepts only the fixed-final checkpoint")
    if checkpoint.get("experiment") != "E034":
        raise ValueError(
            f"checkpoint experiment mismatch: {checkpoint.get('experiment')} != E034"
        )
    validation_field_access = checkpoint["provenance"].get(
        "validation_field_values_accessed",
        checkpoint["provenance"].get("validation_files_accessed"),
    )
    if validation_field_access != []:
        raise ValueError("checkpoint provenance reports validation access during training")
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    interval = config["evaluation"]["interval"]
    k_frames = int(config["evaluation"]["persist_first_frames"])

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.manual_seed(int(train_report["seed"]))
    model, metadata = load_baseline("cno", str(checkpoint_path), device=str(device))
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inputs = np.load(E002_ARRAYS / "inputs.npy", mmap_mode="r")
    targets = np.load(E002_ARRAYS / "targets.npy", mmap_mode="r")
    windows = json.loads((E002_ARRAYS / "windows.json").read_text(encoding="utf-8"))
    if inputs.shape != (357, 20, 32, 64, 3) or targets.shape != inputs.shape:
        raise ValueError(f"unexpected Fold A arrays: {inputs.shape}/{targets.shape}")

    prediction_path = args.artifacts / "predictions.npy"
    mean_seconds, batch_times = timed_prediction(
        model, device, inputs, prediction_path, args.batch_size, checkpoint["normalizer"]
    )
    raw_prediction = np.load(prediction_path, mmap_mode="r")
    scoring = importlib.import_module("scoring")

    raw_row = score_pair(raw_prediction, targets, mean_seconds, interval, scoring)
    hybrid = persist_first_frames(raw_prediction, inputs, k_frames)
    hybrid_row = score_pair(hybrid, targets, mean_seconds, interval, scoring)
    slices = by_nominal_re(hybrid, targets, windows, mean_seconds, interval, scoring)

    metrics = {
        "raw": raw_row,
        "persist_first_4": {**hybrid_row, "overall": hybrid_row},
        "by_nominal_re": slices,
    }
    assessment = assess(config, {
        "overall": hybrid_row,
        "by_nominal_re": slices,
    })

    report = {
        "experiment": "E034",
        "status": "complete",
        "evidence_status": "leakage_safe",
        "git_commit": git_head(),
        "checkpoint_git_commit": checkpoint["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "config": config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "normalizer": checkpoint["normalizer"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "batch_size": args.batch_size,
        "mean_seconds_per_sample": mean_seconds,
        "batch_seconds": batch_times,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "prediction_sha256": sha256_file(prediction_path),
        "metrics": metrics,
        "assessment": assessment,
        "timing_note": "RTX 5050 timing excludes model construction and one warm-up; its time_score is not comparable to the official A800.",
    }
    evaluation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "persist_first_4_calibrated": hybrid_row["calibrated"]["scores"],
        "raw_default": raw_row["default"]["scores"],
        "assessment": assessment,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
