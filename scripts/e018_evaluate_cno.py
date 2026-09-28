#!/usr/bin/env python3
"""Evaluate the fixed-final E018 checkpoint once on Fold A."""

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
CONFIG = REPO / "configs" / "e018_residual_adapter.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e018_residual_adapter"
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


def assess(config: dict[str, Any], persist_metrics: dict[str, Any]) -> dict[str, Any]:
    criteria = config["evaluation"]["success_criteria"]
    reference = config["evaluation"]["e013_persist_first_4"]
    persist_scores = persist_metrics["overall"]["calibrated"]["scores"]
    beat_checks = {}
    beat_deltas = {}
    for metric in criteria["persist_first_4_must_beat_e013"]:
        delta = persist_scores[metric] - float(reference[metric])
        beat_deltas[metric] = delta
        beat_checks[metric] = delta > 0
    tke_delta = persist_scores["tke_score"] - float(reference["tke_score"])
    tke_check = tke_delta >= -float(criteria["persist_first_4_tke_max_regression_vs_e013"])
    re3750 = persist_metrics["by_nominal_re"]["3750"]["calibrated"]["scores"]
    re3750_ref = reference["by_nominal_re"]["3750"]
    re3750_checks = {}
    re3750_deltas = {}
    for metric, maximum_regression in criteria[
        "re_3750_persist_first_4_max_regression_vs_e013"
    ].items():
        delta = re3750[metric] - float(re3750_ref[metric])
        re3750_deltas[metric] = delta
        re3750_checks[metric] = delta >= -float(maximum_regression)
    accepted = all(beat_checks.values()) and tke_check and all(re3750_checks.values())
    return {
        "strict_hypothesis_accepted": accepted,
        "persist_first_4_deltas_vs_e013": {
            **beat_deltas,
            "tke_score": tke_delta,
            "sps_score": persist_scores["sps_score"] - float(reference["calibrated_sps_score"]),
        },
        "persist_first_4_beat_checks": beat_checks,
        "persist_first_4_tke_check": tke_check,
        "re_3750_deltas_vs_e013": re3750_deltas,
        "re_3750_checks": re3750_checks,
    }


def main() -> None:
    args = parse_args()
    evaluation_path = args.artifacts / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"E018 final validation was already evaluated: {evaluation_path}; refusing a repeated look"
        )
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    checkpoint_path = args.artifacts / "final.pth"
    train_report_path = args.artifacts / "train_report.json"
    if not checkpoint_path.is_file() or not train_report_path.is_file():
        raise FileNotFoundError("E018 training artifacts are incomplete")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != "fixed_budget_final" or checkpoint.get("experiment") != "E018":
        raise ValueError("E018 evaluator accepts only the fixed-final E018 checkpoint")
    if checkpoint["provenance"].get("validation_field_values_accessed") != []:
        raise ValueError("checkpoint provenance reports validation access during training")
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.manual_seed(int(train_report["seed"]))
    if checkpoint.get("model_kind") != "residual_cno":
        raise ValueError("E018 evaluator accepts only a residual_cno checkpoint")
    model = load_residual_cno(checkpoint, device).eval()
    metadata = {
        "model_kind": "residual_cno",
        "adapter_config": checkpoint.get("adapter_config"),
        "trainable_parameter_count": train_report.get("trainable_parameter_count"),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    arrays = REPO / config["evaluation"]["arrays"]
    inputs = np.load(arrays / "inputs.npy", mmap_mode="r")
    targets = np.load(arrays / "targets.npy", mmap_mode="r")
    windows = json.loads((arrays / "windows.json").read_text(encoding="utf-8"))
    if inputs.shape != (357, 20, 32, 64, 3) or targets.shape != inputs.shape:
        raise ValueError(f"unexpected Fold A arrays: {inputs.shape}/{targets.shape}")

    prediction_path = args.artifacts / "predictions.npy"
    mean_seconds, batch_times = timed_prediction(
        model, device, inputs, prediction_path, args.batch_size, checkpoint["normalizer"]
    )
    prediction = np.load(prediction_path)
    persist = persist_first_frames(
        prediction, np.asarray(inputs), int(config["evaluation"]["persist_first_frames"])
    )
    persist_path = args.artifacts / "persist_first_4.npy"
    np.save(persist_path, persist, allow_pickle=False)

    scoring = importlib.import_module("scoring")
    interval = config["evaluation"]["interval"]
    raw_metrics = {
        "overall": score_pair(prediction, targets, mean_seconds, interval, scoring),
        "by_nominal_re": by_nominal_re(
            prediction, targets, windows, mean_seconds, interval, scoring
        ),
    }
    persist_metrics = {
        "overall": score_pair(persist, targets, mean_seconds, interval, scoring),
        "by_nominal_re": by_nominal_re(
            persist, targets, windows, mean_seconds, interval, scoring
        ),
    }
    assessment = assess(config, persist_metrics)
    report = {
        "experiment": "E018",
        "status": "complete",
        "evidence_status": "leakage_safe",
        "promotion": "evidence_only",
        "strict_hypothesis_accepted": assessment["strict_hypothesis_accepted"],
        "git_commit": git_head(),
        "checkpoint_git_commit": checkpoint["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "split_config_sha256": checkpoint["provenance"]["split_config_sha256"],
        "data_inventory_sha256": checkpoint["provenance"]["data_inventory_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
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
        "persist_first_4_sha256": sha256_file(persist_path),
        "raw": raw_metrics,
        "persist_first_4": persist_metrics,
        "assessment": assessment,
        "timing_note": "RTX 5050 timing excludes model construction and one warm-up; its time_score is not comparable to the official A800.",
        "limitations": config["limitations"],
    }
    temporary = evaluation_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(evaluation_path)
    persist_scores = persist_metrics["overall"]["calibrated"]["scores"]
    print(json.dumps({
        "status": report["status"],
        "strict_hypothesis_accepted": report["strict_hypothesis_accepted"],
        "raw_overall": raw_metrics["overall"]["calibrated"]["scores"],
        "persist_first_4_overall": persist_scores,
        "assessment": assessment,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
