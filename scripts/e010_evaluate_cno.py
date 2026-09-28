#!/usr/bin/env python3
"""Evaluate the fixed-final E010 checkpoint once on the complementary fold."""

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
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e010_secondary_fold_robustness.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e010_secondary_fold"
ARRAYS = DEFAULT_ARTIFACTS / "arrays"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from e003_evaluate_cno import timed_prediction  # noqa: E402
from realpde_t1.validation import (  # noqa: E402
    audit_split,
    build_validation_arrays,
    load_config,
    score_prediction,
    score_slices,
)
from load_baseline import load_baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
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


def persistence_prediction(inputs: np.ndarray) -> np.ndarray:
    result = np.repeat(inputs[:, -1:, ...], 20, axis=1).astype(np.float32, copy=False)
    result[..., 2] = 0.0
    return result


def by_nominal_re(
    prediction: np.ndarray,
    targets: np.ndarray,
    windows: list[dict[str, Any]],
    seconds: float,
    scoring: Any,
) -> dict[str, Any]:
    metrics = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        indices = np.asarray([
            int(row["index"]) for row in windows if int(row["nominal_re"]) == re_value
        ])
        metrics[str(re_value)] = score_prediction(
            prediction[indices], targets[indices], seconds, scoring
        )
    return metrics


def prepare_arrays() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    split = load_config(REPO / config["split_config"])
    data_root = PROJECT / "RealPDE-Competition-Data" / "train_real"
    manifest = audit_split(data_root, split)
    ARRAYS.mkdir(parents=True, exist_ok=True)
    marker = ARRAYS / "manifest.json"
    if (ARRAYS / "inputs.npy").exists() or (ARRAYS / "targets.npy").exists():
        if not marker.exists():
            raise FileExistsError(f"refusing to reuse incomplete complementary-fold arrays: {ARRAYS}")
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("config_sha256") != split["config_sha256"]:
            raise ValueError("existing complementary-fold arrays do not match the frozen split")
        return saved
    input_path, target_path, windows = build_validation_arrays(data_root, manifest, ARRAYS)
    report = {
        "split_config": config["split_config"],
        "config_sha256": split["config_sha256"],
        "data_inventory_sha256": manifest["data_inventory_sha256"],
        "validation_files": list(split["validation_files"]),
        "validation_windows": int(manifest["counts"]["validation_windows"]),
        "input_path": str(input_path),
        "target_path": str(target_path),
        "windows": len(windows),
        "validation_field_values_accessed": "arrays_materialized_for_final_eval_only",
    }
    marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def assess(
    config: dict[str, Any],
    model_metrics: dict[str, Any],
    persistence_metrics: dict[str, Any],
    model_by_re: dict[str, Any],
    persistence_by_re: dict[str, Any],
) -> dict[str, Any]:
    criteria = config["evaluation"]["success_criteria"]
    overall_checks = {}
    overall_deltas = {}
    for metric in criteria["overall_must_exceed_persistence"]:
        delta = (
            model_metrics["overall"]["scores"][metric]
            - persistence_metrics["overall"]["scores"][metric]
        )
        overall_deltas[metric] = delta
        overall_checks[metric] = delta > 0
    low_checks = {}
    low_deltas = {}
    for metric, maximum_regression in criteria[
        "re_6300_max_score_regression_vs_persistence"
    ].items():
        delta = (
            model_by_re["6300"]["scores"][metric]
            - persistence_by_re["6300"]["scores"][metric]
        )
        low_deltas[metric] = delta
        low_checks[metric] = delta >= -float(maximum_regression)
    checks = {
        "overall_exceeds_persistence": overall_checks,
        "re_6300_within_persistence_allowance": low_checks,
    }
    accepted = all(overall_checks.values()) and all(low_checks.values())
    return {
        "strict_hypothesis_accepted": accepted,
        "overall_deltas_vs_persistence": overall_deltas,
        "re_6300_deltas_vs_persistence": low_deltas,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    array_report = prepare_arrays()
    if args.prepare_only:
        print(json.dumps({"status": "arrays_ready", **array_report}, indent=2))
        return

    evaluation_path = args.artifacts / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"E010 final validation was already evaluated: {evaluation_path}; refusing a repeated look"
        )
    config_raw = CONFIG.read_bytes()
    config = json.loads(config_raw)
    checkpoint_path = args.artifacts / "final.pth"
    train_report_path = args.artifacts / "train_report.json"
    if not checkpoint_path.is_file() or not train_report_path.is_file():
        raise FileNotFoundError("E010 training artifacts are incomplete")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != "fixed_budget_final" or checkpoint.get("experiment") != "E010":
        raise ValueError("E010 evaluator accepts only the fixed-final E010 checkpoint")
    if checkpoint["provenance"].get("validation_field_values_accessed") != []:
        raise ValueError("checkpoint provenance reports validation access during training")
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    split = load_config(REPO / config["split_config"])
    if set(checkpoint["provenance"]["training_files"]) != set(split["training_files"]):
        raise ValueError("E010 checkpoint training files do not match the complementary fold")

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

    inputs = np.load(ARRAYS / "inputs.npy", mmap_mode="r")
    targets = np.load(ARRAYS / "targets.npy", mmap_mode="r")
    windows = json.loads((ARRAYS / "windows.json").read_text(encoding="utf-8"))
    expected = int(array_report["validation_windows"])
    if inputs.shape != (expected, 20, 32, 64, 3) or targets.shape != inputs.shape:
        raise ValueError(f"unexpected complementary-fold arrays: {inputs.shape}/{targets.shape}")

    persistence_path = args.artifacts / "persistence.npy"
    persist_started = time.perf_counter()
    persist = persistence_prediction(np.asarray(inputs))
    persist_seconds = (time.perf_counter() - persist_started) / len(inputs)
    np.save(persistence_path, persist, allow_pickle=False)

    prediction_path = args.artifacts / "predictions.npy"
    mean_seconds, batch_times = timed_prediction(
        model, device, inputs, prediction_path, args.batch_size, checkpoint["normalizer"]
    )
    prediction = np.load(prediction_path, mmap_mode="r")
    scoring = importlib.import_module("scoring")
    model_metrics = score_slices(prediction, targets, windows, mean_seconds, scoring)
    persist_metrics = score_slices(persist, targets, windows, persist_seconds, scoring)
    model_by_re = by_nominal_re(prediction, targets, windows, mean_seconds, scoring)
    persist_by_re = by_nominal_re(persist, targets, windows, persist_seconds, scoring)
    assessment = assess(config, model_metrics, persist_metrics, model_by_re, persist_by_re)
    report = {
        "experiment": "E010",
        "status": "complete",
        "evidence_status": "leakage_safe_complementary_fold",
        "promotion": "evidence_only",
        "strict_hypothesis_accepted": assessment["strict_hypothesis_accepted"],
        "git_commit": git_head(),
        "checkpoint_git_commit": checkpoint["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "split_config_sha256": split["config_sha256"],
        "data_inventory_sha256": array_report["data_inventory_sha256"],
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
        "persistence_sha256": sha256_file(persistence_path),
        "metrics": model_metrics,
        "metrics_by_nominal_re": model_by_re,
        "persistence": persist_metrics,
        "persistence_by_nominal_re": persist_by_re,
        "assessment": assessment,
        "timing_note": "RTX 5050 timing excludes model construction and one warm-up; its time_score is not comparable to the official A800.",
        "limitations": config["limitations"],
    }
    temporary = evaluation_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(evaluation_path)
    print(json.dumps({
        "status": report["status"],
        "strict_hypothesis_accepted": report["strict_hypothesis_accepted"],
        "overall": model_metrics["overall"]["scores"],
        "persistence_overall": persist_metrics["overall"]["scores"],
        "assessment": assessment,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
