#!/usr/bin/env python3
"""Evaluate a fixed-final CNO checkpoint exactly once on frozen E002."""

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
from typing import Callable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e003_cno_finetune"
E002_ARRAYS = REPO / "artifacts" / "e002_validation" / "arrays"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.validation import score_prediction, score_slices  # noqa: E402
from load_baseline import load_baseline  # noqa: E402


def parse_args(default_artifacts: Path = DEFAULT_ARTIFACTS) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=default_artifacts)
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


def timed_prediction(
    model: torch.nn.Module,
    device: torch.device,
    inputs: np.ndarray,
    destination: Path,
    batch_size: int,
    normalizer: dict,
) -> tuple[float, list[float]]:
    mean_input = torch.tensor(normalizer["mean_input"], dtype=torch.float32, device=device)
    std_input = torch.tensor(normalizer["std_input"], dtype=torch.float32, device=device)
    mean_target = torch.tensor(normalizer["mean_target"], dtype=torch.float32, device=device)
    std_target = torch.tensor(normalizer["std_target"], dtype=torch.float32, device=device)

    def predict(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(batch).to(device)
            result = model((tensor - mean_input) / std_input)
            result = result * std_target + mean_target
            result[..., 2] = 0.0
            return result.float().cpu().numpy()

    synchronize: Callable[[], None]
    synchronize = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    warmup = np.array(inputs[:1], dtype=np.float32, copy=True, order="C")
    _ = predict(warmup)
    synchronize()
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=inputs.shape
    )
    batch_times: list[float] = []
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        batch = np.array(inputs[start:stop], dtype=np.float32, copy=True, order="C")
        synchronize()
        before = time.perf_counter()
        prediction = predict(batch)
        synchronize()
        batch_times.append(time.perf_counter() - before)
        if prediction.shape != output[start:stop].shape or prediction.dtype != np.float32:
            raise ValueError(f"prediction contract failure: {prediction.shape}/{prediction.dtype}")
        if not np.all(np.isfinite(prediction)):
            raise ValueError("prediction contains non-finite values")
        output[start:stop] = prediction
    output.flush()
    return float(sum(batch_times) / len(inputs)), batch_times


def main(
    default_artifacts: Path = DEFAULT_ARTIFACTS,
    expected_experiment: str = "E003",
) -> None:
    args = parse_args(default_artifacts)
    evaluation_path = args.artifacts / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"{expected_experiment} final validation was already evaluated: {evaluation_path}; refusing a repeated look"
        )
    checkpoint_path = args.artifacts / "final.pth"
    train_report_path = args.artifacts / "train_report.json"
    if not checkpoint_path.is_file() or not train_report_path.is_file():
        raise FileNotFoundError(f"{expected_experiment} training artifacts are incomplete")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != "fixed_budget_final":
        raise ValueError(f"{expected_experiment} evaluator accepts only the fixed-final checkpoint")
    if checkpoint.get("experiment") != expected_experiment:
        raise ValueError(
            f"checkpoint experiment mismatch: {checkpoint.get('experiment')} != {expected_experiment}"
        )
    validation_field_access = checkpoint["provenance"].get(
        "validation_field_values_accessed",
        checkpoint["provenance"].get("validation_files_accessed"),
    )
    if validation_field_access != []:
        raise ValueError("checkpoint provenance reports validation access during training")
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))

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

    input_path = E002_ARRAYS / "inputs.npy"
    target_path = E002_ARRAYS / "targets.npy"
    windows_path = E002_ARRAYS / "windows.json"
    inputs = np.load(input_path, mmap_mode="r")
    targets = np.load(target_path, mmap_mode="r")
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    if inputs.shape != (357, 20, 32, 64, 3) or targets.shape != inputs.shape:
        raise ValueError(f"unexpected E002 arrays: {inputs.shape}/{targets.shape}")
    prediction_path = args.artifacts / "predictions.npy"
    mean_seconds, batch_times = timed_prediction(
        model, device, inputs, prediction_path, args.batch_size, checkpoint["normalizer"]
    )
    prediction = np.load(prediction_path, mmap_mode="r")
    scoring = importlib.import_module("scoring")
    metrics = score_slices(prediction, targets, windows, mean_seconds, scoring)
    metrics_by_nominal_re = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        indices = np.asarray([
            int(row["index"]) for row in windows if int(row["nominal_re"]) == re_value
        ])
        metrics_by_nominal_re[str(re_value)] = score_prediction(
            prediction[indices], targets[indices], mean_seconds, scoring
        )
    report = {
        "experiment": expected_experiment,
        "status": "complete",
        "evidence_status": "leakage_safe",
        "git_commit": git_head(),
        "checkpoint_git_commit": checkpoint["provenance"]["git_commit"],
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
        "metrics": metrics,
        "metrics_by_nominal_re": metrics_by_nominal_re,
        "timing_note": "RTX 5050 timing excludes model construction and one warm-up; its time_score is not comparable to the official A800.",
    }
    evaluation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
