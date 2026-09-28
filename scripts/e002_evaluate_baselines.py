#!/usr/bin/env python3
"""Run E002: frozen trajectory split and released-baseline evaluation.

Large arrays and per-model reports are written under the ignored
``artifacts/e002_validation/`` directory. Model construction and warm-up are
excluded from timing, matching the public evaluator's timing policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any, Callable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
DATA = PROJECT / "RealPDE-Competition-Data"
CONFIG_PATH = REPO / "configs" / "e002_split.json"
DEFAULT_ARTIFACTS = REPO / "artifacts" / "e002_validation"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.validation import (  # noqa: E402
    audit_split,
    build_validation_arrays,
    load_config,
    score_slices,
)


MEAN_IN = np.array([0.154960856, -0.000513992854, 0.0], dtype=np.float32)
STD_IN = np.array([0.0968056545, 0.015960684, 1.0], dtype=np.float32)
MEAN_TGT = np.array([0.154962569, -0.000517793698, 0.0], dtype=np.float32)
STD_TGT = np.array([0.0968104079, 0.0159636438, 1.0], dtype=np.float32)

MODEL_SPECS = {
    "sim_real_cno": DATA / "baseline_checkpoints" / "sim_real_ft" / "sim_real_cno.pth",
    "sim_real_fno_fp16": DATA / "baseline_checkpoints" / "sim_real_ft" / "sim_real_fno_fp16.pth",
    "sim_real_transolver": DATA / "baseline_checkpoints" / "sim_real_ft" / "sim_real_transolver.pth",
}
DEFAULT_MODELS = ["persistence", *MODEL_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, default=DATA / "train_real")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-windows", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace existing per-model predictions/results")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def prepare_data(
    config: dict[str, Any], data_root: Path, artifacts: Path, rebuild: bool
) -> tuple[dict[str, Any], Path, Path, list[dict[str, Any]]]:
    manifest = audit_split(data_root, config)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    arrays = artifacts / "arrays"
    input_path = arrays / "inputs.npy"
    target_path = arrays / "targets.npy"
    windows_path = arrays / "windows.json"
    if rebuild or not (input_path.is_file() and target_path.is_file() and windows_path.is_file()):
        input_path, target_path, windows = build_validation_arrays(data_root, manifest, arrays)
    else:
        windows = json.loads(windows_path.read_text(encoding="utf-8"))
    inputs = np.load(input_path, mmap_mode="r")
    targets = np.load(target_path, mmap_mode="r")
    expected = int(manifest["counts"]["validation_windows"])
    shape = (expected, 20, 32, 64, 3)
    if inputs.shape != shape or targets.shape != shape or len(windows) != expected:
        raise ValueError(f"cached validation arrays do not match the manifest: {inputs.shape}/{targets.shape}")
    if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(targets))):
        raise ValueError("validation arrays contain non-finite values")
    if np.any(inputs[..., 2] != 0.0) or np.any(targets[..., 2] != 0.0):
        raise ValueError("real pressure channel must be exactly zero")
    return manifest, input_path, target_path, windows


def _time_batches(
    inputs: np.ndarray,
    output_path: Path,
    batch_size: int,
    predict: Callable[[np.ndarray], np.ndarray],
    synchronize: Callable[[], None],
) -> tuple[float, list[float]]:
    n = int(inputs.shape[0])
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(n, 20, 32, 64, 3)
    )
    warmup_input = np.array(inputs[:1], dtype=np.float32, copy=True, order="C")
    _ = predict(warmup_input)
    synchronize()
    batch_times = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        # Input assembly is evaluator work, not participant predict() time. The
        # copy also mirrors the ordinary writable NumPy array given to predict.
        batch_input = np.array(inputs[start:stop], dtype=np.float32, copy=True, order="C")
        synchronize()
        before = time.perf_counter()
        batch = predict(batch_input)
        synchronize()
        elapsed = time.perf_counter() - before
        expected_shape = (stop - start, 20, 32, 64, 3)
        if batch.shape != expected_shape or batch.dtype != np.float32:
            raise ValueError(f"bad prediction contract: {batch.shape}/{batch.dtype}")
        if not np.all(np.isfinite(batch)):
            raise ValueError("prediction contains non-finite values")
        batch[..., 2] = 0.0
        output[start:stop] = batch
        batch_times.append(elapsed)
    output.flush()
    return float(sum(batch_times) / n), batch_times


def run_persistence(inputs: np.ndarray, output_path: Path, batch_size: int) -> dict[str, Any]:
    def predict(batch: np.ndarray) -> np.ndarray:
        result = np.repeat(batch[:, -1:, ...], 20, axis=1).astype(np.float32, copy=False)
        result[..., 2] = 0.0
        return result

    mean_seconds, batch_times = _time_batches(inputs, output_path, batch_size, predict, lambda: None)
    return {
        "kind": "non_learned",
        "validation_status": "leakage_safe",
        "mean_seconds_per_sample": mean_seconds,
        "batch_seconds": batch_times,
        "parameter_count": 0,
        "checkpoint_bytes": 0,
        "checkpoint_sha256": None,
        "peak_gpu_allocated_bytes": 0,
    }


def run_neural(
    name: str,
    inputs: np.ndarray,
    output_path: Path,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    load_baseline = importlib.import_module("load_baseline").load_baseline
    checkpoint = MODEL_SPECS[name]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model, metadata = load_baseline(str(checkpoint), device=device)
    model = model.to(device).eval()
    mean_in = torch.from_numpy(MEAN_IN).to(device)
    std_in = torch.from_numpy(STD_IN).to(device)
    mean_tgt = torch.from_numpy(MEAN_TGT).to(device)
    std_tgt = torch.from_numpy(STD_TGT).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def predict(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            tensor = (tensor - mean_in) / std_in
            result = model(tensor) * std_tgt + mean_tgt
            result[..., 2] = 0.0
            return result.float().cpu().numpy()

    synchronize = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    mean_seconds, batch_times = _time_batches(inputs, output_path, batch_size, predict, synchronize)
    peak_gpu = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return {
        "kind": "released_checkpoint",
        "validation_status": "contaminated_reference_only",
        "contamination_reason": "The released sim_real_ft checkpoint was fine-tuned on the complete public real-data release, including E002 validation trajectories.",
        "normalization": {
            "source": "official submission_example_fno.py real-train Gaussian statistics",
            "mean_in": MEAN_IN.tolist(), "std_in": STD_IN.tolist(),
            "mean_target": MEAN_TGT.tolist(), "std_target": STD_TGT.tolist(),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metadata": metadata,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "mean_seconds_per_sample": mean_seconds,
        "batch_seconds": batch_times,
        "peak_gpu_allocated_bytes": peak_gpu,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    config = load_config(args.config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    manifest, input_path, target_path, windows = prepare_data(
        config, args.data_root, args.artifacts, args.rebuild_windows
    )
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    if args.prepare_only:
        print(f"E002 data prepared under {args.artifacts}")
        return

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    inputs = np.load(input_path, mmap_mode="r")
    targets = np.load(target_path, mmap_mode="r")
    scoring = importlib.import_module("scoring")
    results_dir = args.artifacts / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for name in args.models:
        result_path = results_dir / f"{name}.json"
        prediction_path = results_dir / f"{name}_predictions.npy"
        if result_path.is_file() and prediction_path.is_file() and not args.force:
            results[name] = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"reused {name}", flush=True)
            continue
        print(f"running {name} on {device}", flush=True)
        started = time.perf_counter()
        if name == "persistence":
            run = run_persistence(inputs, prediction_path, args.batch_size)
        else:
            run = run_neural(name, inputs, prediction_path, args.batch_size, device)
        run["batch_size"] = args.batch_size
        prediction = np.load(prediction_path, mmap_mode="r")
        run["metrics"] = score_slices(
            prediction, targets, windows, run["mean_seconds_per_sample"], scoring
        )
        run["wall_seconds_including_scoring"] = time.perf_counter() - started
        run["cumulative_process_peak_rss_kib"] = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        run["prediction_bytes"] = prediction_path.stat().st_size
        run["prediction_sha256"] = sha256_file(prediction_path)
        result_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        results[name] = run
        print(json.dumps({name: run["metrics"]["overall"]["scores"]}, indent=2), flush=True)
        del prediction
        if device == "cuda":
            torch.cuda.empty_cache()

    # Preserve completed model results when a later invocation evaluates only a
    # subset (for example, persistence on CPU followed by neural models on GPU).
    all_results: dict[str, Any] = {}
    for name in DEFAULT_MODELS:
        result_path = results_dir / f"{name}.json"
        if result_path.is_file():
            all_results[name] = json.loads(result_path.read_text(encoding="utf-8"))

    report = {
        "experiment": "E002",
        "status": "complete",
        "git_commit": git_head(),
        "seed": seed,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "device": device,
        "batch_size": args.batch_size,
        "config": str(args.config),
        "config_sha256": config["config_sha256"],
        "data_inventory_sha256": manifest["data_inventory_sha256"],
        "split_counts": manifest["counts"],
        "validation_nominal_re": manifest["validation_nominal_re"],
        "window_policy": manifest["window"],
        "timing_note": "Local timing excludes model construction and one warm-up. RTX 5050 time_score is not comparable to the official A800 leaderboard time_score.",
        "leakage_note": "Only persistence is a clean local validation anchor. Released sim_real_ft weights saw all public real trajectories and are reported solely as contaminated reference diagnostics.",
        "models": all_results,
    }
    report_path = args.artifacts / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"E002 report: {report_path}")


if __name__ == "__main__":
    main()
