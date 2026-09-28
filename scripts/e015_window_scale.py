#!/usr/bin/env python3
"""Evaluate frozen E005 CNO with per-window scale transfer on Fold A."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e015_window_scale.json"
OUTPUT = REPO / "artifacts" / "e015_window_scale" / "evaluation.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.scaling import apply_window_scale, window_scale_ratios  # noqa: E402
from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402
from realpde_t1.validation import persist_first_frames, score_prediction  # noqa: E402
from load_baseline import load_baseline  # noqa: E402


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


def score_both(prediction: np.ndarray, targets: np.ndarray, seconds: float, interval: dict, scoring) -> dict:
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
    return {"default": default, "calibrated": calibrated, "seconds_per_sample": seconds}


def predict_scaled(
    model: torch.nn.Module,
    device: torch.device,
    inputs: np.ndarray,
    normalizer: dict,
    ratios: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    mean_input = torch.tensor(normalizer["mean_input"], dtype=torch.float32, device=device)
    std_input = torch.tensor(normalizer["std_input"], dtype=torch.float32, device=device)
    mean_target = torch.tensor(normalizer["mean_target"], dtype=torch.float32, device=device)
    std_target = torch.tensor(normalizer["std_target"], dtype=torch.float32, device=device)
    scaled = apply_window_scale(np.asarray(inputs), ratios, invert=True)
    output = np.empty_like(inputs, dtype=np.float32)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    warmup = torch.from_numpy(np.ascontiguousarray(scaled[:1])).to(device)
    with torch.inference_mode():
        _ = model((warmup - mean_input) / std_input)
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            batch = torch.from_numpy(np.ascontiguousarray(scaled[start:stop])).to(device)
            result = model((batch - mean_input) / std_input)
            result = result * std_target + mean_target
            result[..., 2] = 0.0
            output[start:stop] = result.float().cpu().numpy()
    synchronize()
    seconds = (time.perf_counter() - started) / len(inputs)
    return apply_window_scale(output, ratios, invert=False), seconds


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"E015 already evaluated: {OUTPUT}")
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    arrays = REPO / config["evaluation"]["arrays"]
    inputs = np.load(arrays / "inputs.npy", mmap_mode="r")
    targets = np.load(arrays / "targets.npy", mmap_mode="r")
    windows = json.loads((arrays / "windows.json").read_text(encoding="utf-8"))
    normalizer = json.loads((REPO / config["normalizer"]).read_text(encoding="utf-8"))
    checkpoint = REPO / config["point_model"]
    e005 = np.load(REPO / config["saved_e005_predictions"])
    device_name = config["evaluation"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)

    ratios = window_scale_ratios(
        np.asarray(inputs),
        np.asarray(normalizer["std_input"]),
        measured_channels=int(config["scaling"]["measured_channels"]),
        min_std=float(config["scaling"]["min_std"]),
    )
    model, metadata = load_baseline("cno", str(checkpoint), device=str(device))
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    scaled_pred, seconds = predict_scaled(
        model, device, inputs, normalizer, ratios, int(config["evaluation"]["batch_size"])
    )
    k = int(config["persist_first_frames"])
    scoring = importlib.import_module("scoring")
    variants = {
        "e005": score_both(e005, np.asarray(targets), 0.06682714105883468, config["interval"], scoring),
        "e013_persist_first_4": score_both(
            persist_first_frames(e005, np.asarray(inputs), k),
            np.asarray(targets),
            0.06682714105883468,
            config["interval"],
            scoring,
        ),
        "window_scale": score_both(
            scaled_pred, np.asarray(targets), seconds, config["interval"], scoring
        ),
        "window_scale_persist_first_4": score_both(
            persist_first_frames(scaled_pred, np.asarray(inputs), k),
            np.asarray(targets),
            seconds,
            config["interval"],
            scoring,
        ),
    }
    e013 = variants["e013_persist_first_4"]["default"]["scores"]
    candidate = variants["window_scale_persist_first_4"]["default"]["scores"]
    checks = {
        metric: candidate[metric] > e013[metric]
        for metric in config["success_criteria"]["must_beat_e013_persist_first_4"]
    }
    tke_delta = candidate["tke_score"] - e013["tke_score"]
    checks["tke_within_allowance"] = tke_delta >= -float(
        config["success_criteria"]["tke_max_regression_vs_e013"]
    )
    by_re = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        idx = np.asarray([row["index"] for row in windows if int(row["nominal_re"]) == re_value])
        by_re[str(re_value)] = {
            "mean_u_ratio": float(ratios[idx, 0].mean()),
            "mean_v_ratio": float(ratios[idx, 1].mean()),
            "e013": score_prediction(
                persist_first_frames(e005[idx], np.asarray(inputs)[idx], k),
                np.asarray(targets)[idx],
                0.06682714105883468,
                scoring,
            )["scores"],
            "window_scale_persist_first_4": score_prediction(
                persist_first_frames(scaled_pred[idx], np.asarray(inputs)[idx], k),
                np.asarray(targets)[idx],
                seconds,
                scoring,
            )["scores"],
        }
    report = {
        "experiment": "E015",
        "status": "complete",
        "promotion": "evidence_only",
        "strict_hypothesis_accepted": all(checks.values()),
        "git_commit": git_head(),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metadata": metadata,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "mean_u_ratio": float(ratios[:, 0].mean()),
        "mean_v_ratio": float(ratios[:, 1].mean()),
        "ratio_u_min_max": [float(ratios[:, 0].min()), float(ratios[:, 0].max())],
        "variants": variants,
        "by_nominal_re": by_re,
        "checks": checks,
        "tke_delta_vs_e013": tke_delta,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "limitations": config["limitations"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT.parent / "ratios.npy", ratios, allow_pickle=False)
    np.save(OUTPUT.parent / "predictions.npy", scaled_pred, allow_pickle=False)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "strict_hypothesis_accepted": report["strict_hypothesis_accepted"],
        "checks": checks,
        "e013": e013,
        "window_scale_persist_first_4": candidate,
        "mean_ratios": [report["mean_u_ratio"], report["mean_v_ratio"]],
    }, indent=2))


if __name__ == "__main__":
    main()
