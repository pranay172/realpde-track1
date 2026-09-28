#!/usr/bin/env python3
"""Zero-shot sim-only FNO on Fold A, with and without persist-first-4."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e014_fno_preflight.json"
OUTPUT = REPO / "artifacts" / "e014_fno_preflight" / "report.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from e003_evaluate_cno import timed_prediction  # noqa: E402
from realpde_t1.training import WindowGeometry, compute_gaussian_stats  # noqa: E402
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


def persistence(inputs: np.ndarray) -> np.ndarray:
    result = np.repeat(inputs[:, -1:, ...], 20, axis=1).astype(np.float32, copy=False)
    result[..., 2] = 0.0
    return result


def as_normalizer(block: dict) -> dict:
    return {
        "mean_input": np.asarray(block["mean_input"], dtype=np.float32).tolist(),
        "std_input": np.asarray(block["std_input"], dtype=np.float32).tolist(),
        "mean_target": np.asarray(block["mean_target"], dtype=np.float32).tolist(),
        "std_target": np.asarray(block["std_target"], dtype=np.float32).tolist(),
    }


def score_variant(
    prediction: np.ndarray,
    targets: np.ndarray,
    seconds: float,
    interval: dict,
    scoring,
) -> dict:
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
    return {
        "default": default,
        "calibrated": calibrated,
        "seconds_per_sample": seconds,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"E014 preflight already exists: {OUTPUT}")
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    arrays = REPO / config["evaluation"]["arrays"]
    inputs = np.load(arrays / "inputs.npy", mmap_mode="r")
    targets = np.load(arrays / "targets.npy", mmap_mode="r")
    checkpoint = (REPO / config["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print("computing train_sim Gaussian statistics", flush=True)
    sim_root = PROJECT / "RealPDE-Competition-Data" / "train_sim"
    sim_files = sorted(path.name for path in sim_root.glob("*.h5"))
    if not sim_files:
        raise FileNotFoundError(f"no simulation trajectories in {sim_root}")
    geometry = WindowGeometry()
    sim_stats = compute_gaussian_stats(sim_root, sim_files, geometry)
    normalizers = {
        "official_real_train": as_normalizer(config["normalizers"]["official_real_train"]),
        "train_sim": sim_stats.as_json(),
    }

    device_name = config["evaluation"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    scoring = importlib.import_module("scoring")
    persist_pred = persistence(np.asarray(inputs))
    persist_metrics = score_prediction(persist_pred, np.asarray(targets), 1e-4, scoring)

    model, metadata = load_baseline("fno", str(checkpoint), device=str(device))
    model = model.to(device).eval()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    k = int(config["persist_first_frames"])
    variants = {}
    for name, normalizer in normalizers.items():
        pred_path = OUTPUT.parent / f"pred_{name}.npy"
        seconds, _ = timed_prediction(
            model,
            device,
            inputs,
            pred_path,
            int(config["evaluation"]["batch_size"]),
            normalizer,
        )
        raw_pred = np.load(pred_path)
        hybrid = persist_first_frames(raw_pred, np.asarray(inputs), k)
        variants[name] = {
            "raw": score_variant(raw_pred, np.asarray(targets), seconds, config["interval"], scoring),
            "persist_first_4": score_variant(
                hybrid, np.asarray(targets), seconds, config["interval"], scoring
            ),
        }
        print(name, json.dumps({
            "raw": variants[name]["raw"]["default"]["scores"],
            "persist_first_4": variants[name]["persist_first_4"]["default"]["scores"],
        }, indent=2), flush=True)

    def tke_mvpe(row: dict) -> float:
        scores = row["persist_first_4"]["default"]["scores"]
        return scores["tke_score"] + scores["mvpe_score"]

    better = max(variants, key=lambda name: tke_mvpe(variants[name]))
    best_scores = variants[better]["persist_first_4"]["default"]["scores"]
    persist_scores = persist_metrics["scores"]
    checks = {
        metric: best_scores[metric] > persist_scores[metric]
        for metric in config["success_criteria"]["better_normalizer_persist_first_4_must_beat_persistence"]
    }
    report = {
        "experiment": "E014",
        "phase": "preflight",
        "status": "complete",
        "promotion": "evidence_only",
        "strict_hypothesis_accepted": all(checks.values()),
        "git_commit": git_head(),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metadata": metadata,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_sim_files": len(sim_files),
        "normalizers": {
            "official_real_train": normalizers["official_real_train"],
            "train_sim": normalizers["train_sim"],
        },
        "persistence": persist_metrics,
        "variants": variants,
        "better_normalizer": better,
        "checks": checks,
        "comparators": {
            "e013_fold_a_persist_first_4": {
                "rel_l2_score": 94.54135815777938,
                "tke_score": 72.25424795631467,
                "mvpe_score": 95.20480086031074,
                "calibrated_sps_score": 38.086005352050464,
            },
            "e003_cno": {
                "rel_l2_score": 92.236,
                "tke_score": 64.009,
                "mvpe_score": 94.219,
            },
        },
        "limitations": config["limitations"],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "strict_hypothesis_accepted": report["strict_hypothesis_accepted"],
        "better_normalizer": better,
        "checks": checks,
        "best_persist_first_4": best_scores,
        "persistence": persist_scores,
    }, indent=2))


if __name__ == "__main__":
    main()
