#!/usr/bin/env python3
"""Select E009 interval widths from true out-of-fold training residuals."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
DEFAULT_CONFIG = REPO / "configs" / "e009_oof_uncertainty.json"
DEFAULT_CHECKPOINT = REPO / "artifacts" / "e009_oof_uncertainty" / "final.pth"
DEFAULT_OUTPUT = REPO / "artifacts" / "e009_oof_uncertainty" / "calibration.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.training import RealWindowDataset, WindowGeometry  # noqa: E402
from realpde_t1.uncertainty import candidate_interval_sums, finalize_sps  # noqa: E402
from realpde_t1.validation import files_for_reynolds, load_config, parse_trajectory_name  # noqa: E402
from load_baseline import load_baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
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


def accuracy_factors(prediction: np.ndarray, target: np.ndarray, scoring: Any) -> np.ndarray:
    errors = np.stack(
        [
            scoring.rel_l2_per_sample(prediction, target, 2),
            scoring.tke_rel_l2_per_sample(prediction, target, 2),
            scoring.mvpe_rel_l2_per_sample(prediction, target),
        ],
        axis=1,
    )
    return np.asarray(1.0 - errors / (0.5 + errors), dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace E009 calibration: {args.output}")
    raw_config = args.config.read_bytes()
    config = json.loads(raw_config)
    split_path = (REPO / config["split_config"]).resolve()
    split = load_config(split_path)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != "E009" or checkpoint.get("status") != "fixed_budget_final":
        raise ValueError("E009 requires the fixed-final E009 OOF checkpoint")
    if checkpoint["provenance"].get("validation_field_values_accessed") != []:
        raise ValueError("E009 provenance reports validation field access")

    training_nominal_re = {int(value) for value in config["training_nominal_re"]}
    calibration_res = {
        int(value) for value in config["training_side_calibration"]["calibration_nominal_re"]
    }
    if training_nominal_re & calibration_res:
        raise ValueError("OOF training Reynolds groups overlap calibration groups")
    split_training = list(split["training_files"])
    oof_train_files = files_for_reynolds(split_training, sorted(training_nominal_re))
    calibration_files = files_for_reynolds(split_training, sorted(calibration_res))
    checkpoint_files = list(checkpoint["provenance"]["training_files"])
    if set(checkpoint_files) != set(oof_train_files):
        raise ValueError("OOF checkpoint training files do not match the frozen E009 subset")
    if set(checkpoint_files) & set(calibration_files):
        raise ValueError("OOF checkpoint trained on a calibration trajectory")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.manual_seed(int(config["seed"]))
    model, metadata = load_baseline("cno", str(checkpoint_path), device=str(device))
    model = model.to(device).eval()

    inference_files = oof_train_files + calibration_files
    batch_size = int(config["training_side_calibration"]["batch_size"])
    if args.smoke:
        inference_files = [oof_train_files[0], calibration_files[0]]
        batch_size = 1
    geometry = WindowGeometry(temporal_stride=int(
        config["training_side_calibration"]["window_stride"]
    ))
    dataset = RealWindowDataset(
        PROJECT / "RealPDE-Competition-Data" / "train_real",
        inference_files,
        geometry,
        preload=bool(config["training_side_calibration"]["preload_trajectories"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    interval = config["interval"]
    candidates = list(itertools.product(interval["alpha_grid"], interval["beta_grid"]))
    alpha = torch.tensor([row[0] for row in candidates], dtype=torch.float32, device=device)
    beta = torch.tensor([row[1] for row in candidates], dtype=torch.float32, device=device)
    sigma = float(interval["sigma_global"])
    totals = {
        subset: {
            "branch": np.zeros((len(candidates), 3), dtype=np.float64),
            "coverage": np.zeros(len(candidates), dtype=np.int64),
            "scored": 0,
            "windows": 0,
        }
        for subset in ("calibration", "diagnostic_in_sample")
    }
    normalizer = checkpoint["normalizer"]
    mean_input = torch.tensor(normalizer["mean_input"], dtype=torch.float32, device=device)
    std_input = torch.tensor(normalizer["std_input"], dtype=torch.float32, device=device)
    mean_target = torch.tensor(normalizer["mean_target"], dtype=torch.float32, device=device)
    std_target = torch.tensor(normalizer["std_target"], dtype=torch.float32, device=device)
    scoring = importlib.import_module("scoring")
    cursor = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader, start=1):
            names = [name for name, _ in dataset.entries[cursor:cursor + len(inputs)]]
            cursor += len(inputs)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model((inputs - mean_input) / std_input) * std_target + mean_target
            prediction[..., 2] = 0.0
            for subset in ("calibration", "diagnostic_in_sample"):
                use_calibration = subset == "calibration"
                mask_values = [
                    (parse_trajectory_name(name)[0] in calibration_res) == use_calibration
                    for name in names
                ]
                if not any(mask_values):
                    continue
                mask = torch.tensor(mask_values, dtype=torch.bool, device=device)
                pred_part = prediction[mask]
                target_part = targets[mask]
                factors = accuracy_factors(
                    pred_part.float().cpu().numpy(), target_part.float().cpu().numpy(), scoring
                )
                branch, coverage, scored = candidate_interval_sums(
                    pred_part,
                    target_part,
                    torch.from_numpy(factors).to(device),
                    alpha,
                    beta,
                    sigma,
                )
                totals[subset]["branch"] += branch.cpu().numpy()
                totals[subset]["coverage"] += coverage.cpu().numpy()
                totals[subset]["scored"] += scored
                totals[subset]["windows"] += int(mask.sum())
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                print(
                    f"batch={batch_index}/{len(loader)} windows={cursor}/{len(dataset)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    if cursor != len(dataset):
        raise AssertionError(f"processed {cursor} windows, expected {len(dataset)}")

    results = {}
    for subset in ("calibration", "diagnostic_in_sample"):
        sps, coverage = finalize_sps(
            totals[subset]["branch"], totals[subset]["coverage"], totals[subset]["scored"]
        )
        results[subset] = [
            {
                "index": index,
                "alpha": float(a),
                "beta": float(b),
                "sps_score": float(sps[index]),
                "coverage": float(coverage[index]),
            }
            for index, (a, b) in enumerate(candidates)
        ]
    selected_index = int(np.argmax([row["sps_score"] for row in results["calibration"]]))
    default_index = candidates.index((interval["default_alpha"], interval["default_beta"]))
    e006_index = candidates.index((interval["e006_alpha"], interval["e006_beta"]))
    selected = {
        "index": selected_index,
        "alpha": float(candidates[selected_index][0]),
        "beta": float(candidates[selected_index][1]),
    }
    e006_oof_sps = results["calibration"][e006_index]["sps_score"]
    e006_in_sample_sps = float(interval["e006_in_sample_calibration_sps"])
    selected_differs = selected["alpha"] != float(interval["e006_alpha"]) or selected[
        "beta"
    ] != float(interval["e006_beta"])
    e006_oof_drop = e006_in_sample_sps - e006_oof_sps
    gains = {
        "calibration": (
            results["calibration"][selected_index]["sps_score"]
            - results["calibration"][default_index]["sps_score"]
        ),
        "diagnostic_in_sample": (
            results["diagnostic_in_sample"][selected_index]["sps_score"]
            - results["diagnostic_in_sample"][default_index]["sps_score"]
        ),
    }
    acceptance = {
        "oof_calibration_gain_pass": gains["calibration"] >= float(
            config["training_side_calibration"]["minimum_oof_calibration_sps_gain_vs_default"]
        ),
        "selected_differs_from_e006": selected_differs,
        "e006_interval_oof_drop_pass": e006_oof_drop >= float(
            config["training_side_calibration"]["minimum_e006_interval_oof_sps_drop_vs_e006_in_sample"]
        ),
    }
    acceptance["strict_hypothesis_accepted"] = bool(
        acceptance["selected_differs_from_e006"] or acceptance["e006_interval_oof_drop_pass"]
    )
    report = {
        "experiment": "E009",
        "status": "smoke_complete" if args.smoke else "calibration_complete_not_yet_validated",
        "git_commit": git_head(),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "split_config_sha256": split["config_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": metadata,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "candidates": len(candidates),
        "oof_training_files": len(oof_train_files),
        "calibration_files": len(calibration_files),
        "validation_field_values_accessed": [],
        "windows": {subset: totals[subset]["windows"] for subset in totals},
        "selected": selected,
        "default_index": default_index,
        "e006_index": e006_index,
        "e006_oof_calibration_sps": e006_oof_sps,
        "e006_in_sample_calibration_sps": e006_in_sample_sps,
        "e006_interval_oof_sps_drop": e006_oof_drop,
        "sps_gain_vs_default": gains,
        "training_side_acceptance": acceptance,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "limitations": config["limitations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: report[key] for key in (
        "status", "selected", "e006_oof_calibration_sps", "e006_interval_oof_sps_drop",
        "sps_gain_vs_default", "training_side_acceptance",
        "wall_seconds", "peak_gpu_allocated_bytes"
    )}, indent=2))


if __name__ == "__main__":
    main()
