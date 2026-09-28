#!/usr/bin/env python3
"""Run E004 using only saved E002/E003 arrays and checkpoint states."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
DEFAULT_CONFIG = REPO / "configs" / "e004_postmortem.json"
DEFAULT_OUTPUT = REPO / "artifacts" / "e004_postmortem" / "report.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.diagnostics import (  # noqa: E402
    checkpoint_drift,
    compare_predictions,
    field_diagnostics,
    lead_relative_l2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    return (REPO / value).resolve()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def subset(array: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.asarray(array[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


def persistence(inputs: np.ndarray) -> np.ndarray:
    result = np.repeat(inputs[:, -1:, ...], 20, axis=1).astype(np.float32, copy=False)
    result[..., 2] = 0.0
    return result


def comparison_for_indices(
    prediction: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
    scoring: Any,
) -> dict[str, Any]:
    learned = subset(prediction, indices)
    input_slice = subset(inputs, indices)
    truth = subset(targets, indices)
    return compare_predictions(learned, persistence(input_slice), truth, scoring)


def grouped_comparisons(
    prediction: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    windows: list[dict[str, Any]],
    key: str,
    scoring: Any,
) -> dict[str, Any]:
    values = sorted({row[key] for row in windows})
    return {
        str(value): comparison_for_indices(
            prediction,
            inputs,
            targets,
            [row["index"] for row in windows if row[key] == value],
            scoring,
        )
        for value in values
    }


def simple_relative_comparison(
    prediction: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    indices: list[int],
) -> dict[str, Any]:
    learned = subset(prediction, indices)
    input_slice = subset(inputs, indices)
    truth = subset(targets, indices)
    baseline = persistence(input_slice)
    learned_leads = lead_relative_l2(learned, truth)
    baseline_leads = lead_relative_l2(baseline, truth)
    return {
        "n_samples": len(indices),
        "e003_by_lead": learned_leads,
        "persistence_by_lead": baseline_leads,
        "e003_minus_persistence_by_lead": [
            learned_value - baseline_value
            for learned_value, baseline_value in zip(learned_leads, baseline_leads)
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace completed E004 report: {args.output}")
    raw_config = args.config.read_bytes()
    config = json.loads(raw_config)
    paths = {name: resolve(value) for name, value in config["sources"].items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing E004 source {name}: {path}")

    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    persistence_report = json.loads(paths["persistence_report"].read_text(encoding="utf-8"))
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if source_hashes["prediction"] != evaluation["prediction_sha256"]:
        raise ValueError("saved E003 prediction hash does not match evaluation.json")
    if source_hashes["final_checkpoint"] != evaluation["checkpoint_sha256"]:
        raise ValueError("saved E003 final-checkpoint hash does not match evaluation.json")

    prediction = np.load(paths["prediction"], mmap_mode="r")
    inputs = np.load(paths["validation_inputs"], mmap_mode="r")
    targets = np.load(paths["validation_targets"], mmap_mode="r")
    windows = json.loads(paths["validation_windows"].read_text(encoding="utf-8"))
    expected_shape = (357, 20, 32, 64, 3)
    if prediction.shape != expected_shape or inputs.shape != expected_shape or targets.shape != expected_shape:
        raise ValueError(f"unexpected saved-array shapes: {prediction.shape}/{inputs.shape}/{targets.shape}")
    if len(windows) != expected_shape[0] or [row["index"] for row in windows] != list(range(357)):
        raise ValueError("saved E002 window metadata is incomplete or out of order")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("saved E003 prediction contains non-finite values")

    scoring = importlib.import_module("scoring")
    all_indices = list(range(len(windows)))
    overall = comparison_for_indices(prediction, inputs, targets, all_indices, scoring)
    for branch, expected in (("e003", evaluation["metrics"]["overall"]), (
        "persistence", persistence_report["models"]["persistence"]["metrics"]["overall"]
    )):
        for name in ("relative_l2", "tke_relative_l2", "mvpe_relative_l2", "sps_aggregate"):
            if not np.isclose(overall[branch]["raw"][name], expected["raw"][name], rtol=0, atol=1e-7):
                raise ValueError(f"recomputed {branch} {name} does not match its source report")

    by_stratum = grouped_comparisons(prediction, inputs, targets, windows, "group", scoring)
    by_re = grouped_comparisons(prediction, inputs, targets, windows, "nominal_re", scoring)
    by_aoa = grouped_comparisons(prediction, inputs, targets, windows, "nominal_aoa", scoring)
    horizon = {
        "overall": simple_relative_comparison(prediction, inputs, targets, all_indices),
        **{
            group: simple_relative_comparison(
                prediction,
                inputs,
                targets,
                [row["index"] for row in windows if row["group"] == group],
            )
            for group in sorted({row["group"] for row in windows})
        },
    }
    by_start = {
        str(start): simple_relative_comparison(
            prediction,
            inputs,
            targets,
            [row["index"] for row in windows if row["start"] == start],
        )
        for start in sorted({row["start"] for row in windows})
    }
    normalizer = evaluation["normalizer"]
    distribution_by_re = {}
    for re_value in sorted({row["nominal_re"] for row in windows}):
        indices = [row["index"] for row in windows if row["nominal_re"] == re_value]
        distribution_by_re[str(re_value)] = field_diagnostics(
            subset(prediction, indices), subset(targets, indices), normalizer
        )

    initial_checkpoint = torch.load(paths["initial_checkpoint"], map_location="cpu", weights_only=False)
    final_checkpoint = torch.load(paths["final_checkpoint"], map_location="cpu", weights_only=False)
    drift = checkpoint_drift(
        initial_checkpoint["model_state_dict"], final_checkpoint["model_state_dict"]
    )

    rule = config["decision_rule"]
    edge_res = ("3750", "26700")
    edge_checks = {
        re_value: by_re[re_value]["e003_minus_persistence"]["raw"]["relative_l2"] > 0
        for re_value in edge_res
    }
    bn_check = (
        drift["batchnorm_running_state"]["running_mean_shift_standardized_rms"]
        >= float(rule["batchnorm_running_mean_standardized_rms_min"])
    )
    weight_check = (
        drift["categories"]["all_trainable"]["relative_l2"]
        <= float(rule["global_trainable_parameter_relative_l2_max"])
    )
    conditions = {
        "edge_re_checks": edge_checks,
        "both_edge_re_raw_rel_l2_worse_than_persistence": all(edge_checks.values()),
        "batchnorm_running_mean_shift_threshold_met": bn_check,
        "global_trainable_parameter_drift_threshold_met": weight_check,
    }
    all_hold = all(edge_checks.values()) and bn_check and weight_check
    decision = {
        "all_preregistered_conditions_hold": all_hold,
        "next_action": rule["if_all_hold"] if all_hold else rule["otherwise"],
        "causal_caveat": config["limitations"][0],
    }

    report = {
        "experiment": "E004",
        "status": "complete",
        "scope": config["scope"],
        "git_commit": git_head(),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "source_hashes": source_hashes,
        "overall": overall,
        "by_stratum": by_stratum,
        "by_nominal_re": by_re,
        "by_nominal_aoa": by_aoa,
        "relative_l2_by_forecast_lead": horizon,
        "relative_l2_by_window_start": by_start,
        "field_diagnostics_by_nominal_re": distribution_by_re,
        "checkpoint_drift": drift,
        "decision_conditions": conditions,
        "decision": decision,
        "limitations": config["limitations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "experiment": report["experiment"],
        "status": report["status"],
        "config_sha256": report["config_sha256"],
        "decision_conditions": conditions,
        "decision": decision,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
