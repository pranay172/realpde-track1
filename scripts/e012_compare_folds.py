#!/usr/bin/env python3
"""Build Fold C arrays and score the frozen E012 multi-fold comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
CONFIG = REPO / "configs" / "e012_multifold_comparison.json"
OUTPUT = REPO / "artifacts" / "e012_multifold" / "comparison.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402
from realpde_t1.validation import (  # noqa: E402
    audit_split,
    build_validation_arrays,
    load_config,
    persist_first_frames,
    score_prediction,
    training_touches_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def prepare_fold_c(config: dict[str, Any]) -> dict[str, Any]:
    fold = config["folds"]["C"]
    split = load_config(REPO / fold["split_config"])
    destination = REPO / fold["arrays"]
    marker = destination / "manifest.json"
    if (destination / "inputs.npy").exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("config_sha256") != split["config_sha256"]:
            raise ValueError("existing Fold C arrays do not match the frozen split")
        return saved
    data_root = PROJECT / "RealPDE-Competition-Data" / "train_real"
    manifest = audit_split(data_root, split)
    input_path, target_path, windows = build_validation_arrays(data_root, manifest, destination)
    report = {
        "split_config": fold["split_config"],
        "config_sha256": split["config_sha256"],
        "data_inventory_sha256": manifest["data_inventory_sha256"],
        "validation_files": list(split["validation_files"]),
        "validation_windows": int(manifest["counts"]["validation_windows"]),
        "input_path": str(input_path),
        "target_path": str(target_path),
        "windows": len(windows),
    }
    marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_fold_arrays(folder: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    inputs = np.load(folder / "inputs.npy", mmap_mode="r")
    targets = np.load(folder / "targets.npy", mmap_mode="r")
    windows = json.loads((folder / "windows.json").read_text(encoding="utf-8"))
    return inputs, targets, windows


def by_re(
    prediction: np.ndarray,
    targets: np.ndarray,
    windows: list[dict[str, Any]],
    seconds: float,
    scoring: Any,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        indices = np.asarray([
            int(row["index"]) for row in windows if int(row["nominal_re"]) == re_value
        ])
        lo = None if lower is None else lower[indices]
        up = None if upper is None else upper[indices]
        metrics[str(re_value)] = score_prediction(
            prediction[indices], targets[indices], seconds, scoring, lower=lo, upper=up
        )
    return metrics


def score_method(
    name: str,
    spec: dict[str, Any],
    fold_id: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    windows: list[dict[str, Any]],
    methods: dict[str, Any],
    interval: dict[str, Any],
    scoring: Any,
) -> dict[str, Any]:
    if name == "persistence":
        started = time.perf_counter()
        prediction = persistence_prediction(np.asarray(inputs))
        seconds = (time.perf_counter() - started) / max(len(inputs), 1)
        default = score_prediction(prediction, np.asarray(targets), seconds, scoring)
        return {
            "status": "scored",
            "seconds_per_sample": seconds,
            "default": default,
            "calibrated": None,
            "by_nominal_re": by_re(prediction, np.asarray(targets), windows, seconds, scoring),
        }

    base_name = spec.get("base_method", name)
    base = methods[base_name]
    pred_path = base.get("predictions", {}).get(fold_id)
    if pred_path is None:
        return {"status": "not_available_on_this_fold"}
    prediction = np.load(REPO / pred_path)
    if spec.get("kind") == "hybrid":
        prediction = persist_first_frames(
            prediction, np.asarray(inputs), int(spec["persist_first_frames"])
        )
    seconds = float(base.get("seconds_per_sample", {}).get(fold_id, 0.0668))
    default = score_prediction(prediction, np.asarray(targets), seconds, scoring)
    lower, upper = interval_bounds_numpy(
        np.asarray(prediction),
        float(interval["alpha"]),
        float(interval["beta"]),
        float(interval["sigma_global"]),
    )
    calibrated = score_prediction(
        np.asarray(prediction), np.asarray(targets), seconds, scoring, lower=lower, upper=upper
    )
    return {
        "status": "scored",
        "seconds_per_sample": seconds,
        "default": default,
        "calibrated": calibrated,
        "by_nominal_re": by_re(
            np.asarray(prediction), np.asarray(targets), windows, seconds, scoring, lower, upper
        ),
    }


def main() -> None:
    args = parse_args()
    if OUTPUT.exists() and not args.prepare_only:
        raise FileExistsError(f"E012 comparison already exists: {OUTPUT}")
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    fold_c = prepare_fold_c(config)
    if args.prepare_only:
        print(json.dumps({"status": "fold_c_arrays_ready", **fold_c}, indent=2))
        return

    scoring = importlib.import_module("scoring")
    splits = {
        fold_id: load_config(REPO / spec["split_config"])
        for fold_id, spec in config["folds"].items()
    }
    fold_re = {
        fold_id: {int(name.split("_")[0]) for name in split["validation_files"]}
        for fold_id, split in splits.items()
    }
    disjoint = {
        "A_B": fold_re["A"].isdisjoint(fold_re["B"]),
        "A_C": fold_re["A"].isdisjoint(fold_re["C"]),
        "B_C": fold_re["B"].isdisjoint(fold_re["C"]),
    }
    rows: dict[str, Any] = {}
    for fold_id, fold in config["folds"].items():
        inputs, targets, windows = load_fold_arrays(REPO / fold["arrays"])
        fold_rows = {}
        for method_name, spec in config["methods"].items():
            training_files: list[str] = []
            if spec.get("training_files_from"):
                source = json.loads((REPO / spec["training_files_from"]).read_text(encoding="utf-8"))
                training_files = list(source[spec["training_key"]])
            clean = not training_touches_fold(training_files, splits[fold_id]["validation_files"])
            if method_name != "persistence" and fold_id == "C":
                fold_rows[method_name] = {
                    "evidence_status": "withheld_fold_c_reserved",
                    "clean": clean,
                    "status": "not_scored",
                }
                continue
            scored = score_method(
                method_name, spec, fold_id, inputs, targets, windows,
                config["methods"], config["interval"], scoring,
            )
            scored["clean"] = clean
            scored["evidence_status"] = "leakage_safe" if clean else "contaminated"
            if not clean:
                scored["status"] = "not_used_for_ranking"
                scored.pop("default", None)
                scored.pop("calibrated", None)
                scored.pop("by_nominal_re", None)
            fold_rows[method_name] = scored
        rows[fold_id] = {
            "n_windows": int(len(inputs)),
            "validation_files": len(splits[fold_id]["validation_files"]),
            "nominal_re": sorted(fold_re[fold_id]),
            "methods": fold_rows,
        }

    persist_first_checks = {}
    for fold_id, base, hybrid in (
        ("A", "e005_cno", "e005_persist_first_4"),
        ("B", "e010_cno", "e010_persist_first_4"),
    ):
        base_scores = rows[fold_id]["methods"][base]["default"]["scores"]
        hybrid_scores = rows[fold_id]["methods"][hybrid]["default"]["scores"]
        base_sps = rows[fold_id]["methods"][base]["calibrated"]["scores"]["sps_score"]
        hybrid_sps = rows[fold_id]["methods"][hybrid]["calibrated"]["scores"]["sps_score"]
        persist_first_checks[fold_id] = {
            metric: hybrid_scores[metric] > base_scores[metric]
            for metric in ("rel_l2_score", "tke_score", "mvpe_score")
        }
        persist_first_checks[fold_id]["calibrated_sps_score"] = hybrid_sps > base_sps
        persist_first_checks[fold_id]["deltas"] = {
            metric: hybrid_scores[metric] - base_scores[metric]
            for metric in ("rel_l2_score", "tke_score", "mvpe_score")
        }
        persist_first_checks[fold_id]["deltas"]["calibrated_sps_score"] = hybrid_sps - base_sps

    clean_learned_on_c = [
        name for name, row in rows["C"]["methods"].items()
        if name != "persistence" and row.get("evidence_status") == "leakage_safe" and row.get("status") == "scored"
    ]
    checks = {
        "folds_are_pairwise_re_disjoint": all(disjoint.values()),
        "persist_first_4_beats_base_on_every_clean_fold": all(
            all(values[metric] for metric in config["success_criteria"]["persist_first_4_beats_base_on_every_clean_fold"])
            for values in persist_first_checks.values()
        ),
        "no_clean_learned_row_on_fold_c": clean_learned_on_c == [],
    }
    report = {
        "experiment": "E012",
        "status": "complete",
        "promotion": "protocol_only",
        "strict_hypothesis_accepted": all(checks.values()),
        "git_commit": git_head(),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "fold_c": fold_c,
        "fold_disjointness": disjoint,
        "folds": rows,
        "persist_first_checks": persist_first_checks,
        "checks": checks,
        "ranking_rule": config["ranking"],
        "limitations": config["limitations"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({
        "status": report["status"],
        "strict_hypothesis_accepted": report["strict_hypothesis_accepted"],
        "checks": checks,
        "persist_first_checks": persist_first_checks,
        "fold_c_windows": fold_c["validation_windows"],
    }, indent=2))


if __name__ == "__main__":
    main()
