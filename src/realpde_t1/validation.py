"""Leakage-resistant Track 1 split, window, and scoring utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from types import ModuleType
from typing import Any, Sequence

import h5py
import numpy as np


_TRAJECTORY_RE = re.compile(r"^(?P<re>\d+)_(?P<aoa>\d+)\.h5$")


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON split config and retain a content hash for provenance."""
    raw = path.read_bytes()
    config = json.loads(raw)
    config["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def parse_trajectory_name(name: str) -> tuple[int, int]:
    """Return nominal Reynolds number and angle of attack from a filename."""
    match = _TRAJECTORY_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected trajectory filename: {name}")
    return int(match.group("re")), int(match.group("aoa"))


def files_for_reynolds(files: Sequence[str], reynolds: Sequence[int]) -> list[str]:
    """Return the named files whose nominal Reynolds number is in ``reynolds``."""
    allowed = {int(value) for value in reynolds}
    selected = [name for name in files if parse_trajectory_name(name)[0] in allowed]
    actual = {parse_trajectory_name(name)[0] for name in selected}
    if actual != allowed:
        missing = sorted(allowed - actual)
        extra = sorted(actual - allowed)
        raise ValueError(
            f"Reynolds file selection mismatch; missing={missing}, extra={extra}"
        )
    if not selected:
        raise ValueError("Reynolds file selection is empty")
    return selected


def window_starts(n_frames: int, window: dict[str, int]) -> list[int]:
    """Return starts whose complete input and target fit in the trajectory."""
    t_in = int(window["input_frames"])
    t_out = int(window["output_frames"])
    stride = int(window["temporal_stride"])
    offset = int(window["start_offset"])
    if min(t_in, t_out, stride) <= 0 or offset < 0:
        raise ValueError(f"invalid window settings: {window}")
    if stride < t_in + t_out:
        raise ValueError("temporal_stride must prevent overlap between evaluation samples")
    stop = n_frames - t_in - t_out + 1
    return list(range(offset, max(offset, stop), stride))


def _field(handle: h5py.File, name: str) -> h5py.Dataset:
    if name in handle:
        return handle[name]
    if "measured_data" in handle and name in handle["measured_data"]:
        return handle["measured_data"][name]
    raise KeyError(f"missing field {name!r} in {handle.filename}")


def _inspect_trajectory(path: Path, window: dict[str, int]) -> dict[str, Any]:
    nominal_re, nominal_aoa = parse_trajectory_name(path.name)
    with h5py.File(path, "r") as handle:
        u = _field(handle, "u")
        v = _field(handle, "v")
        if u.shape != v.shape or u.ndim != 3:
            raise ValueError(f"invalid u/v shapes in {path}: {u.shape}/{v.shape}")
        expected_hw = (int(window["raw_height"]), int(window["raw_width"]))
        if tuple(u.shape[1:]) != expected_hw:
            raise ValueError(f"unexpected spatial shape in {path}: {u.shape[1:]}")
        if "aoa" not in handle or "re" not in handle or "t" not in handle:
            raise KeyError(f"missing scalar/time metadata in {path}")
        stored_aoa = int(handle["aoa"][()])
        stored_re = int(handle["re"][()])
        if stored_aoa != nominal_aoa:
            raise ValueError(f"AoA mismatch in {path}: {stored_aoa} != {nominal_aoa}")
        if len(handle["t"]) != u.shape[0]:
            raise ValueError(f"time/field length mismatch in {path}")
        n_frames = int(u.shape[0])
    starts = window_starts(n_frames, window)
    if not starts:
        raise ValueError(f"trajectory is too short for one evaluation window: {path}")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "nominal_re": nominal_re,
        "nominal_aoa": nominal_aoa,
        "stored_re": stored_re,
        "stored_aoa": stored_aoa,
        "frames": n_frames,
        "window_starts": starts,
        "n_windows": len(starts),
    }


def audit_split(data_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen split against disk and return its full manifest."""
    actual = {path.name for path in data_root.glob("*.h5")}
    train = list(config["training_files"])
    validation = list(config["validation_files"])
    excluded = list(config["excluded_files"])
    declared = train + validation + excluded
    if len(declared) != len(set(declared)):
        raise ValueError("the split config contains duplicate filenames")
    if set(declared) != actual:
        missing = sorted(set(declared) - actual)
        extra = sorted(actual - set(declared))
        raise ValueError(f"split/data mismatch; missing={missing}, extra={extra}")
    if len(actual) != int(config["expected_h5_count"]):
        raise ValueError(f"expected {config['expected_h5_count']} HDF5 files, found {len(actual)}")
    if excluded != ["7575_0.h5"]:
        raise ValueError("E002 must exclude exactly the known-bad 7575_0.h5 trajectory")

    train_res = {parse_trajectory_name(name)[0] for name in train}
    validation_res = {parse_trajectory_name(name)[0] for name in validation}
    if train_res & validation_res:
        raise ValueError(f"nominal Reynolds groups cross the split: {sorted(train_res & validation_res)}")
    configured_groups = {
        group: {int(value) for value in values}
        for group, values in config["validation_groups"].items()
    }
    grouped_res = set().union(*configured_groups.values())
    if grouped_res != validation_res:
        raise ValueError(f"validation group mismatch: {grouped_res} != {validation_res}")

    rows = []
    for split, names in (("train", train), ("validation", validation), ("excluded", excluded)):
        for name in names:
            row = _inspect_trajectory(data_root / name, config["window"])
            row["split"] = split
            row["group"] = None
            if split == "validation":
                labels = [label for label, values in configured_groups.items() if row["nominal_re"] in values]
                if len(labels) != 1:
                    raise ValueError(f"ambiguous validation group for {name}: {labels}")
                row["group"] = labels[0]
            rows.append(row)

    validation_rows = [row for row in rows if row["split"] == "validation"]
    inventory_basis = [
        {key: row[key] for key in ("file", "bytes", "nominal_re", "nominal_aoa", "stored_re", "stored_aoa", "frames")}
        for row in rows
    ]
    inventory_raw = json.dumps(inventory_basis, sort_keys=True, separators=(",", ":")).encode()
    return {
        "experiment": config["experiment"],
        "config_sha256": config["config_sha256"],
        "data_inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
        "data_root": str(data_root),
        "counts": {
            "all": len(rows),
            "train": len(train),
            "validation": len(validation),
            "excluded": len(excluded),
            "validation_windows": sum(row["n_windows"] for row in validation_rows),
        },
        "validation_nominal_re": sorted(validation_res),
        "training_nominal_re": sorted(train_res),
        "window": config["window"],
        "trajectories": rows,
    }


def build_validation_arrays(
    data_root: Path,
    manifest: dict[str, Any],
    destination: Path,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Materialize deterministic float32 input/target memmaps and window metadata."""
    destination.mkdir(parents=True, exist_ok=True)
    window = manifest["window"]
    n = int(manifest["counts"]["validation_windows"])
    t_in = int(window["input_frames"])
    t_out = int(window["output_frames"])
    spatial_stride = int(window["spatial_stride"])
    h = int(window["evaluation_height"])
    w = int(window["evaluation_width"])
    shape_in = (n, t_in, h, w, 3)
    shape_out = (n, t_out, h, w, 3)
    input_path = destination / "inputs.npy"
    target_path = destination / "targets.npy"
    inputs = np.lib.format.open_memmap(input_path, mode="w+", dtype=np.float32, shape=shape_in)
    targets = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32, shape=shape_out)
    inputs[...] = 0.0
    targets[...] = 0.0

    window_rows: list[dict[str, Any]] = []
    index = 0
    validation_rows = [row for row in manifest["trajectories"] if row["split"] == "validation"]
    for row in validation_rows:
        with h5py.File(data_root / row["file"], "r") as handle:
            u = np.asarray(_field(handle, "u")[:, ::spatial_stride, ::spatial_stride], dtype=np.float32)
            v = np.asarray(_field(handle, "v")[:, ::spatial_stride, ::spatial_stride], dtype=np.float32)
        if u.shape[1:] != (h, w) or v.shape != u.shape:
            raise ValueError(f"downsampled shape mismatch for {row['file']}: {u.shape}/{v.shape}")
        for start in row["window_starts"]:
            middle = start + t_in
            end = middle + t_out
            inputs[index, ..., 0] = u[start:middle]
            inputs[index, ..., 1] = v[start:middle]
            targets[index, ..., 0] = u[middle:end]
            targets[index, ..., 1] = v[middle:end]
            window_rows.append({
                "index": index,
                "file": row["file"],
                "nominal_re": row["nominal_re"],
                "nominal_aoa": row["nominal_aoa"],
                "group": row["group"],
                "start": start,
                "input_stop": middle,
                "target_stop": end,
            })
            index += 1
    inputs.flush()
    targets.flush()
    if index != n:
        raise AssertionError(f"materialized {index} windows, expected {n}")
    (destination / "windows.json").write_text(json.dumps(window_rows, indent=2) + "\n", encoding="utf-8")
    return input_path, target_path, window_rows


def persist_first_frames(
    prediction: np.ndarray,
    inputs: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Replace the first ``n_frames`` predicted frames with last-frame persistence."""
    if prediction.shape != inputs.shape:
        raise ValueError(f"prediction/input shape mismatch: {prediction.shape}/{inputs.shape}")
    if n_frames < 0 or n_frames > prediction.shape[1]:
        raise ValueError(f"invalid persist-first frame count: {n_frames}")
    output = np.array(prediction, dtype=np.float32, copy=True)
    if n_frames:
        output[:, :n_frames] = inputs[:, -1:, ...].astype(np.float32, copy=False)
    output[..., 2] = 0.0
    return output


def training_touches_fold(training_files: Sequence[str], validation_files: Sequence[str]) -> bool:
    """Return True when a method's training Re groups intersect a fold's holdout."""
    train_re = {parse_trajectory_name(name)[0] for name in training_files}
    fold_re = {parse_trajectory_name(name)[0] for name in validation_files}
    return bool(train_re & fold_re)


def score_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    mean_seconds_per_sample: float,
    scoring: ModuleType,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the five official subscores plus their raw error diagnostics."""
    scoring.validate_shapes(prediction, target)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction contains non-finite values")
    channels = scoring.measured_channels(target)
    rel_each = scoring.rel_l2_per_sample(prediction, target, channels)
    tke_each = scoring.tke_rel_l2_per_sample(prediction, target, channels)
    mvpe_each = scoring.mvpe_rel_l2_per_sample(prediction, target)
    sps, coverage = scoring.aggregate_sps(
        prediction, target, channels, lower=lower, upper=upper
    )
    rel = float(np.mean(rel_each))
    tke = float(np.mean(tke_each))
    mvpe = float(np.mean(mvpe_each))
    return {
        "n_samples": int(prediction.shape[0]),
        "raw": {
            "relative_l2": rel,
            "tke_relative_l2": tke,
            "mvpe_relative_l2": mvpe,
            "sps_aggregate": float(sps),
            "interval_coverage": float(coverage),
        },
        "scores": {
            "rel_l2_score": scoring.score_error(rel),
            "tke_score": scoring.score_error(tke),
            "mvpe_score": scoring.score_error(mvpe),
            "time_score": scoring.score_time(mean_seconds_per_sample),
            "sps_score": scoring.score_sps(sps),
        },
    }


def score_slices(
    prediction: np.ndarray,
    target: np.ndarray,
    windows: list[dict[str, Any]],
    mean_seconds_per_sample: float,
    scoring: ModuleType,
) -> dict[str, Any]:
    """Score the overall set and the frozen ID/OOD validation strata."""
    result = {
        "overall": score_prediction(prediction, target, mean_seconds_per_sample, scoring)
    }
    groups = sorted({row["group"] for row in windows})
    for group in groups:
        indices = np.asarray([row["index"] for row in windows if row["group"] == group])
        result[group] = score_prediction(
            prediction[indices], target[indices], mean_seconds_per_sample, scoring
        )
    return result
