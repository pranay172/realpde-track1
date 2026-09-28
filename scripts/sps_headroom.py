#!/usr/bin/env python3
"""Decompose frozen saved predictions, without fitting or selecting parameters."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "realpde_t1_starting_kit_v9")]
import scoring
from realpde_t1.sps_diagnostics import sps_components, summarize_sps
from realpde_t1.uncertainty import interval_bounds_numpy
from realpde_t1.validation import persist_first_frames
from e003_train_cno import sha256_file


def analyze(prediction_path, arrays, split_path, k):
    windows = json.loads((arrays / "windows.json").read_text())
    split = json.loads(split_path.read_text())
    train = set(split["training_files"])
    valid = set(split["validation_files"])
    observed = {w["file"] for w in windows}
    if train & observed or observed != valid:
        raise ValueError("window manifest does not match declared holdout")
    pred = np.load(prediction_path, mmap_mode="r")
    inputs = np.load(arrays / "inputs.npy", mmap_mode="r")
    targets = np.load(arrays / "targets.npy", mmap_mode="r")
    if pred.shape != targets.shape or len(windows) != len(pred):
        raise ValueError("saved prediction/target/manifest shape mismatch")
    chunks = []
    official_sum = 0.0
    for start in range(0, len(pred), 8):
        p = persist_first_frames(pred[start:start+8], inputs[start:start+8], k)
        t = targets[start:start+8]
        lo, hi = interval_bounds_numpy(p, 0.025, 0.15, scoring.SIGMA_GLOBAL)
        part = sps_components(p, t, lo, hi, scoring)
        official, _ = scoring.aggregate_sps(p, t, 2, lo, hi)
        official_sum += official * part["count"].sum()
        chunks.append(part)
    parts = {key: np.concatenate([p[key] for p in chunks]) for key in chunks[0]}
    summary = summarize_sps(parts)
    official = 100 * official_sum / parts["count"].sum()
    if abs(summary["sps_score"] - official) > 1e-5:
        raise AssertionError("decomposition differs from official scorer")
    return {
        "prediction": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
        "split_sha256": sha256_file(split_path), "persist_first_frames": k,
        "note": "Declared split checked; caller must also verify model training provenance. Oracle uses targets and is diagnostic only.",
        "overall": summary, "official_sps": official,
        "by_re": {str(re): summarize_sps(parts, [i for i,w in enumerate(windows) if w["nominal_re"] == re])
                  for re in sorted({w["nominal_re"] for w in windows})},
        "by_horizon": {f"{a+1}-{b}": summarize_sps(parts, horizon=slice(a,b))
                       for a,b in [(0,4),(4,10),(10,20)]},
        "worst_window_relative_l2": float(parts["errors"][:,0].max()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {str(k): analyze(args.predictions, args.arrays, args.split, k) for k in (0,4)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v["overall"] for k,v in report.items()}, indent=2))
