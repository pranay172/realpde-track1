#!/usr/bin/env python3
"""Evaluate a fixed-final backbone/adapter once with verified fold provenance."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "realpde_t1_starting_kit_v9")]
import scoring
from load_baseline import load_baseline
from realpde_t1.adapter import load_residual_cno
from realpde_t1.validation import persist_first_frames
from e003_evaluate_cno import timed_prediction
from e003_train_cno import sha256_file
from e034_evaluate_backbone import score_pair, by_nominal_re
from sps_headroom import analyze


def assess(candidate, reference, slices, ref_slices, evaluation):
    checks = {}
    for metric, floor in evaluation["minimum_deltas"].items():
        delta = candidate[metric] - reference[metric]
        checks[metric] = {"delta": delta, "minimum_delta":floor, "passed":bool(delta >= floor)}
    for re, scores in slices.items():
        for metric, allowance in evaluation["slice_max_regression"].items():
            delta = scores["calibrated"]["scores"][metric] - ref_slices[re]["calibrated"]["scores"][metric]
            checks[f"Re{re}_{metric}"] = {"delta":delta,"minimum_delta":-allowance,"passed":bool(delta >= -allowance)}
    return {"passed":all(v["passed"] for v in checks.values()),"checks":checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--artifacts",type=Path,required=True)
    ap.add_argument("--device",default="cuda")
    args = ap.parse_args()
    out = args.artifacts / "evaluation.json"
    if out.exists() or (args.artifacts / "predictions.npy").exists():
        raise FileExistsError("refusing repeated evaluation; preserve existing predictions/report")
    config = json.loads(args.config.read_text())
    split_path = REPO / config["split_config"]
    split = json.loads(split_path.read_text())
    checkpoint_path = args.artifacts / "final.pth"
    ckpt = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if ckpt.get("experiment") != config["experiment"] or ckpt.get("status") not in ("fixed_budget_final", "early_stopped_final"):
        raise ValueError("checkpoint experiment/status mismatch")
    prov = ckpt.get("provenance",{})
    if ckpt["status"] == "early_stopped_final":
        if prov.get("early_stopping_policy",{}).get("validation_used") is not False:
            raise ValueError("early stopping must be training-only")
        if not 0 < ckpt["training"]["updates"] < config["training"]["num_updates"]:
            raise ValueError("invalid early stopping update")
    if prov.get("config_sha256") != sha256_file(args.config):
        raise ValueError("training configuration changed")
    if set(prov.get("training_files",[])) != set(split["training_files"]) or prov.get("validation_field_values_accessed") != []:
        raise ValueError("missing or incompatible training provenance")
    ev = config["evaluation"]
    arrays = REPO / ev["arrays"]
    inputs = np.load(arrays / "inputs.npy",mmap_mode="r")
    targets = np.load(arrays / "targets.npy",mmap_mode="r")
    windows = json.loads((arrays / "windows.json").read_text())
    observed = {w["file"] for w in windows}
    if observed != set(split["validation_files"]) or observed & set(split["training_files"]):
        raise ValueError("holdout manifest mismatch")
    if inputs.shape != (ev["windows"],20,32,64,3) or inputs.shape != targets.shape:
        raise ValueError("unexpected validation shape")
    device = torch.device(args.device)
    if ckpt.get("model_kind") == "residual_cno":
        model = load_residual_cno(ckpt,device=device)
    else:
        model,_ = load_baseline("cno",str(checkpoint_path),device=str(device))
    model.eval()
    dest = args.artifacts / "predictions.npy"
    seconds,_ = timed_prediction(model,device,inputs,dest,4,ckpt["normalizer"])
    p = np.load(dest,mmap_mode="r")
    if not np.isfinite(p).all():
        raise FloatingPointError("nonfinite predictions")
    interval = ev["interval"]
    hybrid = persist_first_frames(p,inputs,4)
    raw = score_pair(p,targets,seconds,interval,scoring)
    cal = score_pair(hybrid,targets,seconds,interval,scoring)
    slices = by_nominal_re(hybrid,targets,windows,seconds,interval,scoring)
    ref = persist_first_frames(np.load(REPO / ev["reference_predictions"],mmap_mode="r"),inputs,4)
    ref_scores = score_pair(ref,targets,seconds,interval,scoring)["calibrated"]["scores"]
    ref_slices = by_nominal_re(ref,targets,windows,seconds,interval,scoring)
    report = {"experiment":config["experiment"],"evidence_status":"leakage_safe",
              "checkpoint_status":ckpt["status"],"training":ckpt.get("training",{}),
              "config_sha256":sha256_file(args.config),"checkpoint_sha256":sha256_file(checkpoint_path),
              "prediction_sha256":sha256_file(dest),"raw":raw,"persist_first_4":cal,
              "by_nominal_re":slices,"reference_scores":ref_scores,
              "assessment":assess(cal["calibrated"]["scores"],ref_scores,slices,ref_slices,ev)}
    out.write_text(json.dumps(report,indent=2)+"\n")
    del hybrid, ref
    diagnostics = {str(k):analyze(dest,arrays,split_path,k) for k in (0,4)}
    (args.artifacts / "sps_headroom.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    print(json.dumps({"experiment":config["experiment"],"scores":cal["calibrated"]["scores"],"assessment":report["assessment"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
