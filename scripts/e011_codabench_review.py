#!/usr/bin/env python3
"""Contaminated/diagnostic review of E011 versus E008 after Codabench."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
KIT = REPO / "realpde_t1_starting_kit_v9"
E002_ARRAYS = REPO / "artifacts" / "e002_validation" / "arrays"
E005_ROOT = REPO / "artifacts" / "e005_scale_balanced_uv"
E006_EVAL = REPO / "artifacts" / "e006_uncertainty" / "evaluation.json"
E011_CKPT = REPO / "artifacts" / "e011_full_public" / "final.pth"
OUTPUT = REPO / "artifacts" / "e011_codabench_review" / "report.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from e003_evaluate_cno import timed_prediction  # noqa: E402
from realpde_t1.uncertainty import interval_bounds_numpy  # noqa: E402
from realpde_t1.validation import parse_trajectory_name, score_prediction  # noqa: E402
from load_baseline import load_baseline  # noqa: E402


def score_map(err: float) -> float:
    return 100.0 / (1.0 + 0.5 * err)


def inv_score(score: float) -> float:
    return 2.0 * (100.0 / score - 1.0)


def persistence(inputs: np.ndarray) -> np.ndarray:
    result = np.repeat(inputs[:, -1:, ...], 20, axis=1).astype(np.float32, copy=False)
    result[..., 2] = 0.0
    return result


def field_std(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        key = "u" if "u" in handle else "measured_data/u"
        group = handle if "u" in handle else handle["measured_data"]
        u = np.asarray(group["u"][:, ::2, ::2], dtype=np.float32)
        v = np.asarray(group["v"][:, ::2, ::2], dtype=np.float32)
    scored_u = u[u != 0]
    scored_v = v[v != 0]
    return {
        "u_std": float(scored_u.std()) if scored_u.size else 0.0,
        "v_std": float(scored_v.std()) if scored_v.size else 0.0,
        "u_mean": float(scored_u.mean()) if scored_u.size else 0.0,
        "zero_frac_u": float(np.mean(u == 0)),
    }


def main() -> None:
    scoring = importlib.import_module("scoring")
    inputs = np.load(E002_ARRAYS / "inputs.npy", mmap_mode="r")
    targets = np.load(E002_ARRAYS / "targets.npy", mmap_mode="r")
    windows = json.loads((E002_ARRAYS / "windows.json").read_text(encoding="utf-8"))
    e005 = np.load(E005_ROOT / "predictions.npy", mmap_mode="r")
    e006 = json.loads(E006_EVAL.read_text(encoding="utf-8"))
    persist = persistence(np.asarray(inputs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(E011_CKPT, map_location="cpu", weights_only=False)
    model, _ = load_baseline("cno", str(E011_CKPT), device=str(device))
    model = model.to(device).eval()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT.parent / "e011_on_e002.npy"
    mean_s, _ = timed_prediction(
        model, device, inputs, pred_path, 1, checkpoint["normalizer"]
    )
    e011 = np.load(pred_path, mmap_mode="r")
    lower, upper = interval_bounds_numpy(np.asarray(e011), 0.025, 0.15, scoring.SIGMA_GLOBAL)
    e011_metrics = score_prediction(np.asarray(e011), np.asarray(targets), mean_s, scoring)
    channels = scoring.measured_channels(np.asarray(targets))
    sps, coverage = scoring.aggregate_sps(
        np.asarray(e011), np.asarray(targets), channels, lower=lower, upper=upper
    )
    e011_metrics["scores"]["sps_score"] = scoring.score_sps(sps)
    e011_metrics["raw"]["interval_coverage"] = float(coverage)

    persist_metrics = score_prediction(persist, np.asarray(targets), 1e-6, scoring)
    e005_metrics = score_prediction(np.asarray(e005), np.asarray(targets), 0.0668, scoring)

    blends = []
    for mix in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        blended = (1.0 - mix) * persist + mix * np.asarray(e005)
        blended[..., 2] = 0.0
        row = score_prediction(blended, np.asarray(targets), 0.0668, scoring)
        blends.append({
            "e005_weight": mix,
            "rel_l2_score": row["scores"]["rel_l2_score"],
            "tke_score": row["scores"]["tke_score"],
            "mvpe_score": row["scores"]["mvpe_score"],
        })

    lead = []
    for t in range(20):
        p = np.asarray(e005)[:, t : t + 1]
        q = persist[:, t : t + 1]
        y = np.asarray(targets)[:, t : t + 1]
        lead.append({
            "frame": t + 1,
            "e005_rel_l2": float(np.mean(scoring.rel_l2_per_sample(p, y, 2))),
            "persistence_rel_l2": float(np.mean(scoring.rel_l2_per_sample(q, y, 2))),
        })

    by_re = {}
    for re_value in sorted({int(row["nominal_re"]) for row in windows}):
        idx = np.asarray([row["index"] for row in windows if int(row["nominal_re"]) == re_value])
        e005_re = score_prediction(np.asarray(e005)[idx], np.asarray(targets)[idx], 0.0668, scoring)
        e011_re = score_prediction(np.asarray(e011)[idx], np.asarray(targets)[idx], mean_s, scoring)
        l, u = interval_bounds_numpy(np.asarray(e011)[idx], 0.025, 0.15, scoring.SIGMA_GLOBAL)
        sps_re, cov_re = scoring.aggregate_sps(
            np.asarray(e011)[idx], np.asarray(targets)[idx], 2, lower=l, upper=u
        )
        e011_re["scores"]["sps_score"] = scoring.score_sps(sps_re)
        e011_re["raw"]["interval_coverage"] = float(cov_re)
        by_re[str(re_value)] = {
            "e005": e005_re["scores"],
            "e011_contaminated": e011_re["scores"],
            "e011_coverage": float(cov_re),
        }

    matched = []
    real_root = PROJECT / "RealPDE-Competition-Data" / "train_real"
    sim_root = PROJECT / "RealPDE-Competition-Data" / "train_sim"
    sample_files = [
        "5025_0.h5", "8850_10.h5", "12675_5.h5", "16500_15.h5",
        "19050_0.h5", "24150_10.h5", "25425_20.h5",
    ]
    for name in sample_files:
        real_stats = field_std(real_root / name)
        sim_stats = field_std(sim_root / name)
        matched.append({
            "file": name,
            "nominal_re": parse_trajectory_name(name)[0],
            "real": real_stats,
            "sim": sim_stats,
            "u_std_real_over_sim": real_stats["u_std"] / max(sim_stats["u_std"], 1e-8),
        })

    cb_e008 = {
        "rel_l2_score": 92.875151, "tke_score": 71.383921, "mvpe_score": 92.532160,
        "time_score": 85.964314, "sps_score": 26.377220, "final_score": 74.843999,
    }
    cb_e011 = {
        "rel_l2_score": 92.544418, "tke_score": 70.433707, "mvpe_score": 92.108736,
        "time_score": 85.852481, "sps_score": 25.111476, "final_score": 74.189149,
    }
    report = {
        "status": "diagnostic_only",
        "warning": "E011 predictions on E002 are contaminated: E011 trained on those trajectories.",
        "codabench": {
            "e008": cb_e008,
            "e011": cb_e011,
            "e011_minus_e008": {k: cb_e011[k] - cb_e008[k] for k in cb_e008},
        },
        "raw_errors": {
            "e006_local_rel_l2": inv_score(94.23649110952087),
            "e008_codabench_rel_l2": inv_score(92.875151),
            "e011_codabench_rel_l2": inv_score(92.544418),
            "e006_local_tke": inv_score(70.38103897050776),
            "e008_codabench_tke": inv_score(71.383921),
            "e006_local_mvpe": inv_score(94.82818884232233),
            "e008_codabench_mvpe": inv_score(92.532160),
            "rel_l2_error_ratio_hidden_vs_local": inv_score(92.875151) / inv_score(94.23649110952087),
            "mvpe_error_ratio_hidden_vs_local": inv_score(92.532160) / inv_score(94.82818884232233),
        },
        "score_sensitivity": {
            "rel_l2_from_0.122_to_0.08": [score_map(0.1223), score_map(0.08)],
            "tke_from_0.84_to_0.50": [score_map(0.8417), score_map(0.50)],
            "sps_needed_if_other_e008_fixed_equal_weight_80": "unpublished weights; SPS is the largest local-to-hidden gap",
        },
        "e002_overall": {
            "persistence": persist_metrics["scores"],
            "e005_ood": e005_metrics["scores"],
            "e006_ood": e006["overall"]["scores"],
            "e011_contaminated": e011_metrics["scores"],
            "e011_contaminated_coverage": e011_metrics["raw"]["interval_coverage"],
            "e011_mean_seconds": mean_s,
        },
        "e002_by_re": by_re,
        "persistence_blend_on_e005": blends,
        "lead_time_rel_l2": lead,
        "matched_sim_real_scale": matched,
        "checkpoint_sha256": hashlib.sha256(E011_CKPT.read_bytes()).hexdigest(),
        "wall_note": "E011-on-E002 is an in-sample diagnostic, not a generalization score.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "e002_overall": report["e002_overall"],
        "codabench_delta": report["codabench"]["e011_minus_e008"],
        "raw_errors": report["raw_errors"],
        "best_blend": max(blends, key=lambda row: row["rel_l2_score"] + row["tke_score"] + row["mvpe_score"]),
        "u_std_ratios": [row["u_std_real_over_sim"] for row in matched],
    }, indent=2))


if __name__ == "__main__":
    main()
