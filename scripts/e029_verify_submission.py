#!/usr/bin/env python3
"""Build and gate the E029 All-Data Residual Rollout Curriculum CNO submission candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import e007_verify_submission as gate

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "RealPDE-Competition-Data"

gate.CONFIG = REPO / "configs" / "e029_full_rollout_curriculum.json"
gate.WRAPPER = REPO / "submission" / "e029" / "submission.py"
gate.CHECKPOINT = REPO / "artifacts" / "e029_full_rollout" / "final.pth"
gate.CHECKPOINT_SHA256 = (
    "fc59848e8cc4d4d28458d6abbfb381b153cf964720c5e643d39862b19db492f8"
)
gate.NORMALIZER = REPO / "artifacts" / "e029_full_rollout" / "normalizer.json"
gate.NORMALIZER_SHA256 = (
    "305a51f7e629d4f9de2b035cbe07c98d8a1ec198fbce858497db27c78064bbb7"
)
gate.ARTIFACTS = REPO / "artifacts" / "e029_submission"
gate.ARCHIVE_NAME = "e029_candidate.zip"


def audit_candidate_inputs(config: dict) -> dict:
    missing = [
        str(path)
        for path in (gate.CONFIG, gate.WRAPPER, gate.CHECKPOINT, gate.NORMALIZER, gate.RUNNER)
        if not path.is_file()
    ]
    example = DATA / "example_data" / "3750_0.h5"
    if not example.is_file():
        missing.append(str(example))
    if missing:
        raise FileNotFoundError("missing E029 inputs:\n" + "\n".join(missing))

    checkpoint_hash = gate.sha256_file(gate.CHECKPOINT)
    normalizer_hash = gate.sha256_file(gate.NORMALIZER)
    if (
        checkpoint_hash != gate.CHECKPOINT_SHA256
        or checkpoint_hash != config["candidate"]["checkpoint_sha256"]
    ):
        raise AssertionError(f"unexpected checkpoint hash: {checkpoint_hash}")
    if normalizer_hash != gate.NORMALIZER_SHA256:
        raise AssertionError(f"unexpected normalizer hash: {normalizer_hash}")

    checkpoint = torch.load(gate.CHECKPOINT, map_location="cpu", weights_only=False)
    normalizer = json.loads(gate.NORMALIZER.read_text(encoding="utf-8"))
    if (
        checkpoint.get("experiment") != "E029"
        or checkpoint.get("status") != "fixed_budget_final"
        or checkpoint.get("model_kind") != "residual_cno"
    ):
        raise AssertionError("checkpoint is not the frozen E029 fixed-budget final residual_cno")

    wrapper = gate.load_module(gate.WRAPPER, "e029_source_audit")
    expected = {
        "mean_input": wrapper._MEAN_INPUT.tolist(),
        "std_input": wrapper._STD_INPUT.tolist(),
        "mean_target": wrapper._MEAN_TARGET.tolist(),
        "std_target": wrapper._STD_TARGET.tolist(),
    }
    for key, value in expected.items():
        if not np.allclose(
            np.asarray(value, dtype=np.float32),
            np.asarray(normalizer[key], dtype=np.float32),
            atol=1e-6,
        ):
            raise AssertionError(f"wrapper {key} differs from the frozen normalizer")

    uncertainty = config["candidate"]["uncertainty"]
    expected_floor = np.float32(uncertainty["beta"] * uncertainty["sigma_global"])
    if wrapper._ALPHA != np.float32(uncertainty["alpha"]):
        raise AssertionError("wrapper alpha differs from E029 config")
    if wrapper._ADDITIVE_HALF_WIDTH != expected_floor:
        raise AssertionError("wrapper additive interval floor differs from E029 config")
    if wrapper._BATCH_SIZE != config["candidate"]["point_batch_size"]:
        raise AssertionError("wrapper batch size differs from E029 config")
    if wrapper._BATCH_SIZE != 1:
        raise AssertionError("E029 must keep singleton micro-batching")

    expected_k = int(config["candidate"]["persist_first_frames"])
    if int(wrapper._PERSIST_FIRST_FRAMES) != expected_k:
        raise AssertionError("wrapper persist-first frame count differs from E029 config")

    dummy = np.zeros((2, 20, 32, 64, 3), dtype=np.float32)
    dummy[:, -1, ..., :2] = 0.3
    neural = np.ones((2, 20, 32, 64, 3), dtype=np.float32)
    hybrid = wrapper._persist_first(neural, dummy)
    if not np.allclose(hybrid[:, :expected_k, ..., :2], dummy[:, -1:, ..., :2]):
        raise AssertionError("persist-first did not copy the last input frame")
    if not np.allclose(hybrid[:, expected_k:, ..., :2], 1.0):
        raise AssertionError("persist-first mutated the neural tail")
    if not np.all(hybrid[..., 2] == 0.0):
        raise AssertionError("persist-first left non-zero pressure")

    return {
        "status": "passed",
        "checkpoint_bytes": gate.CHECKPOINT.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_experiment": checkpoint["experiment"],
        "checkpoint_status": checkpoint["status"],
        "model_kind": checkpoint["model_kind"],
        "normalizer_sha256": normalizer_hash,
        "normalizer_exact_float32_match": True,
        "uncertainty_parameters_exact_float32_match": True,
        "wrapper_sha256": gate.sha256_file(gate.WRAPPER),
        "persist_first_frames": expected_k,
        "persist_first_helper_pass": True,
    }


import shutil


if __name__ == "__main__":
    shutil.rmtree(gate.ARTIFACTS, ignore_errors=True)
    gate.audit_candidate_inputs = audit_candidate_inputs
    gate.main("E029")
