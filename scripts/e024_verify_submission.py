#!/usr/bin/env python3
"""Build and gate the E024 Shared-Backbone Residual Ensemble submission candidate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

import e007_verify_submission as gate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "realpde_t1_starting_kit_v9"))
from realpde_t1.ensemble import assert_bundle_matches_sources, load_member_checkpoints

DATA = REPO.parent / "RealPDE-Competition-Data"
SOURCE_CHECKPOINTS = [
    REPO / "artifacts" / "e019_residual_adapter_e005" / "final.pth",
    REPO / "artifacts" / "e022_heteroskedastic_residual" / "final.pth",
    REPO / "artifacts" / "e023_adapter3" / "final.pth",
]

gate.CONFIG = REPO / "configs" / "e024_shared_backbone_ensemble_submission.json"
gate.WRAPPER = REPO / "submission" / "e024" / "submission.py"
gate.CHECKPOINT = REPO / "artifacts" / "e023_ensemble" / "ensemble_bundle.pth"
gate.CHECKPOINT_SHA256 = (
    "dfbd15d93d8b6324e06b6d5e4f66ad004e5441b825b1708465719f71dc0ff10b"
)
gate.NORMALIZER = REPO / "artifacts" / "e005_scale_balanced_uv" / "normalizer.json"
gate.ARTIFACTS = REPO / "artifacts" / "e024_submission"
gate.ARCHIVE_NAME = "e024_candidate.zip"


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
        raise FileNotFoundError("missing E024 inputs:\n" + "\n".join(missing))

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
        checkpoint.get("model_kind") != "shared_backbone_residual_ensemble"
        or "backbone_state_dict" not in checkpoint
        or "adapters" not in checkpoint
    ):
        raise AssertionError("checkpoint is not a valid shared_backbone_residual_ensemble")

    wrapper = gate.load_module(gate.WRAPPER, "e024_source_audit")
    expected = {
        "mean_input": wrapper._MEAN_INPUT.tolist(),
        "std_input": wrapper._STD_INPUT.tolist(),
        "mean_target": wrapper._MEAN_TARGET.tolist(),
        "std_target": wrapper._STD_TARGET.tolist(),
    }
    for key, value in expected.items():
        if not np.array_equal(
            np.asarray(value, dtype=np.float32), np.asarray(normalizer[key], dtype=np.float32)
        ):
            raise AssertionError(f"wrapper {key} differs from the frozen normalizer")

    uncertainty = config["candidate"]["uncertainty"]
    expected_floor = np.float32(uncertainty["beta"] * uncertainty["sigma_global"])
    if wrapper._ALPHA != np.float32(uncertainty["alpha"]):
        raise AssertionError("wrapper alpha differs from E024 config")
    if wrapper._ADDITIVE_HALF_WIDTH != expected_floor:
        raise AssertionError("wrapper additive interval floor differs from E024 config")
    if wrapper._BATCH_SIZE != config["candidate"]["point_batch_size"]:
        raise AssertionError("wrapper batch size differs from E024 config")
    if wrapper._BATCH_SIZE != 1:
        raise AssertionError("E024 must keep singleton micro-batching")
    if any(not path.is_file() for path in SOURCE_CHECKPOINTS):
        raise FileNotFoundError("E024 ensemble source checkpoints are missing")
    members, _ = load_member_checkpoints(SOURCE_CHECKPOINTS)
    assert_bundle_matches_sources(checkpoint, members)

    expected_k = int(config["candidate"]["persist_first_frames"])
    if int(wrapper._PERSIST_FIRST_FRAMES) != expected_k:
        raise AssertionError("wrapper persist-first frame count differs from E024 config")

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
        "model_kind": checkpoint["model_kind"],
        "adapters_count": len(checkpoint["adapters"]),
        "normalizer_sha256": normalizer_hash,
        "normalizer_exact_float32_match": True,
        "uncertainty_parameters_exact_float32_match": True,
        "wrapper_sha256": gate.sha256_file(gate.WRAPPER),
        "persist_first_frames": expected_k,
        "persist_first_helper_pass": True,
        "source_bundle_parity": True,
        "uncertainty": "frozen_e006_interval_not_learned_log_variance",
    }


if __name__ == "__main__":
    gate.audit_candidate_inputs = audit_candidate_inputs
    gate.main("E024")
