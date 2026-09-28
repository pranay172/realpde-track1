#!/usr/bin/env python3
"""Build and gate the E011 all-public-data submission candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import e007_verify_submission as gate


REPO = Path(__file__).resolve().parents[1]

gate.CONFIG = REPO / "configs" / "e011_full_public_submission.json"
gate.WRAPPER = REPO / "submission" / "e011" / "submission.py"
gate.CHECKPOINT = REPO / "artifacts" / "e011_full_public" / "final.pth"
gate.NORMALIZER = REPO / "artifacts" / "e011_full_public" / "normalizer.json"
gate.ARTIFACTS = REPO / "artifacts" / "e011_submission"
gate.ARCHIVE_NAME = "e011_candidate.zip"


def audit_candidate_inputs(config: dict) -> dict:
    missing = [
        str(path)
        for path in (gate.CONFIG, gate.WRAPPER, gate.CHECKPOINT, gate.NORMALIZER, gate.RUNNER)
        if not path.is_file()
    ]
    example = gate.DATA / "example_data" / "3750_0.h5"
    if not example.is_file():
        missing.append(str(example))
    if missing:
        raise FileNotFoundError("missing E011 inputs:\n" + "\n".join(missing))

    checkpoint_hash = gate.sha256_file(gate.CHECKPOINT)
    normalizer_hash = gate.sha256_file(gate.NORMALIZER)
    gate.CHECKPOINT_SHA256 = checkpoint_hash
    gate.NORMALIZER_SHA256 = normalizer_hash
    if checkpoint_hash != config["candidate"].get("checkpoint_sha256", checkpoint_hash):
        raise AssertionError(f"unexpected checkpoint hash: {checkpoint_hash}")

    checkpoint = torch.load(gate.CHECKPOINT, map_location="cpu", weights_only=False)
    normalizer = json.loads(gate.NORMALIZER.read_text(encoding="utf-8"))
    if checkpoint.get("experiment") != "E011" or checkpoint.get("status") != "fixed_budget_final":
        raise AssertionError("checkpoint is not the frozen E011 fixed-budget final")
    if checkpoint.get("normalizer") != normalizer:
        raise AssertionError("checkpoint and standalone normalizer disagree")
    split_files = json.loads(
        (REPO / config["split_config"]).read_text(encoding="utf-8")
    )["training_files"]
    if set(checkpoint["provenance"]["training_files"]) != set(split_files):
        raise AssertionError("E011 checkpoint did not train on the frozen 81-file set")
    if checkpoint["provenance"].get("validation_field_values_accessed") != []:
        raise AssertionError("E011 provenance reports validation field access")

    wrapper = gate.load_module(gate.WRAPPER, "e011_source_audit")
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
            raise AssertionError(f"wrapper {key} differs from the trained normalizer")
    uncertainty = config["candidate"]["uncertainty"]
    expected_floor = np.float32(uncertainty["beta"] * uncertainty["sigma_global"])
    if wrapper._ALPHA != np.float32(uncertainty["alpha"]):
        raise AssertionError("wrapper alpha differs from the frozen interval")
    if wrapper._ADDITIVE_HALF_WIDTH != expected_floor:
        raise AssertionError("wrapper additive interval floor differs from the frozen interval")
    if wrapper._BATCH_SIZE != config["candidate"]["point_batch_size"]:
        raise AssertionError("wrapper batch size differs from E011 config")
    return {
        "status": "passed",
        "checkpoint_bytes": gate.CHECKPOINT.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_experiment": checkpoint["experiment"],
        "checkpoint_status": checkpoint["status"],
        "normalizer_sha256": normalizer_hash,
        "normalizer_exact_float32_match": True,
        "uncertainty_parameters_exact_float32_match": True,
        "wrapper_sha256": gate.sha256_file(gate.WRAPPER),
        "training_files": len(split_files),
    }


if __name__ == "__main__":
    gate.audit_candidate_inputs = audit_candidate_inputs
    gate.main("E011")
