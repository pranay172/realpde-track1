#!/usr/bin/env python3
"""Build and gate the E013 persist-first-4 submission candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import e007_verify_submission as gate


REPO = Path(__file__).resolve().parents[1]

gate.CONFIG = REPO / "configs" / "e013_persist_first_submission.json"
gate.WRAPPER = REPO / "submission" / "e013" / "submission.py"
gate.ARTIFACTS = REPO / "artifacts" / "e013_submission"
gate.ARCHIVE_NAME = "e013_candidate.zip"


_ORIGINAL_AUDIT = gate.audit_candidate_inputs


def audit_candidate_inputs(config: dict) -> dict:
    report = _ORIGINAL_AUDIT(config)
    wrapper = gate.load_module(gate.WRAPPER, "e013_persist_audit")
    expected_k = int(config["candidate"]["persist_first_frames"])
    if int(wrapper._PERSIST_FIRST_FRAMES) != expected_k:
        raise AssertionError("wrapper persist-first frame count differs from E013 config")
    if wrapper._BATCH_SIZE != 1:
        raise AssertionError("E013 must keep singleton micro-batching")
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
    report["persist_first_frames"] = expected_k
    report["persist_first_helper_pass"] = True
    return report


if __name__ == "__main__":
    gate.audit_candidate_inputs = audit_candidate_inputs
    gate.main("E013")
