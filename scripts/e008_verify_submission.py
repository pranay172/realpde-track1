#!/usr/bin/env python3
"""Run E008 through the reusable E007 submission acceptance gate."""

from __future__ import annotations

from pathlib import Path

import e007_verify_submission as gate


REPO = Path(__file__).resolve().parents[1]

gate.CONFIG = REPO / "configs" / "e008_batch_invariant_submission.json"
gate.WRAPPER = REPO / "submission" / "e008" / "submission.py"
gate.ARTIFACTS = REPO / "artifacts" / "e008_submission"
gate.ARCHIVE_NAME = "e006_candidate.zip"


if __name__ == "__main__":
    gate.main("E008")
