#!/usr/bin/env python3
"""Evaluate the fixed-final E005 checkpoint exactly once on frozen E002."""

from pathlib import Path

from e003_evaluate_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_artifacts=REPO / "artifacts" / "e005_scale_balanced_uv",
        expected_experiment="E005",
    )
