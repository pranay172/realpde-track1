#!/usr/bin/env python3
"""Train the preregistered E005 scale-balanced CNO ablation."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e005_scale_balanced_uv.json",
        default_artifacts=REPO / "artifacts" / "e005_scale_balanced_uv",
    )
