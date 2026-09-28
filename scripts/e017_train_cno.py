#!/usr/bin/env python3
"""Train the preregistered E017 wake-weighted CNO ablation."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e017_wake_weighted_loss.json",
        default_artifacts=REPO / "artifacts" / "e017_wake_weighted_loss",
    )
