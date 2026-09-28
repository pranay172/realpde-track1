#!/usr/bin/env python3
"""Train the preregistered E010 complementary-fold robustness CNO."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e010_secondary_fold_robustness.json",
        default_artifacts=REPO / "artifacts" / "e010_secondary_fold",
    )
