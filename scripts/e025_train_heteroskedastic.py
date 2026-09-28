#!/usr/bin/env python3
"""Train the E025 Fold B heteroskedastic residual member."""

from pathlib import Path

from e022_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e025_fold_b_heteroskedastic.json",
        default_artifacts=REPO / "artifacts" / "e025_fold_b_heteroskedastic",
    )
