#!/usr/bin/env python3
"""Train the E025 Fold B h32 AdamW residual member."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e025_fold_b_adapter3.json",
        default_artifacts=REPO / "artifacts" / "e025_fold_b_adapter3",
    )
