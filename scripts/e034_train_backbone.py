#!/usr/bin/env python3
"""Train the fixed-final E034 long-budget CNO backbone on Fold A without touching validation fields."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e034_long_backbone.json",
        default_artifacts=REPO / "artifacts" / "e034_long_backbone",
    )
