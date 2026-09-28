#!/usr/bin/env python3
"""Train the preregistered E020 Fold-B residual adapter."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e020_secondary_fold_residual.json",
        default_artifacts=REPO / "artifacts" / "e020_secondary_fold_residual",
    )
