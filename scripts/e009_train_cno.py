#!/usr/bin/env python3
"""Train the preregistered E009 out-of-fold CNO used only for residual calibration."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e009_oof_uncertainty.json",
        default_artifacts=REPO / "artifacts" / "e009_oof_uncertainty",
    )
