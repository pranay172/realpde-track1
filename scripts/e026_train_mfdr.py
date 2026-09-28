#!/usr/bin/env python3
"""Train Experiment E026: Decoupled Mean-Fluctuation (MFDR) + Log-Spectral Loss on Frozen E005 CNO."""

from pathlib import Path
from e003_train_cno import main

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e026_mfdr_residual_adapter.json",
        default_artifacts=REPO / "artifacts" / "e026_mfdr_residual",
    )
