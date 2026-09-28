#!/usr/bin/env python3
"""Train Adapter 3 (diverse seed 20260816, hidden_channels=32, AdamW) for Experiment E023."""

from pathlib import Path
from e003_train_cno import main

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e023_residual_adapter_seed2.json",
        default_artifacts=REPO / "artifacts" / "e023_adapter3",
    )
