#!/usr/bin/env python3
"""Train the preregistered E019 frozen-E005 residual adapter."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e019_residual_adapter_e005.json",
        default_artifacts=REPO / "artifacts" / "e019_residual_adapter_e005",
    )
