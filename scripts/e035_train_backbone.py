#!/usr/bin/env python3
"""Train the fixed-final E035 all-public long-budget CNO backbone."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e035_all_data_long_backbone.json",
        default_artifacts=REPO / "artifacts" / "e035_all_data_long_backbone",
    )
