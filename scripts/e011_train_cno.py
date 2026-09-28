#!/usr/bin/env python3
"""Train the preregistered E011 all-public-data CNO."""

from pathlib import Path

from e003_train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "e011_full_public_submission.json",
        default_artifacts=REPO / "artifacts" / "e011_full_public",
    )
