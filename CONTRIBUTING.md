# Contributing to this research archive

This is an archived competition study, not a maintained production library.
Read AGENT.md for development conventions and LEDGER.md for historical evidence.

- Preserve result provenance and distinguish official, holdout, calibration and
  contaminated scores. Never invent a leaderboard aggregate.
- Do not commit datasets, credentials, checkpoints, generated archives or logs.
- Preserve upstream notices and check LICENSING.md before copying code.
- Keep configuration changes explicit; use whole-trajectory splits and tests.
- Run the data-free archive checks before proposing changes.
- Do not publish credential reports or private data in issues.

Training or inference changes require restored environments and appropriate
data/contract tests; passing syntax CI alone is insufficient.
