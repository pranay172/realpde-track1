# Reproducing Track 1

**Status: research archive, partially reconstructable pipeline.**
The environments, full data, learned weights and submission artifacts were
deleted. Historical tests/results are preserved; no fresh end-to-end run was
performed during publication preparation.

## Workspace and dependencies

Use the historical directory layout even if the GitHub names use hyphens:

```text
RealPDE/
  .venv/                         # recreate; not shipped
  .venv-gpu/                     # recreate; not shipped
  RealPDE-Competition-Data/       # reacquire from authorized release
  RealPDE_T1/                    # clone realpde-track1 here
  RealPDE_T2/                    # clone realpde-track2 here
```

Only the track you need must be cloned. Copy its `environment-locks/` folder
to the parent RealPDE root and follow [the installation guide](environment-locks/README.md).
Snapshots pin Python 3.10.20, all installed dependencies and PyTorch indexes.
They are online version snapshots, not archived wheels or cryptographic wheel
locks. Linux x86-64 is the supported historical platform. CPU evaluation used
Torch 2.2.2+cu121; local RTX 5050 work used Torch 2.7.1+cu128.

Obtain the **competition** release (not an arbitrary newer benchmark dataset)
from [the authorized dataset source](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).
Restore `train_real/` and `baseline_checkpoints/sim_pretrain/` beneath the shared
data directory. Preserve release filenames and do not use `7575_0.h5`.
Data/checkpoint permissions remain separate from our code license.
No full dataset or trained solution checkpoint is included.

## Organizer dependencies (not bundled)

Before model imports, training, evaluation, or packaging, follow
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md) to obtain the exact historical
kit. Do not treat an absent dependency as a passing test. Some model imports
fail immediately without the kit. Kit downloads were not verified in this pass;
if you cannot obtain the required version legitimately, full reproduction is
blocked, while the source, results, and data-free checks remain usable.

## Data-free checks

From the track repository:

```bash
python3 scripts/check_archive.py
python3 scripts/publication_audit.py
```

These require only Python before Git initialization; they inspect source/JSON,
publication-document paths, requirement pins/checksums, and selected privacy
patterns. Once this snapshot has its own Git repository, the audit also checks
its reachable history. They do not
run models. GitHub CI runs the same archive check, not the full historical suite.


## Reconstruct the clean E038 experiment

With the exact kit and data restored, from `RealPDE_T1/`:

```bash
../.venv-gpu/bin/python scripts/e003_train_cno.py \
  --config configs/e038_fold_b_long_backbone.json \
  --artifacts artifacts/e038_fold_b_long_backbone --device cuda
```

This recreates the registered 3000-update backbone (historically ~3h8m on an
RTX 5050), not necessarily byte-identical weights. The saved-array evaluator
also requires regenerating the Fold B input/target/window manifests and E028
reference predictions; they were deleted. Do not claim that the training command
alone regenerates the recorded evaluation reports.

## Best scored E037: known reconstruction gap

E037 is **not E038/E043**. Its all-public backbone was trained with:

```bash
../.venv-gpu/bin/python scripts/e035_train_backbone.py --device cuda
```

The residual member originated from
`scripts/e033_train_all_data.py` using
`configs/e036_long_backbone_ensemble.json`. That trainer produces an ensemble
payload containing `backbone_state_dict`, `member1_adapter_state`,
`member1_adapter_config` and normalization. The E037 wrapper consumes a
single `residual_cno` checkpoint with prefixed `backbone.*` / `adapter.*`
state keys and adapter configuration.

A dedicated, tested conversion command for that historical assembly is absent.
The original assembled `artifacts/e037_robust_single/final.pth` was deleted.
Consequently **do not run the E037 packaging command and expect it to rebuild
the full model from scratch**. Reconstruct and test the assembly first, then
use the historical wrapper and exact-archive verification. We deliberately do
not advertise an untested invented command as a reproducible recipe.

E043's builder also depends on E038 predictions/evaluation and E041's source
configurations; those arrays/reports must be regenerated before fitting bounds.
E043 failed official evaluation; reproducing it is diagnostic work, not a
recommendation to submit it.

## Restored-environment tests

```bash
../.venv/bin/python -m unittest discover -s tests -p test_interval_crossfit.py
../.venv/bin/python -m unittest discover -s tests -p test_e043_submission.py
python3 -m unittest discover -s tests -p test_e042_protocol.py
```

The first two are synthetic/data-free but require restored packages.
The full suite includes data/checkpoint/configuration-specific tests; restore
their dependencies rather than suppressing failures. GPU/Docker tests require
host permissions and their documented image. Local runtime is not a guarantee
for the organizer's GPU, hidden inputs, or evaluation environment.

See [METHOD.md](METHOD.md), [RESULTS.md](RESULTS.md), [VALIDATION.md](VALIDATION.md)
and [LEDGER.md](LEDGER.md) for controls, exclusions and provenance caveats.
