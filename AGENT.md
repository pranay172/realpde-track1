# Track 1 Agent Onboarding and Development Guide

This repository is the primary workspace for RealPDE Competition Track 1:
simulation-to-real forecasting of airfoil flow. This document is the operating
contract for any new agent or contributor. Keep it aligned with the repository
as the project evolves.

## Public snapshot boundary

Read `EXTERNAL_DEPENDENCIES.md` before running models. The organizer kit,
example data, copied pages, host-specific handoff, and private Git history are
not shipped. Do not re-add them or claim that restoring a kit grants rights to
redistribute it. Historical experiment notes retain old artifact paths and
commit hashes as provenance, not promises that those assets are present.

## Start here

Read these in order before changing code or proposing an experiment:

1. `AGENT.md` — workflow, conventions, and guardrails.
2. `LEDGER.md` — current state, active ideas, recent experiments, and decisions.
   Follow its links into `docs/ledger_archive/` when older evidence is needed.
3. `README.md` — workspace entry points and local assets.
4. `VALIDATION.md` — frozen split, local baseline table, and model-selection
   interpretation rules.
5. `TRACK1_REFERENCE.md` — consolidated official task, data, rules, scoring,
   submission contract, schedule, discrepancies, and environment audit.
   `TRACK1_LITERATURE_NOTES.md` is the Track 1 paper filter; it does not
   override this ledger or the official reference.
6. `environment-locks/README.md` — shared Track 1/2 environments, GPU/Docker facts,
   Codex sandbox behavior, and verified smoke tests in this local workspace.
7. `realpde_t1_starting_kit_v9/README.md`, `scoring.py`, and
   `submission_example_fno.py` — executable organizer-facing contract.

If live competition information conflicts with a note, verify the current
Codabench/competition source, update the relevant reference, and record the
change in `LEDGER.md`.

## Non-negotiable Track 1 facts

- Input and output shapes are `(N, 20, 32, 64, 3)` in `[u, v, p]` order. Only
  real `u` and `v` are scored; real pressure is represented as zero.
- Work from starting kit v9 and the restarted phase beginning 2026-08-05.
  Archived-phase scores are not comparable.
- Exclude `train_real/7575_0.h5` everywhere; it duplicates `6300_0.h5` fields
  while carrying an incorrect Reynolds-number scalar.
- Split validation by complete trajectory/parameter regime. Never create train
  and validation windows from the same trajectory.
- Each prediction may depend only on its own input and learned training state.
  Do not exploit adjacency or other evaluation windows, persistent caches,
  hidden files, or batch-level retrieval.
- Training inputs and weights must trace to the released competition data.
  External datasets, externally pretrained weights, and newly generated
  simulation trajectories are prohibited. Consult `TRACK1_REFERENCE.md` before
  adding any augmentation or learned preprocessing.
- Record AI assistance and external development resources. Organizer guidance
  on Codex disclosure was still unresolved at the last reference update.

## Repository structure

Current tracked structure:

```text
RealPDE_T1/
├── AGENT.md                    # this onboarding and process guide
├── LEDGER.md                   # live experiments, ideas, results, decisions
├── README.md                   # workspace entry point
├── TRACK1_REFERENCE.md         # consolidated competition reference
├── TRACK1_LITERATURE_NOTES.md  # Track 1 paper filter and transplant ranking
├── docs/ledger_archive/        # compacted historical ledger rows
├── VALIDATION.md               # frozen local validation protocol and anchors
├── configs/                    # versioned experiment/split configurations
├── scripts/                    # experiment and packaging entry points
├── src/realpde_t1/             # reusable participant-side utilities
├── tests/                      # focused unit/integration tests
└── realpde_t1_starting_kit_v9/ # external dependency, ignored and not shipped
```

Preferred structure for new work:

```text
configs/       versioned experiment configurations
scripts/       thin train/evaluate/package entry points
src/           reusable data, model, training, and inference modules
tests/         focused unit and integration tests
submission/    minimal evaluator-facing code before packaging
artifacts/     ignored local weights, predictions, plots, and logs
```

Treat the extracted starting kit as a reference implementation. Avoid modifying
organizer files unless fixing or instrumenting the kit itself; put competition
work in the directories above. If an organizer file must change, document why
and keep the diff minimal.

Large assets live outside this Git repository:

- shared data: `../RealPDE-Competition-Data/`;
- simulation checkpoints: `../RealPDE-Competition-Data/baseline_checkpoints/sim_pretrain/`;
- simulation-to-real checkpoints: `../RealPDE-Competition-Data/baseline_checkpoints/sim_real_ft/`.

Do not commit virtual environments, datasets, HDF5 files, model weights,
generated predictions, logs, or submission archives.

## Environment selection

For reconstruction after cleanup, use the tracked `environment-locks/README.md`.
Install the environments in the parent project root, shared
between tracks; dependency snapshots do not restore data or model artifacts.

From the parent project root (`RealPDE/`):

```bash
# Exact evaluator-compatible checks (local CPU)
source .venv/bin/activate

# RTX 5050 development and training
source .venv-gpu/bin/activate
```

- `.venv`: Python 3.10.20, Torch 2.2.2+cu121. Preserve it unchanged. Its Torch
  wheel cannot execute on the local Blackwell `sm_120` GPU.
- `.venv-gpu`: Python 3.10.20, Torch 2.7.1+cu128. Use it for local GPU work.
- Submission code must remain Python 3.10/Torch 2.2.2 compatible even when it is
  developed under Torch 2.7.1. Prefer portable `state_dict` checkpoints.
- The default Codex sandbox hides the GPU and Docker socket. Request an approved
  host-level command for `nvidia-smi`, CUDA tests/training, or Docker. Do not
  diagnose an unavailable sandbox device as a broken host driver.

Use the package/CUDA smoke commands in `environment-locks/README.md` after
restoring the environments. Participant helpers inside the private kit are not
shipped in this snapshot. GPU/Docker checks still require host permissions.

## Development process

### 1. Establish the question

- Read the current ledger and reference before proposing work.
- State one falsifiable hypothesis and a comparison baseline.
- Create or reserve an `E###` ledger entry before a meaningful run.
- Specify the whole-trajectory split, metrics, resource budget, and stopping
  condition. Avoid broad exploratory runs without a decision they can inform.

### 2. Implement the smallest reproducible change

- Put reusable logic in `src/`, configuration in `configs/`, and entry points in
  `scripts/`.
- Use relative/configured paths, never machine-specific absolute paths.
- Make device and dtype placement explicit. Seed Python, NumPy, and Torch.
- Keep data-loader, preprocessing, normalization, mask handling, and temporal
  window construction testable independently.
- Preserve channel order and shapes at module boundaries.
- Avoid dependencies absent from the evaluator. Vendor only suitable pure-Python
  code and test with the evaluator package blockers in the starter kit.

### 3. Validate in increasing cost order

1. Syntax/import and focused unit tests.
2. Tiny CPU batch with shape, finiteness, and determinism assertions.
3. Single GPU batch and checkpoint round-trip in `.venv-gpu`.
4. Leakage-safe validation with all five published subscores: relative L2, TKE,
   MVPE, time, and SPS. Never invent a local aggregate score when the official
   aggregation formula is unpublished.
5. Evaluator-compatible import and representative forward in `.venv`.
6. Submission archive validation: root-level `submission.py`, extracted size
   below 256 MB, no network/install step, finite exact-shape output, and total
   execution below five minutes.
7. Candidate robustness: repeat in fresh processes, compare supported batch
   sizes, replay representative real inputs, and probe bounded perturbations,
   missing values/masks, and localized outliers relevant to PIV data.

Use `torch.utils.data.DataLoader(num_workers=0)` in submission/evaluator paths.
Check both ordinary and edge cases: batch size, empty/zero pressure, PIV masks,
multiple fresh `predict` subprocesses, and absent metadata.

### Submission artifact discipline

- Validate the exact archive that will be uploaded, preferably after fresh
  extraction in the evaluator image with networking disabled and the package
  mounted read-only.
- Record SHA-256 hashes for the archive and material checkpoint/configuration
  inputs. A rebuilt or recompressed archive is a new artifact and must be
  revalidated.
- Keep the last scored baseline immutable. Build new candidates in separate
  directories and never silently replace accepted weights or wrappers.
- Treat contract checks, local quality, robustness probes, and hidden
  leaderboard performance as distinct evidence. Passing one does not imply the
  others.
- Quarantine a failed candidate until its failure mode is understood; do not
  reuse its wrapper or calibration merely because ordinary local inputs pass.

### 4. Record and decide

After every substantive conversation or attempted approach:

- update the `LEDGER.md` date and current snapshot;
- fill in the experiment result, including failures and uncertainty;
- record the commit/config/seed/split/environment/hardware/runtime;
- add or close ideas and record durable decisions;
- state the next action compactly.

Keep the live ledger focused on roughly the latest 10–15 experiments and
decisions plus the active queue. Move settled verbatim rows into a dated file
under `docs/ledger_archive/`, link it from the live ledger, and preserve IDs,
metrics, provenance, negative results, and reconstruction pointers. Never reuse
an archived ID or rewrite a historical result to match a later interpretation.

An approach is not complete until its code/tests and ledger entry agree. Raw
logs may be cleaned; measured outcomes and provenance must remain recoverable.

### 5. Commit coherently

- Inspect `git status` before and after work; preserve unrelated user changes.
- Keep commits small and purpose-specific. Do not mix an experiment result with
  unrelated refactors.
- Include necessary tests/config/docs with the implementation.
- Use imperative messages such as `Add trajectory-level validation split`.
- Never commit secrets, credentials, data, weights, or generated archives.

## Experiment reporting minimum

Every completed `E###` entry should make the following reconstructable:

- hypothesis and comparator;
- Git commit and configuration;
- exact data provenance, exclusion list, and split manifest;
- random seed(s), environment, device, and precision;
- trainable parameter count, training time, and peak memory when material;
- official validation subscores and the official aggregate when available;
- inference timing method and extracted submission size;
- conclusion, limitations, and next action.

Prefer tables and short conclusions. Store large plots/logs under ignored
artifact directories and reference their path or external run ID in the ledger.

## Coding preferences

- Target Python 3.10 and favor clear, composable functions over monolithic
  notebooks or scripts.
- Use `pathlib`, explicit configuration, type hints where they clarify tensor or
  data contracts, and concise docstrings for non-obvious behavior.
- Keep tensor-layout transformations named and asserted; silent permutation bugs
  are expensive here.
- Fail early on missing/duplicate trajectories, non-finite tensors, wrong shapes,
  or unexpected HDF5 schemas. Loaders should defensively support both observed
  top-level fields and the documented `measured_data/` grouping when practical.
- Optimize only after profiling. Report both model quality and evaluation cost;
  the competition score includes time and uncertainty quality.
- Prefer deterministic, reversible changes and preserve a runnable baseline.

## Definition of done

A change is ready to hand off when:

- its intended behavior and comparison are clear;
- focused tests and a representative forward pass succeed;
- evaluator compatibility has been checked when submission code is affected;
- no forbidden or generated artifacts are staged;
- `LEDGER.md` reflects the result and next decision;
- relevant onboarding/reference documentation remains accurate.
