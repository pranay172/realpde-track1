# RealPDE Track 1 — Sim-to-Real Flow Forecasting

Research code and experiment records for our NeurIPS 2026 RealPDE Track 1
participation. Predict 20 future velocity frames from 20 observed frames using
simulation-initialized convolutional neural operators (CNOs).

**Best reported development-leaderboard final_score: 77.307902 (E037).**
This is a user-reported official score, not a final private-test ranking.
The later E043 submission **failed evaluation**; no failure log or cause is
available. Passing local checks did not establish evaluator success.

## Approach

- Simulation-initialized CNO with scale-balanced physical u/v training loss.
- Longer real-data fine-tuning, validated on whole-regime splits.
- A frozen backbone plus one rollout-trained residual adapter for E037.
- Last-frame persistence for the first four predicted frames.
- Calibrated symmetric uncertainty bounds; later conditional-width experiments
  are documented separately and are not the best officially scored solution.

Read [METHOD.md](METHOD.md), [RESULTS.md](RESULTS.md), and
[REPRODUCING.md](REPRODUCING.md). Experiment IDs are track-local.

## What is available

Participant source, configurations, tests, historical results and hashes,
and pinned [environment snapshots](environment-locks/README.md).
Full competition data, learned checkpoints, and generated submission archives
were deleted during cleanup. This is **not a download-and-run pretrained model**.
Fresh end-to-end reproduction has not been performed after cleanup; E037's
checkpoint-assembly gap is explicitly documented.

Organizer code, copied competition pages, and the original Git history are
intentionally omitted. Obtain required dependencies using
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).

## Start here

```bash
# No Python packages, GPU, or competition data required:
python3 scripts/check_archive.py
python3 scripts/publication_audit.py
```

For installation/data/training, follow [REPRODUCING.md](REPRODUCING.md).
The sibling project is `realpde-track2` (streaming adaptation); clone this
repository locally as `RealPDE_T1` for the historical workspace layout.

## Repository map

- `src/realpde_t1/`, `scripts/`, `submission/`: research and deployment code.
- `configs/`, `tests/`: experimental controls and regression checks.
- [LEDGER.md](LEDGER.md), `docs/`: full history, including failures and caveats.
- [TRACK1_REFERENCE.md](TRACK1_REFERENCE.md), [VALIDATION.md](VALIDATION.md):
  dated protocol/reference notes, not live competition status.
- [AGENT.md](AGENT.md): development conventions.
- [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md): release checks and open issues.

## Licensing and credit

MIT for original contributions only; bundled/derived code has separate,
including non-commercial, restrictions. **Do not treat the whole repository as
MIT.** Read [LICENSING.md](LICENSING.md) and
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md); the path-level inventory and preserved
license copies are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This source-only snapshot omits organizer bundles and private Git history.
See [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) for release checks and limitations.
