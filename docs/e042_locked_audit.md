# E042: locked conditional-interval audit

Registered 2026-09-21, after E041 exploratory screening. E040 remains paused.
Compute approval is pending; registration/preflight do not launch training.

## Question and roles

Does the locked E041 input-variability-conditioned channel/horizon interval rule
improve exact SPS on a genuinely disjoint real-data audit?

- Point training: 47 complete trajectories, excluding all A and C holdouts.
- Calibration: 17 Fold A trajectories, Re 3750/11400/20325/26700.
- Audit: 17 reserved Fold C trajectories, Re 5025/16500/22875/25425.

Lists are explicit in `configs/e042_locked_protocol.json` and enforced by
`configs/e042_point_split.json`. No role shares a Reynolds group. The duplicate
7575_0.h5 remains excluded. Released simulation-only initialization is permitted;
no existing real-finetuned checkpoint is clean for these three roles.

## Fixed execution order

1. Validate role membership, data presence, and hashes without loading C fields.
2. With compute approval, train one CNO from simulation initialization using
   E038's 3000-update recipe on the 47 point-training trajectories. Fit Gaussian
   normalization there only. Keep the existing training-only plateau rule.
   Expected cost ~3 GPU hours, maximum training wall budget 260 minutes; no
   automatic retries, adapters, seeds, or alternative architectures.
3. Freeze final/early-stopped checkpoint and ancestry. Generate calibration
   predictions on A, apply persist-first-4, fit the input-variability median
   and 12 width multipliers (two variability bins × three horizons × two
   channels). Candidate multipliers are [1, .5, .75, 1.25, 1.5, 2]; first wins
   ties. Unsupported bins retain baseline multiplier 1. Optimize exact masked
   SPS, including full-window accuracy weights; no target-derived inference
   features. Bounds scale the existing `.025*abs(point)+.15*SIGMA_GLOBAL` width.
4. Save and hash the fitted calibration artifact before opening C fields.
   Validate finite/ordered bounds, zero pressure, deterministic batch-invariant
   output, and unchanged point predictions on calibration/synthetic inputs.
5. Evaluate C once using the frozen point model and bounds. Compare baseline
   versus calibrated intervals on **identical** persisted point arrays. Report
   official quality metrics, SPS, coverage, widths, per-Re results and timing.
   The original point scores cannot change; only bounds and timing can differ.
6. Accept only if overall SPS gains >=0.3 and no Re loses >0.5 SPS, with all
   contract tests passing. Stop on failure; do not tune on C or pick another
   checkpoint, rule, adapter, or seed after seeing C results.

## Interpretation and deployment boundary

E041 used development folds; it is not an independent audit. Prior A/B models
also trained on C trajectories, so they cannot be reused for this clean audit.
C has not been used for learned-model ranking, though its fields appeared in
past development training; this is not an untouched external dataset.

E042 tests the interval mechanism on a backbone-only substrate. It does not
certify transfer to E037's all-public backbone plus adapter, and its point
scores are not a controlled comparison with E038 (different training set).
Even a pass requires a separate deployment/transfer decision; no archive,
all-public retraining, or submission is part of this protocol. No numerical
leaderboard final-score prediction is made.

Preflight: `../.venv-gpu/bin/python scripts/e042_protocol_preflight.py`.
The full calibration/freeze/audit runner must enforce the above artifact order
and be tested before consuming the one-time C audit. It is not yet implemented.
