# Score-80 development route

Archive update (2026-09-28): this document records the historical development
plan, not an active queue. E040 was paused, E042 was not run, and E043 was
submitted but failed evaluation (user report; error log unavailable). E037's
77.307902 remains the best recorded development score. Data/artifacts and
environments were removed; see the root publication/reproduction documents.

Registered 2026-09-21. E037 remains the immutable scored baseline (77.307902).
No local aggregate is estimated from fitted leaderboard weights.

## Current priority: uncertainty, not further backbone scaling

Next protocol registered: [E042 locked audit](e042_locked_audit.md). Metadata
preflight passes for 47 point-training / 17 calibration / 17 audit trajectories.
A new ~3-hour point-training run is necessary for clean C ancestry; explicit
compute approval is pending. No E042 training or audit has started.

User paused E040 on 2026-09-21. Its update-400 recovery (model, optimizer,
scheduler, normalizer) is preserved, SHA
`b23b13d92c13657f084b0c46b2a2666a8812fc7ee5c4266c5d49b1a94d4bf714`.
All three E040 processes were stopped; a pause marker blocks accidental restart.
There is no E040 final/validation result. Resume would lose the few updates
since recovery and has the trainer's documented batch-order reproducibility caveat.

E041 screens interval scaling on existing predictions without point-model
training. It leaves one nominal-Re group out, fits six candidate width
multipliers per channel/horizon on the other groups, and optionally separates
windows by input temporal variability (median fitted on calibration groups).
Neither thresholds nor width selection use held-group target statistics.
A and B residuals never mix; complete regimes, not windows, define the split.

| Variant | A SPS (delta) | B SPS (delta) |
|---|---:|---:|
| Existing bounds | 42.72081 | 41.26547 |
| Channel/horizon | 44.08455 (+1.36373) | 41.75503 (+0.48956) |
| + input variability | 44.27115 (+1.55033) | 42.25795 (+0.99248) |

Both pass the registered +0.3 per-fold / max 0.5 Re-regression screen.
The conditioned variant improves every Re group (minimum gains +0.232 A,
+0.273 B). Coverage is 80.39% A / 79.09% B; mean scored-element full widths
0.01636 / 0.01818. SPS gains trade coverage for tightness; they are not a claim
of nominal statistical coverage. Deterministic predictions remain byte-unchanged.

**Limit:** development folds previously informed point-model selection. This is
cross-fitted exploratory evidence, not a pristine nested OOF/final audit, and
the report's held-group fits are not a deployable calibration model. No new
weights or submission are generated. Next lock this simple mechanism and build
a disjoint point-training / calibration / audit ancestry. Existing all-public
weights cannot provide a clean public audit, and existing A/B backbones must
not be assumed clean on Fold C. Account for any required fresh training budget
before launch. Do not spend six hours on E040 merely to obtain interval data.

## Training-only early stopping (user amendment, 2026-09-21)

`scripts/backbone_plateau.py` now monitors the running E038 backbone and wraps
future queued backbone training. Every 100 updates it compares the trailing
200-update mean training loss to the best materially improved mean. A decrease
of at least 1% resets patience; after at least 1000 total updates, 600 updates
without such improvement stops training. Validation is never consulted.
The best qualifying recovery state is copied unchanged to an explicitly labelled
`early_stopped_final`; both best and latest recoveries remain available. Policy,
decision, selected update, and trigger update are recorded beside the checkpoint.
If the budget finishes first, the original fixed-budget final is preserved.

This changes the stopping protocol, not the live configuration hash. An early
E038 result can be evaluated and used by E039B but is **not** a matched-budget
replication; the coordinator skips E040 pending review in that case. A training
loss plateau is a compute-saving heuristic, not proof of validation stagnation.

## Completed diagnostics

### Stage-1 results and recovery (2026-09-21)

E038 completed 3000 updates without early stopping (11308.43 s). Fold B
persist-first-4 scores: Rel-L2 95.559614, TKE 75.386750, MVPE 96.064382,
SPS 41.265471; all aggregate and slice gates passed against E028.
E039B completed 600 updates (208.47 s): 95.601625 / 75.250552 / 96.136419 /
41.545223. Its -0.136197 TKE delta fails the registered -0.1 gate despite small
improvements elsewhere. E039A remains accepted on A; the adapter is not a
confirmed all-metric improvement across both folds.

E040 is still justified by E038's successful backbone gate, independently of
E039B. Its first launch encountered a monitor startup race: an empty process
command line was interpreted as exit before the child subsequently logged
normalization. Owned-child liveness now uses the process handle, failure cleanup
is bounded, and heartbeat/PID records distinguish stale queue state. Startup
diagnostics are preserved separately; the restart keeps the registered recipe.
No learned uncertainty, all-public retraining, or packaging is automatically
authorized by these local results.

`scripts/sps_headroom.py` reads saved predictions in batches and reproduces the
official SPS numerator/denominator. Each window retains its full-horizon
Rel-L2/TKE/MVPE accuracy factors when sliced by lead time. Nonzero target counts
weight aggregation; coverage is not averaged equally across windows.

| Clean model/fold, persist-first-4 | SPS | Symmetric oracle | Accuracy-only ceiling | Coverage |
|---|---:|---:|---:|---:|
| E034 / A | 42.5074 | 60.5172 | 72.4723 | 84.7047% |
| E028 / B | 38.0333 | 55.7797 | 70.0082 | 78.8479% |

The symmetric oracle uses target-informed width `abs(target - prediction)`.
It is an upper bound at fixed predictions, not an achievable calibration claim
or a deployable method. A/B rows are different recipes and cannot estimate the
long-backbone cross-fold effect. E037's contaminated replay is excluded.

E034 coverage by Re is 97.38 / 87.98 / 79.77 / 70.51% for
3750 / 11400 / 20325 / 26700. This motivates conditioning intervals on input
features, but improvement must be measured with an independently fitted rule.

Detailed reports:

- `artifacts/e038_sps_diagnostics/fold_a_e034.json`
- `artifacts/e038_sps_diagnostics/fold_b_e028.json`

The non-mutating E037 verifier also passed in two fresh official containers:
all prediction/bound outputs were exactly identical; stress output maxima were
0.308 / 0.543 / 0.832 for sigma 20 / 50 / 100. Archive SHA stayed
`0e799edde8660f1fd85e3f2eb17fad020788e18bfa3380465968cb1dcdc2236a`.
Report: `artifacts/e038_sps_diagnostics/e037_immutable_verification.json`.

## Registered execution

E039A has completed and passed all gates: hybrid Rel-L2/TKE/MVPE/SPS
`95.648237/75.048099/96.077909/42.720815`, versus E034 deltas
`+0.032448/+0.253239/+0.049991/+0.213392`. Every Re slice improved on the
three deterministic metrics. Training took 489.88 s while sharing the GPU with
E038; its local time score is not a usable speed comparison. Raw output trades
higher Rel-L2 (95.697) for lower TKE (73.875), MVPE (95.997), and SPS (42.524).
Full report: `artifacts/e039a_long_backbone_rollout/evaluation.json`.

1. E038: 3000-update backbone on Fold B. Exact E010 training controls except
   budget and cosine schedule length. Compare against saved E028 points using
   the official scorer; all four quality deltas must be nonnegative. Each Re
   permits at most 0.25 Rel-L2/MVPE and 1.0 TKE regression.
2. E039A/B: frozen long backbone plus the original 600-update rollout adapter.
   Compare against the corresponding backbone, including raw and persistence
   rows. Gate: nonnegative Rel-L2/MVPE/SPS, TKE delta >= -0.1, same slice limits.
   Backbone training-file provenance must exactly match the adapter fold.
3. E040: if E038 passes, restart from the sim checkpoint for 6000 Fold A updates
   with a stretched cosine schedule. A continuation from a zero-LR final is a
   different experiment and is not used. Require Rel-L2 +0.1, SPS +0.3,
   nonnegative MVPE, TKE delta >= -0.1, and the same slice limits.

The runner is `scripts/run_score80_stage1.py`. `--adopt-running` waits for the
already launched E038/E039A jobs; the default launches missing training jobs.
An exclusive lock prevents duplicate queues. Reports are hash-checked and
evaluation refuses to overwrite a prior prediction file or evaluation report.

Live status: `artifacts/score80_stage1/state.json`. Child logs are in that same
directory. A failed process stops the queue; `complete_review_required` means
the registered stage finished, not that a candidate is approved for submission.

The 3000-update backbone previously took about three hours. The conditional
6000-update ablation may take about six hours. Simultaneous work can affect
timing; these timings must not be interpreted as leaderboard runtime scores.

## Decisions after these results

- If E040 improves, run a matched 6000-update Fold B confirmation before
  selecting its recipe for all-public training.
- Fit conditional intervals only from training-only residuals, preferably
  nested OOF predictions: the outer validation trajectories must not train
  either the point predictor producing residuals or the uncertainty model.
  Exchanging A/B residuals naively is not clean: an A backbone was trained on
  B's validation trajectories. Start with a small channel/horizon/input-variance
  rule before adding a convolutional uncertainty head.
- The existing SmoothSPSLoss is historical. A future scorer-aligned objective
  must mask zero targets in all fitted terms and account for the official
  per-window accuracy weighting. Report coverage and width alongside SPS;
  require unchanged point predictions for an interval-only experiment.
- Preserve Fold C for a locked recipe check. Check any calibration model's
  training ancestry against C as well as the point model's ancestry.
- `scripts/verify_existing_submission.py` now validates an existing zip without
  rebuilding it: expected SHA, extraction size/path checks, two fresh offline
  read-only official-image runs, all-output finiteness/dtype/shape, batch and
  repeat equality, real input windows, and localized-outlier probes. Stress
  magnitudes are recorded; finite values alone do not establish tail quality.
  Full quality scoring and whole-run GPU timing remain separate requirements.
- Only an accepted recipe proceeds to all-public retraining and exact-archive
  verification. Upload is a separate user action; nothing in this runner submits.

No fixed subscore tuple is guaranteed to reach 80 because the official final
aggregation is unpublished.
