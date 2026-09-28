# E043 experimental submission handoff

## Outcome update — 2026-09-28

The user confirms E043 was submitted and **failed evaluation**. No evaluator
error log, failure stage or numeric score was supplied, so the cause is unknown.
The local checks below are historical evidence only and did not establish server
success. Data, checkpoints and the candidate archive were deleted during cleanup.
E037 remains the best recorded scored Track 1 solution; E043 is not recommended
for resubmission without diagnosing the failure.

Candidate: `artifacts/e043_submission/e043_candidate.zip`

SHA-256: `651b7e74af1af1140acd392593c4b4b10e2c4ef3d36050518a86f6714aba5bd3`

29,580,288 bytes zipped; 32,385,879 bytes extracted (limit 256,000,000).

## Contents and provenance

- E038 3000-update Fold B CNO backbone; no residual adapter or new training.
- Source checkpoint SHA: `dd30981880136a895b37cd45b8db9aa16368c9ede20671dc220c872179376d49`.
- E038 normalization copied from its checkpoint; first four forecasts use
  last-frame persistence. Each point is inferred separately for batch invariance.
- Conditional intervals use the E041 mechanism, fitted on all 376 Fold B
  calibration windows (19 trajectories), separately from 62 point-training files.
- Input temporal-variability threshold: 0.005166921764612198, fixed at inference.
  Low-variability scales by horizon 1–4 / 5–10 / 11–20: .5 / .75 / .75;
  high-variability: .75 / 1 / 1. Both channels independently selected the same
  scales. Base half-width `.025*abs(point)+.15*SIGMA_GLOBAL`.
- `model.pth` contains the unchanged point state and normalizer only;
  `calibration.json` holds fitted scales and data/prediction hashes.

Build: `../.venv-gpu/bin/python scripts/e043_build_submission.py`. Builder
refuses an existing output directory; do not rebuild or replace this verified zip.

## Completed checks

- Six focused tests cover interval batch invariance, pressure/bounds, input
  rejection, persistence, group-disjoint fitting, and candidate selection.
- Two fresh official Torch 2.2.2 Docker containers: offline, read-only package,
  all arrays finite float32 with correct shape, exact repeat/batch/fresh equality.
- Localized input outliers at 20/50/100 sigma: maximum absolute forecasts
  0.351866 / 0.665946 / 1.082042; stress probes are boundedness evidence, not
  proof of arbitrary out-of-distribution quality.
- Exact zip GPU replay across 376 Fold B windows: 29.03 s warm; first cold sample
  3.17 s. Exact GPU batch/permutation checks and finite all-zero input test pass.
- Maximum point difference from E038 saved batch-4 predictions: 7.91e-5,
  within registered replay tolerance (rtol 1e-3, atol 1e-4). No weight or point
  recipe change; tiny differences arise between batching/execution paths.
- Calibration replay SPS 42.346364; fitted-statistic SPS 42.346353 agrees.
  Coverage 78.98683%. E041 group-crossfit SPS was 42.25795; these are distinct
  measurements, neither a new independent validation result for E043.
- E037 archive SHA unchanged: `0e799edde8660f1fd85e3f2eb17fad020788e18bfa3380465968cb1dcdc2236a`.

Reports: `build_report.json`, `docker_verification.json`, `gpu_replay.json`
under `artifacts/e043_submission/`.

## Limitations / user decision

E038 trained on 62 rather than E037's 81 trajectories. Fold B is now calibration,
so post-fit scores are optimistic; E043 is an experimental challenger, not a
proven upgrade. Local RTX 5050 timing is not an A800/hidden-set runtime guarantee.
No new independent audit, Fold C use, or score-above-80 guarantee is claimed.
E040 was paused and E042 unlaunched. The candidate was subsequently submitted
and failed evaluation; see the outcome update above. The zip is no longer local.
