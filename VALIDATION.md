# Track 1 local validation protocol

This is the frozen local model-selection protocol established by E002. The
machine-readable split is `configs/e002_split.json`; do not alter it in place
after using it for model selection. Create a new experiment and split file if a
second fold or robustness study is needed.

Current scored status lives in `LEDGER.md`; E037 is the active Codabench
baseline at `77.307902` final. Labels such as “baseline” or “candidate” in the
historical sections below describe the state when that experiment was run, not
the current submission recommendation.

## Frozen E002 split

- Seed: `20260815`
- Split-config SHA-256:
  `91ca400e363b9cff0af7409ac04795371c57551e0dd52506ff6f548a7d352df4`
- Audited data-inventory SHA-256:
  `afa230ea6a864d025700bf40fb75e0b540e09fc6f56f5bbc1a58902b213cf9e2`
- Training: 64 complete real trajectories.
- Validation: 17 complete real trajectories.
- Excluded: `7575_0.h5`, the known duplicate/bad-Re file.
- No nominal Reynolds number occurs in both training and validation.

Validation covers two interpolation and two edge/extrapolation Reynolds groups:

| Stratum | Nominal Reynolds groups | Trajectories | Windows |
|---|---|---:|---:|
| ID interpolation | 11400, 20325 | 10 | 210 |
| OOD edge | 3750, 26700 | 7 | 147 |
| **Total** | 4 groups | **17** | **357** |

All validation trajectories have 868 frames. A sample uses frames `[s,s+20)`
as input and `[s+20,s+40)` as target for `s = 0,40,...,800`. Therefore no raw
frame appears in two validation samples. Fields are spatially subsampled by two
to the official `(20,32,64,3)` shape, with pressure set exactly to zero.

This split tests transfer to real measurements absent from real-data training.
Using released simulation at the same regimes remains valid for the competition's
sim-to-real setting, but held-out real fields, statistics, masks, and targets
must never influence fitting or selection.

## E002 baseline table

All quality metrics below come directly from starter-kit v9 `scoring.py` with
default uncertainty intervals. Times are mean local inference time per sample
after one warm-up, using batch size four. The neural runs used the local RTX
5050 with Torch 2.7.1+cu128. Local time scores are not comparable to Codabench's
A800 time score, and the unpublished leaderboard `final_score` cannot be
reconstructed locally.

| Model | Evidence status | Rel-L2 | TKE | MVPE | Time | SPS | ms/sample | Peak GPU | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | leakage-safe | 93.771 | 66.667 | 93.819 | 98.636 | 16.989 | 0.139 | 0 | 0 |
| Released CNO `sim_real_ft` | contaminated reference | 95.760 | 74.411 | 96.078 | 76.592 | 19.697 | 68.089 | 506.3 MiB | 7,960,467 |
| Released FNO fp16 `sim_real_ft` | contaminated reference | **96.440** | **76.788** | **96.965** | **88.864** | **21.498** | 11.447 | 947.1 MiB | 50,357,955 |
| Released Transolver `sim_real_ft` | contaminated reference | 93.774 | 70.995 | 94.231 | 68.056 | 13.252 | 160.602 | 1,818.8 MiB | 12,541,259 |

The released real-finetuned checkpoints were trained on the complete public real
release, including these 17 validation trajectories. Their rows reproduce and
compare the organizer assets, but they are not valid estimates of unseen-real
generalization and must not be used as the “best leakage-safe local score.”

The edge stratum is consistently harder. For example, persistence SPS falls
from 17.549 on ID to 16.185 on OOD. Released FNO remains the best contaminated
reference with ID/OOD SPS 22.469/20.105, while released Transolver drops to OOD
SPS 8.352.

Checkpoint sizes are 32,154,928 bytes (CNO), 201,396,349 bytes (packed FNO), and
50,401,485 bytes (Transolver). With the kit's 451,300-byte submission code tree,
their projected extracted footprints are about 32.61 MB, 201.85 MB, and 50.85
MB. E001 independently built and verified the FNO package at exactly 201,847,649
bytes; CNO and Transolver figures are projections until candidate wrappers are
packaged.

## Reproduce

From `RealPDE_T1/`, prepare/audit the data on CPU:

```bash
../.venv/bin/python scripts/e002_evaluate_baselines.py --prepare-only --rebuild-windows
../.venv/bin/python -m unittest -v tests/test_validation.py
```

Run the complete baseline matrix with local GPU access:

```bash
../.venv-gpu/bin/python scripts/e002_evaluate_baselines.py \
  --models persistence sim_real_cno sim_real_fno_fp16 sim_real_transolver \
  --device cuda --batch-size 4 --force
```

The ignored detailed report is `artifacts/e002_validation/report.json`. It
contains raw errors, interval coverage, ID/OOD slices, hashes, checkpoint
metadata, timing, parameter counts, and memory measurements. Its verifier code
commit is `3ba439beba9ca8b9f35de30c1183747dd7519957`.

## Rules for future experiments

- Use the explicit 64-file training list; never infer training as “all files.”
- Fit normalization, masks, augmentations, early stopping, and hyperparameters
  from the training side only.
- Use all five official subscores and retain ID/OOD slices.
- Label results from any checkpoint that saw validation real fields as
  contaminated; do not compare them as clean model-selection evidence.
- Keep this split fixed for primary iteration. Add separately named folds to
  test robustness rather than tuning the E002 membership after seeing results.

## E003 clean learned baseline

E003 initialized the released simulation-only CNO and fine-tuned it in FP32 on
the 64 E002 training trajectories. Gaussian statistics used only training-window
field values. The process audited validation filenames/shapes/physical metadata
to enforce the frozen manifest, but it did not load validation `u`/`v`, masks,
statistics, or targets during fitting. The fixed-final checkpoint was evaluated
once after 600 updates; no validation-selected early stopping was used.

| Model/stratum | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Persistence overall | **93.771** | **66.667** | 93.819 | 98.636 | **16.989** |
| E003 CNO overall | 92.236 | 64.009 | **94.219** | 76.772 | 11.735 |
| E003 CNO ID interpolation | 94.822 | 73.167 | 94.939 | 76.772 | 14.595 |
| E003 CNO OOD edge | 88.777 | 54.300 | 93.209 | 76.772 | 7.630 |

The hypothesis that E003 would beat persistence on all four quality subscores
was rejected. It improved overall MVPE by 0.400 points, and its ID slice improved
Rel-L2, TKE, and MVPE relative to ID persistence, but default-interval SPS was
lower. Edge-regime degradation dominated the overall result: OOD Rel-L2/TKE/SPS
fell 4.352/12.367/8.555 points below OOD persistence.

Training used 2,627 stride-20 windows, batch four, Adam at `3e-4`, cosine decay,
and 600 updates (2,400 examples, 0.914 effective epoch). It took 2,211.39 seconds
on the RTX 5050 and peaked at 5,015,075,328 allocated GPU bytes. The first/final
25-update normalized losses were 0.3863/0.1083. The 7,960,467-parameter final
checkpoint is 31,943,371 bytes; its projected extracted package is about 32.39
MB. Torch 2.2.2 strict loading and a finite CPU training-window forward passed.

The ignored detailed artifacts are under `artifacts/e003_cno_finetune/`. The
checkpoint SHA-256 is
`4211fb234a17c8aa10ce4fe997635e7c036c9e28cffa1942a42913ea6211ecc3`,
and the training/evaluation code commit is `d33d365273a7cdad94d5bfd4366ca576430bee00`.
E003 is a negative baseline and is not a submission candidate.

## E004 saved-artifact postmortem

E004 reused only the saved E002 inputs/targets/window manifest, E003
predictions, and the initial/final checkpoint states. It performed no training,
model construction, model forward, or new validation inference. The analyzer
first reproduced the recorded E002 persistence and E003 overall raw metrics to
`1e-7` absolute tolerance.

| Nominal Re | Windows | Rel-L2 delta | TKE delta | MVPE delta | SPS delta | Target `u` mean z | Target `u` std/train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3750 | 84 | **-7.845** | **-20.231** | -0.455 | **-13.665** | -0.981 | 0.200 |
| 11400 | 105 | +0.488 | +8.144 | +0.504 | -4.454 | -0.239 | 0.581 |
| 20325 | 105 | +0.705 | +4.927 | +0.436 | -1.447 | +0.538 | 1.058 |
| 26700 | 63 | +0.690 | +3.471 | +1.335 | -1.725 | +0.954 | 1.385 |

Deltas are E003 minus persistence, so positive is better. The preregistered
claim that both edge groups fail was false: the degradation is confined to the
low-Re 3750 group, where E003 lost to persistence on relative L2 and TKE for
every one of 84 windows. Re 26700 improved all three deterministic quality
scores. Re 3750 contributes 57% of the OOD windows and therefore dominates the
two-group OOD average. The apparent angle-of-attack pattern is confounded by
the missing `3750_15` trajectory and is not treated as independent evidence.

The low-Re result is primarily a dynamics/scale failure, not an obvious mask or
mean-bias failure. Across the four Reynolds groups, target zero fractions stay
within 9.3–10.2% for `u` and 10.7–11.6% for `v`. At Re 3750, target `u` standard
deviation is only 0.200 times the training-global value, while scored-element
prediction bias is only +0.00267. Its raw TKE error is 2.307 versus persistence's
fixed 1.0. On the ID slice, E003 relative L2 becomes better than persistence
from forecast frame five onward, but its direct multi-step prediction is much
worse over the first four frames. Default-interval coverage is lower than
persistence at every Reynolds group, so SPS loses everywhere even when the
deterministic scores improve.

Checkpoint drift also rejected the preregistered conservative-adaptation
picture. Global trainable relative-L2 drift was 0.1087, above the 0.10 threshold;
99.63% of trainable elements changed. Non-BatchNorm trainable drift was 0.1241,
and residual blocks contributed about 74.6% of squared parameter-delta energy.
BatchNorm affine drift was only 0.0040. Running-mean shift was 0.216 initial
standard deviations RMS, below the 0.5 threshold, although running variances did
change (1.137 relative L2; median running-standard-deviation ratio 1.282). These
associations do not prove which state changes caused the validation behavior.

The preregistered E004 rule therefore says not to prioritize BatchNorm freezing.
The next falsifiable training experiment should change only the objective to a
scale-balanced loss over the scored `u/v` channels, testing whether it repairs
Re 3750 without sacrificing the other three held-out groups. E002 has now been
examined at aggregate, stratum, Reynolds, AoA, lead-time, and start-time levels;
this increased model-selection pressure is a limitation and must be considered
when interpreting later improvements.

The reproducible ignored report is `artifacts/e004_postmortem/report.json`.
Configuration SHA-256 is
`aa7b8cabc0c45d2bf874b9e776e2c813c0d6bae69fad121bb3a2a1ec718aa9cd`;
report SHA-256 is
`650fb32c7f789a752966c377afb4f747fd38765f479a174855d0c538efd31cf6`;
the pre-result analyzer commit is `8ff60a2`. The CPU run took 36.92 seconds and
peaked at 2,548,576 KiB RSS.

## E005 scale-balanced measured-channel loss

E005 held the E003 data, simulation-only initialization, architecture,
normalization, seed, shuffle order, optimizer, cosine schedule, batch size, 600
updates, and fixed-final evaluation constant. Its only experimental change was
the objective: mean per-window physical relative squared error over scored
`u/v`, with the real zero-pressure channel excluded from the loss.

| Model/stratum | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Persistence overall | 93.771 | 66.667 | 93.819 | **98.636** | **16.989** |
| E003 CNO overall | 92.236 | 64.009 | 94.219 | 76.772 | 11.735 |
| E005 CNO overall | **94.236** | **70.381** | **94.828** | 76.759 | 14.110 |
| E005 ID interpolation | 95.050 | 71.609 | 95.559 | 76.759 | 15.674 |
| E005 OOD edge | 93.098 | 68.698 | 93.803 | 76.759 | 11.866 |

E005 beat persistence overall on all three deterministic quality scores by
`+0.466/+3.714/+1.010` points and improved over E003 by
`+2.001/+6.372/+0.609`. It remains below persistence on local time and default-
interval SPS, and the unpublished final-score combination prevents declaring a
local overall winner.

| Nominal Re | Rel-L2 | TKE | MVPE | SPS | Rel-L2 delta vs E003 | TKE delta vs E003 |
|---:|---:|---:|---:|---:|---:|---:|
| 3750 | 93.098 | 66.971 | 93.267 | 13.393 | **+7.505** | **+20.536** |
| 11400 | 95.137 | 72.328 | 95.571 | 17.128 | +0.595 | -2.482 |
| 20325 | 94.963 | 70.904 | 95.547 | 14.213 | -0.140 | -0.690 |
| 26700 | 93.097 | 71.144 | 94.527 | 9.826 | -0.312 | +1.007 |

The primary low-Re recovery thresholds passed comfortably. The strict compound
hypothesis was nevertheless rejected because Re 11400 TKE lost 2.482 points to
E003, exceeding the preregistered 1.5-point non-regression allowance. Every
other threshold passed. This is a partial success, not permission to loosen the
criterion after seeing the result. E005 is the best clean learned checkpoint so
far, but it is not yet promoted to submission candidate: SPS remains 2.878
points below persistence, and E004/E005 have created substantial selection
pressure on the single E002 split.

Training took 2,224.13 seconds (37.07 minutes) on the RTX 5050, with
5,014,026,752 peak allocated GPU bytes. The first/final 25-update losses were
0.10787/0.01707. The 7,960,467-parameter checkpoint is 31,943,755 bytes, with a
projected 32.40 MB extracted package. It loaded strictly and produced a finite
CPU forward under Torch 2.2.2 before the single final validation run.

The ignored artifacts are under `artifacts/e005_scale_balanced_uv/`.
Configuration SHA-256 is
`3f5fe9644c3c62c03144821bb5faf5d664a1fbeb2a07db9908c52c965ec784d2`;
checkpoint SHA-256 is
`389f42f3509526b232b3064e168fd652c5668183f8739218cc8dc998d1c5edba`;
evaluation SHA-256 is
`659a3a1c327e96886337862e8a966b33a04a7a7ae6d988f1d1d77848e175df9e`;
the pre-result code commit is `1ad19e7`.

## E006 training-side uncertainty calibration

E006 froze the E005 checkpoint and point predictions. It selected a symmetric
interval of the form

```text
half_width = alpha * abs(point_prediction) + beta * sigma_global
sigma_global = 0.0563870259
```

from a pre-registered 45-candidate grid. Selection used 798 windows from 19
E002-training trajectories at nominal Re 5025/12675/19050/25425. The chosen
candidate was then audited on 1,829 windows from the other 45 training
trajectories without using audit results for selection. No E002-validation
field value was accessed until the interval was fixed.

The selected values were `alpha=0.025`, `beta=0.15`, corresponding to an
additive half-width floor of 0.008458 plus 2.5% of absolute prediction. The
training-side results were:

| Partition | Default SPS | Selected SPS | Gain | Default coverage | Selected coverage |
|---|---:|---:|---:|---:|---:|
| Calibration | 13.822 | 36.250 | **+22.428** | 0.277 | 0.763 |
| Audit | 15.052 | 36.994 | **+21.942** | 0.296 | 0.773 |

The fixed interval was then evaluated once against the saved E005 validation
predictions; no new model inference occurred.

| Model | Rel-L2 | TKE | MVPE | Time | SPS | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 93.771 | 66.667 | 93.819 | **98.636** | 16.989 | 0.348 |
| E005 default interval | 94.236 | 70.381 | 94.828 | 76.759 | 14.110 | 0.283 |
| E006 calibrated interval | **94.236** | **70.381** | **94.828** | 76.640 | **36.484** | **0.773** |

E006 improved SPS by 22.374 points over E005 and 19.495 over persistence. The
point-prediction SHA-256 remained exactly
`93b6d121ef452e1490d3c8bacb5b1db28d77ff9cfd3cb1daf7f798c0ae037abe`,
so Rel-L2/TKE/MVPE are unchanged. ID/OOD SPS was 36.868/35.932, and every
Reynolds slice exceeded 26.5 SPS. Bound construction took an estimated
0.000893 seconds per sample, reducing the local time score by 0.119; an
integrated submission-wrapper measurement is still required.

All preregistered E006 conditions passed, so the hypothesis is accepted and the
method is promoted to a submission candidate pending the E001-style container
gate. The main limitation is that E005 was trained on both calibration and
audit trajectories; separating interval selection makes the audit useful, but
does not turn its model residuals into out-of-sample errors. E002 selection
pressure and exact A800 timing also remain unresolved.

The GPU calibration took 182.27 seconds and peaked at 549,256,704 allocated
bytes. The one-shot saved-array evaluation took 8.56 CPU seconds and peaked at
1,836,932 KiB RSS. Ignored artifacts are under
`artifacts/e006_uncertainty/`. Configuration SHA-256 is
`67f5e71a316d4b5f1a89a7d92c1cd8124dbf5d5d3da228023102cd0ddb8a3ee0`;
calibration SHA-256 is
`230f4ae7403414bed55924fa9bdd66512790670c2bd554f90861ff716ec8bcac`;
evaluation SHA-256 is
`f50bd22dee3e489c18c7de706afa9293a80dbcf589a79b746961d8b335ce44cd`;
the pre-result code commit is `b8ad513`.

## E009 complementary fold (frozen, unused)

E009 freezes a second whole-regime split in `configs/e009_secondary_fold.json`
and does not train, select, or score against it. Membership is complementary to
both the E002 validation Reynolds groups and the E009/E006 calibration groups:

| Group | Nominal Re | Trajectories |
|---|---:|---:|
| low | 6300 | 5 |
| mid_low | 13950 | 5 |
| mid_high | 21600 | 5 |
| high | 24150 | 4 |
| **Total** | 4 groups | **19** |

These are unused-pool complementary groups, not global dataset edges. The
remaining 62 usable real trajectories are reserved as that fold’s training set.
E010 is the first experiment allowed to train on those 62 files and score the
19-file holdout. Do not inspect fold quality metrics before that single
preregistered evaluation.

## E010 complementary-fold robustness

E010 retrained the frozen E005 recipe on the 62 reserved complementary-fold
files and evaluated the fixed-final checkpoint once on the 19 unused
trajectories (376 non-overlapping windows). Persistence on this fold is the
only clean comparator; the E005/E008 checkpoint is contaminated here because it
trained on these holdout files.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Persistence | 93.450 | 66.667 | 93.439 | **98.947** | **16.195** |
| E010 CNO | **94.103** | **70.866** | **94.669** | 76.775 | 13.568 |

| Nominal Re | Rel-L2 | TKE | MVPE | Rel-L2 vs persistence | TKE vs persistence |
|---:|---:|---:|---:|---:|---:|
| 6300 | 93.793 | 71.923 | 93.800 | **+1.054** | **+5.256** |
| 13950 | 94.752 | 70.165 | 95.256 | +0.634 | +3.498 |
| 21600 | 94.110 | 70.667 | 94.886 | +0.497 | +4.000 |
| 24150 | 93.597 | 70.640 | 94.815 | +0.289 | +3.974 |

Every preregistered gate passed. Overall deterministic quality beat persistence
by `+0.653/+4.199/+1.231`, and Re 6300 improved rather than collapsing. Default-
interval SPS remains below persistence, as on E002. The E005 recipe is therefore
not an artifact of the E002 split. This does not promote the E010 checkpoint:
the next submission path is a separately packaged all-public-data run plus the
frozen `(0.025, 0.15)` interval.

Training took 2,197.74 seconds on the RTX 5050 and peaked at 5,014,026,752
allocated GPU bytes. The 7,960,467-parameter checkpoint is 31,943,755 bytes and
produced a finite CPU forward under Torch 2.2.2. Ignored artifacts are under
`artifacts/e010_secondary_fold/`. Configuration SHA-256 is
`90df5655d91543bb7827b0a966dc37690f01a9bf12245a7b110a06b8fb6341db`;
checkpoint SHA-256 is
`31c2947a6ba826c89a21cb7970065c43cac5cad2af005af73d77d9f7bf665293`;
evaluation SHA-256 is
`ecf34fa48c91766342a65c5051bae160e26a1e47f9cd3fe4a968b761fb8d2c40`;
the pre-result protocol commit is `b422a16`.

## E012 Fold C (frozen before scoring)

E012 freezes a third whole-regime split in `configs/e012_fold_c.json`. Membership
is complementary to Fold A (E002) and Fold B (E010) and is not used to select
`k`, intervals, or architectures:

| Group | Nominal Re | Trajectories |
|---|---:|---:|
| low | 5025 | 5 |
| mid_low | 16500 | 5 |
| mid_high | 22875 | 3 |
| high | 25425 | 4 |
| **Total** | 4 groups | **17** |

These are unused-pool complementary groups, not global dataset edges. The
remaining 64 usable real trajectories are reserved as that fold’s training set.
E012 scores persistence on Fold C and withholds learned methods there. Do not
rank E005/E008 on Folds B or C, or E010 on Folds A or C.

## E012 first comparison snapshot

Reproduce from `RealPDE_T1/`:

```bash
../.venv/bin/python -m unittest -v tests/test_e012_multifold.py
../.venv/bin/python scripts/e012_compare_folds.py
```

Ranking uses only **clean** rows. There is no local `final_score`.

| Fold | Method | Evidence | Rel-L2 | TKE | MVPE | Time | Default SPS | Calibrated SPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A (E002, 357) | Persistence | clean | 93.771 | 66.667 | 93.819 | **98.468** | 16.989 | — |
| A | E005 CNO | clean | 94.236 | 70.381 | 94.828 | 76.759 | 14.110 | 36.484 |
| A | E005 persist-first-4 | clean | **94.541** | **72.254** | **95.205** | 76.759 | 15.966 | **38.086** |
| B (complementary, 376) | Persistence | clean | 93.450 | 66.667 | 93.439 | **98.474** | 16.195 | — |
| B | E010 CNO | clean | 94.103 | 70.866 | 94.669 | 76.775 | 13.568 | 34.472 |
| B | E010 persist-first-4 | clean | **94.391** | **72.366** | **95.046** | 76.775 | 15.373 | **36.101** |
| C (reserved, 357) | Persistence | clean | 93.308 | 66.667 | 93.315 | **98.404** | 15.939 | — |

Persist-first-4 minus its CNO base:

| Fold | Rel-L2 | TKE | MVPE | Calibrated SPS |
|---|---:|---:|---:|---:|
| A | +0.305 | +1.873 | +0.377 | +1.602 |
| B | +0.288 | +1.500 | +0.377 | +1.629 |

E005/E008 are contaminated on B and C. E010 is contaminated on A and C. Those
rows were not ranked. Fold C learned scores were not computed.

The ignored report is `artifacts/e012_multifold/comparison.json` (SHA-256
`b789108dbf7d6af908e81436fafde5f40121199177d2291819f909328371639c`).
Protocol commit `3d34ff8`; configuration SHA-256
`7e5573f8affa69a605097a847861beff5badbef1736479f7da037ab25d41bd3f`.

## E013 persist-first-4 submission candidate

E013 packages the E008/E005 checkpoint with frozen `k=4` last-frame persistence
and recomputed `(0.025, 0.15)` bounds. It passed every E008-style local
submission gate and became the scored Codabench baseline at `75.570971` final
(`93.107` Rel-L2, `72.482` TKE, `92.721` MVPE, `86.132` time, `28.097` SPS),
`+0.727` versus E008. Clean local quality evidence remains the E012
persist-first-4 rows, not the two public contract windows.

| Gate | Result |
|---|---|
| Official-image digest / offline read-only container | passed |
| Archive layout and extracted size | 32,395,655 bytes; 223.6 MB headroom |
| Shape, dtype, finiteness, zero pressure, bounds | passed |
| Fresh-process exact repeat | exact |
| Batch-1 vs batch-2 invariance | exact `0.0` |
| Interval formula | exact on the hybrid prediction |
| 357-sample GPU proxy | 24.237 s predict; 160.3 MiB peak |

Do not modify or recompress the accepted archive.

- archive: `artifacts/e013_submission/e013_candidate.zip`
- archive SHA-256: `33c55772209f559b161728c3e59ef972601e29888f3d98e5fe0b8b92d1d8617a`
- report SHA-256: `95952044e017fb4f87bff31cd7fe2301097335e28c85bd1526d528a20f3db1fe`

Configuration SHA-256 is
`5f6e790eea46a33bfe7d62d5218cb7b46684f7b4e621d834b5ec04e660099eb6`.
The protocol commit is `888d4da`.

## E011 all-public training split

E011 trains on every usable public real trajectory and therefore has no
leakage-safe local holdout. The machine-readable membership is
`configs/e011_full_public_split.json`: 81 training files, zero validation
files, and the usual `7575_0.h5` exclusion. Do not score this checkpoint on
E002 or the complementary fold as generalization evidence. Public-example
windows remain contract tests only.

## E011 accepted submission candidate

E011 trained the frozen E005 recipe on the 81 usable public real trajectories
and packaged the fixed-final checkpoint with the frozen `(0.025, 0.15)`
interval and E008 singleton wrapper. The local hypothesis was the submission
gate, not a holdout quality comparison.

| Gate | Result |
|---|---|
| Official-image digest / offline read-only container | passed |
| Archive layout and extracted size | 32,395,477 bytes; 223.6 MB headroom |
| Shape, dtype, finiteness, zero pressure, bounds | passed |
| Fresh-process exact repeat | exact |
| Batch-1 vs batch-2 invariance | exact `0.0` |
| Interval formula | exact E006/E008 |
| 357-sample GPU proxy | 24.150 s predict; 160.3 MiB peak |

Training took 2,194.05 seconds (0.718 effective epoch) on the RTX 5050 and
peaked at 5,014,026,752 allocated GPU bytes. The 7,960,467-parameter
checkpoint is 31,944,139 bytes. Do not modify or recompress the accepted
archive; rebuild it through the verifier if its hash must change.

Ignored artifacts:

- archive: `artifacts/e011_submission/e011_candidate.zip`
- archive SHA-256: `33141e5ecd80b2009b165801f948d86ef46d7bd751921c6b34c10a0f39b70591`
- report SHA-256: `b30b243e074f29046535495b938331957d2e50f0ab8a24f6a4ec3cbd596e2d1f`
- checkpoint SHA-256: `bd0e497f9f8e5b72a3522d7571da892e07592adebd64d5f3eebaeb670bdffe5d`

Configuration SHA-256 is
`f4721f7594c6d4a830453ce0c6ab06eea75e17a416c440ac22cc29795fc69378`.
The protocol commit is `de9d8f2`; the wrapper commit is `ec4e73b`.

## E009 out-of-fold interval selection

E009 trained an E005-recipe CNO on the 45 E002-training trajectories outside
Re 5025/12675/19050/25425, then selected the frozen 45-candidate interval grid
on 798 windows from those 19 held-out training trajectories. The deployed point
model remained the frozen E005 checkpoint. The complementary fold above was not
used.

| Partition | Default SPS | Selected SPS | Gain | Coverage |
|---|---:|---:|---:|---:|
| OOF calibration | 11.786 | 34.874 | **+23.088** | 0.741 |
| Diagnostic in-sample | 15.301 | 37.208 | +21.907 | 0.774 |

The selected values were again `alpha=0.025`, `beta=0.15`. The E006 interval’s
OOF calibration SPS was 34.874, only 1.376 below E006’s in-sample 36.250, below
the preregistered 3.0-point optimism gate. Applying the unchanged interval to
the saved E005 validation predictions reproduced E006 exactly:

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E006 | 94.236 | 70.381 | 94.828 | 76.640 | 36.484 |
| E009 | 94.236 | 70.381 | 94.828 | 76.680 | 36.484 |

The strict hypothesis was rejected. In-sample residual selection did not choose
a different interval, so E008’s bounds stay frozen. The remaining local-to-
Codabench SPS/MVPE gap should be treated as point-prediction/distribution
mismatch, not as permission to retune `alpha`/`beta` on E002 or the leaderboard.

Training took 2,197.54 seconds on the RTX 5050 and peaked at 5,014,026,752
allocated GPU bytes. Calibration took 181.83 seconds and peaked at 549,256,704
allocated bytes. The 7,960,467-parameter checkpoint is 31,943,371 bytes and
produced a finite CPU forward under Torch 2.2.2. Ignored artifacts are under
`artifacts/e009_oof_uncertainty/`. Configuration SHA-256 is
`49cac15515ef478fd16a42d4e6485b30e99943bd40d0b25a7be239158d65ddfb`;
checkpoint SHA-256 is
`0eefc8819c98226554f78f4778ee8870a0839d145190c833974c1373ea68f5ac`;
calibration SHA-256 is
`1fff6c336087ea7d4ac7a0317d531e392a99d7e449a696310c9d72980b2000f5`;
evaluation SHA-256 is
`fcb2f29010aba04b9e579b235563d1ae4164a5e545ea5de1998bb18d0d3288fe`;
the pre-result protocol commit is `7bd94af`.

## E019–E024 residual-adapter and ensemble notes

These rows sit on top of the frozen E002/E012 splits. Rank only clean fold
rows. Do not invent a local `final_score`. Persist-first `k=4` and the E006
interval `(0.025, 0.15)` stay frozen.

| ID | Evidence status | What it is | Fold A persist-first-4 Rel-L2/TKE/MVPE/SPS | Notes |
|---|---|---|---|---|
| E019 | leakage-safe Fold A | Frozen E005 CNO + 10,835-param residual | 94.619 / 72.255 / 95.302 / 38.645 | Quality leader that was packaged as E021 |
| E020 | leakage-safe Fold B | Frozen E010 CNO + same residual recipe | 94.542 / 72.650 / 95.203 / 37.202 | Validates the E019 recipe, not the E022/E023 members |
| E021 | historical scored Codabench baseline | E019 packaged | hidden: 93.258 / 72.406 / 92.766 / 29.040 | final 75.872730; later superseded. |
| E022 | leakage-safe Fold A; **uncertainty hypothesis rejected** | Larger dual-head residual + mean/NLL | 94.601 / 72.247 / 95.291 / **38.843 frozen E006**, **37.153 learned** | Registered gate was learned SPS > 39.0 (`all_criteria_met: false`) |
| E023 | Fold A only | Average of E019, E022 field head, and a third h32 adapter | 94.644 / 72.248 / 95.320 / 38.891 | SPS is frozen E006, not ensemble variance. Member 3 was trained with Adam, not the registered AdamW. |
| E024 | container-ready, **withheld** | Packages E023 with singleton batching | same E023 Fold A numbers | Archive `bf954d5e…58e`. Hold; E025 validates the recipe, not this zip. |
| E025 | leakage-safe Fold B | E020 + Fold B heteroskedastic + Fold B h32 AdamW on frozen E010 | 94.574 / 72.618 / 95.248 / 37.437 | Beat E020 by +0.032 Rel-L2 / +0.045 MVPE. Do not package. |

Protocol commits for these files are the commits that first added them:
E019–E021 `8284dd6`, E022 `338c55c`, E023–E024 `972cb9f`.
