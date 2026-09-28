# RealPDE Track 1 reference

Last reviewed: **2026-08-16**, against local Codabench page snapshots in
`competition_docs/` plus the official site and starter kit v9.

This is a durable, competition-specific reference assembled from the official
competition site, the signed-in Track 1 Codabench pages and forum, the Hugging
Face release, starter kit v9, and direct inspection of the downloaded files.
The source archive held copies of the signed-in Codabench tabs; those copies
are intentionally omitted from this public snapshot. See
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md). When the upstream
RealPDEBench paper/repository conflicts with the competition, the current Track
1 Codabench rules and starter kit take precedence.

## Official sources

- Competition: <https://realpdecompetition.github.io/>
- Track 1 Codabench: <https://www.codabench.org/competitions/17363/>
- Track 1 forum: <https://www.codabench.org/forums/17061/>
- Competition data: <https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data>
- Upstream benchmark code: <https://github.com/AI4Science-WestlakeU/RealPDEBench>
- Upstream paper (background, not the competition specification):
  <https://arxiv.org/abs/2601.01829>
- Contact: `realpde-competition@googlegroups.com`

## Immediate checklist

- **Completed:** the user submitted the separate registration forms for both
  Track 1 and Track 2. Joining Codabench alone would not have counted.
- Use only one Codabench account per team. A team has at most three members.
- Work from starting kit **v9** and the restarted Main Development Phase, not
  results or assumptions from the archived phase.
- Exclude `train_real/7575_0.h5` from every split and training pipeline.
- Split local validation by whole trajectory/parameter regime, not overlapping
  windows, to avoid the same leakage that invalidated the first leaderboard.
- Keep every inference archive under **256 MB after extraction**, not merely as
  a compressed zip. Meet that budget by architecture first. Official Submission
  text **discourages fp16 packing** as a lossy last resort and points at
  safetensors for lossless compactness. The kit's complex-safe FNO packer
  remains available if a full-width FNO will not otherwise fit.
- Keep training fully reproducible from the released data and code. The top ten
  development teams are retrained from scratch by the organizers.

## Track 1 task

Track 1 is simulation-to-real forecasting of flow around an airfoil. A model
receives 20 observed real-flow frames and predicts the next 20 real-flow frames.
The official tensor contract is:

```text
input:  (N, T_in=20,  H=32, W=64, C=3)
output: (N, T_out=20, H=32, W=64, C=3)
channels: [u, v, p]
```

Simulation supplies `u`, `v`, and `p`. Real PIV measures only `u` and `v`; the
real-data `p` channel is represented as zero and is not scored. Raw fields are
`64 x 128`; competition evaluation spatially subsamples by two to `32 x 64`.

The physical parameters are angle of attack (nominally 0, 5, 10, 15, and 20
degrees) and Reynolds number. Evaluation is on hidden real PIV trajectories and
includes unseen parameter regimes. The hidden validation/test data are not in
the public release.

Each output sample must depend only on the input window it answers and on what
the model learned during training. Using other evaluation windows, batch-level
temporal adjacency, lookup/retrieval across the evaluation batch, or reading
hidden files from the container is a disqualifying integrity violation.

## Important 2026-08-05 restart

The original development evaluation windows overlapped: one window's target
appeared inside another window's input. This enabled cross-window copying rather
than forecasting. The organizers rebuilt the evaluation set, changed scoring,
released kit v9, and restarted the Main Development Phase with an empty
leaderboard on 2026-08-05.

Consequences:

- Scores from the archived phase are not comparable to current scores.
- `predict` may be invoked more than once, each time in a fresh isolated
  subprocess; state, caches, and files do not persist between calls.
- `metadata` is `{}` on scored calls. Do not index required keys from it.
- The current SPS maps its aggregate linearly to 0–100; the older logistic SPS
  is obsolete.
- Targets outside the PIV field of view or inside the airfoil are excluded from
  SPS.
- The current whole-run execution limit is five minutes, raised from the older
  three-minute limit.

## Public data: verified local inventory

Local source: `../RealPDE-Competition-Data/`

```text
train_real.tar.gz       6.9 GiB shown by du; 82 HDF5 members
train_sim.tar.gz        8.3 GiB shown by du; 100 HDF5 members
example_data/3750_0.h5  85 MiB
baseline_checkpoints/   eight checkpoints plus the fp16 packing helper
```

The Hugging Face page reports 17.7 GB total using decimal units, with 7.39 GB
for `train_real.tar.gz`, 8.83 GB for `train_sim.tar.gz`, and 1.37 GB for the
checkpoint tree. Full archive listings completed successfully during this
audit.

### Trajectory coverage

The simulated archive is a complete 20-by-5 nominal grid: 20 Reynolds-number
filenames crossed with five angles of attack, for 100 trajectories.

```text
Sim nominal Re:
3750, 5025, 6300, 7575, 8850, 10125, 11400, 12675, 13950, 15225,
16500, 17775, 19050, 20325, 21600, 22875, 24150, 25425, 26700, 27975
```

The real archive contains 82 files across 18 nominal Reynolds-number filenames
(all of the above except 15225 and 27975). Relative to those 18-by-5 possible
combinations, these eight files are absent:

```text
3750_15, 17775_0, 22875_5, 22875_20,
24150_5, 25425_5, 26700_5, 26700_20
```

After discarding the bad `7575_0` file, there are **81 usable real
trajectories** and 81 directly name-matched simulated trajectories. Do not
assume the nominal Reynolds number in a filename is the exact measured scalar:
for example, `train_real/5025_0.h5` stores `re=5028`, and `6300_0.h5` stores
`re=6306`.

### Critical bad file

`train_real/7575_0.h5` must not be used. Direct local comparison confirmed:

- `t`, `u`, `v`, `x`, `y`, and `aoa` are exactly equal to `6300_0.h5`;
- only the scalar `re` differs (`7585` versus `6306` in the files).

The organizers cannot remeasure this case at present, so the bad member remains
inside the published archive.

### Observed HDF5 schema

The current downloaded archives store datasets at the HDF5 top level. This
contradicts both release READMEs, which say training `u`/`v` live below a
`measured_data/` group. Write loaders against the observed files and preferably
support both layouts defensively.

Observed `train_real/5025_0.h5`:

| dataset | shape | dtype | notes |
|---|---:|---|---|
| `aoa` | scalar | int32 | value 0 |
| `re` | scalar | int32 | value 5028 |
| `t` | `(868,)` | float32 | 0.08 to 17.42 s; median step about 0.02 s |
| `u` | `(868,64,128)` | float64 | gzip-compressed |
| `v` | `(868,64,128)` | float64 | gzip-compressed |
| `x` | `(64,128)` | float64 | grid |
| `y` | `(64,128)` | float64 | grid |

Observed `train_sim/5025_0.h5`:

| dataset | shape | dtype | notes |
|---|---:|---|---|
| `aoa` | scalar | int32 | value 0 |
| `re` | scalar | int32 | value 5025 |
| `t` | `(1000,)` | float64 | 0.08 to 20.08 s; median step about 0.02002 s |
| `u` | `(1000,64,128)` | float32 | uncompressed |
| `v` | `(1000,64,128)` | float32 | uncompressed |
| `p` | `(1000,64,128)` | float32 | uncompressed |
| `x` | `(64,128)` | float64 | grid |
| `y` | `(64,128)` | float64 | grid |

The example file has the same top-level real schema and 868 frames. Thus the
website's “~600 time steps” description is only approximate and does not match
the downloaded trajectory lengths inspected here.

Real PIV contains persistent blank/zero regions caused by laser obstruction.
In the inspected real case, zero fractions were about 11.2% for `u` and 12.3%
for `v`. An organizer clarified that the mask is measurement noise, is mostly
zero, and stays almost unchanged within a trajectory. The organizer did not
explicitly answer whether the hidden validation/test masks have the same
distribution.

A single matched-case spot check also shows a major scale/domain gap: real
`5025_0` had mean/std `u` about 0.060/0.028, while simulated `5025_0` had about
0.874/0.405. Treat this as an example, not a dataset-wide statistic, but do not
mix real and simulated fields without deliberate normalization/domain handling.

### License and permitted use

The release is CC BY-NC 4.0 for non-commercial research and competition use.
Do not redistribute hidden validation or test data.

All training data must trace back to this competition release. Allowed:

- official checkpoints trained from the release;
- models trained from scratch on the release;
- standard, identifiable transformations/augmentations of released samples;
- a learned transformation model trained from scratch on the release and
  applied on the fly, provided the whole chain is reproducible.

Not allowed:

- external datasets;
- newly generated simulation trajectories or other synthetic training samples
  that do not trace back to released samples;
- any model or component pretrained on other data;
- outside weights at any point in a sequential pipeline.

Shortlisted teams provide training code, not pre-trained weights. Augmentation
and learned preprocessing code must ship and must be rerunnable from scratch.

## Baseline assets

Both `sim_pretrain/` and `sim_real_ft/` contain CNO, FNO, fp16-packed FNO, and
Transolver checkpoints. For Track 1, prefer the `sim_real_ft` versions because
they were pretrained on simulation and finetuned on real PIV.

| model | released size | construction recorded by kit v9 |
|---|---:|---|
| CNO | 32.2 MB | 3 layers, channel multiplier 32, 3 input/output channels |
| FNO fp32 | 403 MB | modes `(4,12,16)`, width 64, 4 layers |
| FNO fp16-packed | 201 MB | complex-safe packed form of FNO; officially discouraged |
| Transolver | 50.4 MB | 3 blocks, hidden 256, 8 heads, 16 slices |

Do not trust the upstream foil YAML blindly: its Transolver layer count is stale
(one in YAML versus three in the released checkpoints). FNO complex weights
cannot be safely packed with a naive `.half()`. Use the provided complex-aware
packer/loader.

The official FNO submission applies these real-train Gaussian statistics in
channel order `[u,v,p]`:

```text
mean_in  = [ 0.154960856, -0.000513992854, 0.0 ]
std_in   = [ 0.0968056545, 0.015960684,    1.0 ]
mean_tgt = [ 0.154962569, -0.000517793698, 0.0 ]
std_tgt  = [ 0.0968104079, 0.0159636438,   1.0 ]
```

Skipping this normalization is a known reason a correctly loaded baseline can
score poorly.

## Submission contract

Submit a zip whose root contains `submission.py`; optional weights and vendored
pure-Python packages sit beside it. No outer directory should wrap the files.

Minimal interface:

```python
def predict(input_array, metadata=None):
    # input_array: NumPy (N, 20, 32, 64, 3)
    # return: NumPy or Torch (N, 20, 32, 64, 3)
    ...
```

Alternatively, define `SubmissionModel.__init__` and
`SubmissionModel.predict`; if it implements `load_checkpoint(path, device)`,
ingestion calls it when a root-level `model.pth` exists.

Optional uncertainty output:

```python
return {
    "prediction": prediction,
    "lower": lower,
    "upper": upper,
}
```

Bounds must be all-or-nothing on every call, finite, ordered elementwise, and
the same shape as the prediction. If bounds are omitted, scoring uses
`prediction +/- 0.05*abs(prediction)`.

Other constraints:

- extracted archive size: strictly under 256 MB (design the model to fit;
  fp16 packing is officially discouraged; prefer safetensors if a lossless
  shrink is required);
- whole container execution: at most five minutes in Development;
- `torch.utils.data.DataLoader(num_workers=0)` only;
- deterministic, bounded inference is strongly preferred;
- no network access or install step; `requirements.txt` is ignored;
- use paths relative to the extracted submission directory;
- pressure may be returned as zeros because only `u` and `v` are measured.

### Evaluation image

```text
pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
Python 3.10, CUDA 12.1
```

Available includes Torch/Torchvision/Torchaudio 2.2.2, NumPy 1.26, Pillow,
PyYAML, requests, tqdm, sympy, and networkx. Not available includes SciPy,
pandas, matplotlib, h5py, einops, scikit-learn, and OpenCV. Pure-Python packages
may be vendored; compiled/native dependencies are unsafe to assume.

A forum hardware clarification states that one run gets one exclusive
NVIDIA A800-SXM4-80GB. Eight such GPUs back eight independent workers. CPU and
RAM are shared host resources; `/dev/shm` is Docker's 64 MB default. The same
forum post's old 180-second Track 1 limit was superseded by the current
five-minute rule.

Model construction/checkpoint loading is outside the per-sample inference timer
used for `time_score`, but remains inside the five-minute whole-run limit. An
untimed warm-up call pays one-time import/CUDA initialization costs and its
output is discarded.

## Evaluation: executable definition from `scoring.py`

The public leaderboard shows only `final_score`. A submission's detailed result
shows five 0–100 subscores. Their combination into `final_score` is intentionally
not published; do not assume a mean or infer a stable weighting from leaderboard
experiments.

### Relative L2

For each sample, flatten the measured channels and compute
`||prediction-target||_2 / ||target||_2`, then average samples. Map raw error
`e` to:

```text
rel_l2_score = 100 / (1 + 0.5*e)
```

### TKE

For each sample and spatial point:

```text
TKE = 0.5 * (mean_t((u-mean_t(u))^2) + mean_t((v-mean_t(v))^2))
```

Compute relative L2 between predicted and target TKE fields per sample, average,
then apply the same `100/(1+0.5*e)` mapping.

### MVPE

At evaluation resolution, the reference code uses four x probes
`[13,21,29,37]` and nine y positions `[8,10,12,14,16,18,20,22,24]`. It first
time-averages `u,v` at each x and over those y positions, computes relative L2
per sample/probe, averages the four probes, then averages samples. The score is
again `100/(1+0.5*e)`.

### Time

With reference numerical-solver time `0.72896 s` and mean timed neural inference
per sample `t`:

```text
time_score = 100 / (1 + sqrt(t / 0.72896))
```

A missing, non-finite, zero, or negative reported time scores zero.

### Safe Prediction Score (SPS)

Constants and structure:

```text
sigma_global = 0.0563870259
pm(e) = e / (0.5 + e)
element reward = (1-pm) * exp(-(upper-lower)/sigma_global), if target is inside
                 0, otherwise
branch weights = 0.5 Rel-L2 + 0.3 TKE + 0.2 MVPE
sps_score = 100 * weighted_branch_average
```

Targets equal to zero are omitted from the SPS element average as the proxy for
outside-field/airfoil-mask locations. The branch's accuracy term is a per-window
whole-field error, while interval coverage/tightness is elementwise. Tight
bounds help only when they cover the target; an uncovered element earns zero.

### Failure/zero-score hazards

- Prediction shape must exactly match `(N,20,32,64,3)` for the submitted call.
- Any NaN/Inf prediction produces zero on all five subscores.
- Non-finite or reversed supplied bounds produce zero on all five subscores.
- Mismatched sample counts or bound shapes can fail the run/validation path.
- Supplying bounds on only some `predict` calls discards all custom bounds and
  falls back to the default band everywhere.

## Schedule and ranking

All phase boundaries are UTC, not Anywhere-on-Earth.

| event | UTC date/time |
|---|---|
| Launch | 2026-07-05 00:00 |
| Warm-up ended | 2026-07-19 23:59:59 |
| Archived development phase | 2026-07-20 through 2026-08-04 |
| Restarted Main Development | 2026-08-05 00:00 through 2026-09-27 23:59:59 |
| Registration deadline | 2026-08-20 |
| Final Decision Phase | 2026-09-28 through 2026-10-25 |
| Final results | 2026-11-10 |
| Code/fact sheet deadline | 2026-11-25 |
| NeurIPS presentation | 2026-12-06 |

Warm-up quota was three submissions per day. Current Development quota is one
submission per day and 100 per phase. A platform/storage error while downloading
the submission bundle marks the run Failed and does **not** consume quota;
refresh and resubmit about an hour later. Voluntarily cancelled submissions may
behave differently, so do not cancel casually. The official Transolver baseline
is stated to finish in about one minute on Codabench.

The top ten Development teams are shortlisted. Organizers retrain each method
from scratch on their GPU cluster, evaluate private unseen parameter regimes,
audit integrity/rules, and use private-test `final_score` to decide the top
three. The exact finalist training-code package will be announced about five
days before Development ends.

Per track: prizes are USD 6,000 / 3,000 / 1,500 for the top three. Top three are
invited to give an oral presentation; top five are invited to the joint paper
and receive certificates. A team may additionally nominate one external advisor
for the paper/certificate. Winners must open-source the complete reproducible
solution before the presentation; the release license remains the team's choice.

## Rules and disclosure notes

- Participants with access to hidden evaluation data, scoring configuration, or
  the private test set are ineligible.
- Submissions may be stored, rerun, scored, inspected, and described by the
  organizers for the competition/results paper. Ownership is not transferred;
  non-winning code/weights are not released by organizers.
- Open-source software is allowed subject to its license. External development
  resources should be disclosed in the final fact sheet when requested.
- The forum contains an unanswered 2026-08-10 question asking whether OpenAI
  Codex use is allowed and what disclosure is required. Until organizers answer,
  keep a record of AI assistance and avoid assuming a disclosure policy.

## Documentation discrepancies and open questions

1. **Airfoil identity:** current competition pages call the geometry NACA4418.
   The upstream RealPDEBench paper/site describes a NACA0025 foil with different
   parameter ranges, resolution, and trajectory lengths. Treat the competition
   release as a distinct specification; ask the organizers if exact geometry is
   required for physics-derived features.
2. **HDF5 grouping:** competition/release READMEs say training `u` and `v` are
   under `measured_data/`; direct inspection of the downloaded tarballs found
   top-level `u` and `v`.
3. **Published counts:** the competition site says about 100 paired trajectories
   and about 600 steps. The downloaded release has 100 simulation files, 82 real
   files (81 usable), and the inspected files have 1000/868 frames.
4. **Hidden masks:** blank PIV regions are explained as laser obstruction and
   stable within a training trajectory, but whether hidden-set masks follow the
   same distribution was not answered explicitly.
5. **Codex policy:** the forum question about Codex use/disclosure currently has
   no organizer reply.
6. **FNO size path:** the kit still ships `pack_ckpt_fp16.py` because fp32 FNO
   is ~403 MB, but the 2026-08 Codabench Submission page now **discourages
   fp16 packing** and names safetensors as the preferred lossless shrink. A
   future FNO submission should try a smaller FNO or safetensors before fp16.

## Local environment audit

### Frozen local validation

E002 established the participant-side primary split in `configs/e002_split.json`:
64 complete real training trajectories, 17 complete validation trajectories,
and the bad `7575_0.h5` excluded. Validation holds out all real trajectories at
nominal Reynolds numbers 3750 and 26700 (edge/OOD) and 11400 and 20325
(interpolation/ID). Its 357 input/target samples use temporal stride 40, so raw
frames never overlap between validation samples. See `VALIDATION.md` for the
baseline table and exact interpretation rules.

The released `sim_real_ft` checkpoints were fine-tuned on the complete public
real release and therefore saw the E002 validation trajectories. Their local
scores are contaminated reproduction diagnostics, not generalization evidence.
Only models trained/fine-tuned without those 17 real trajectories may be called
leakage-safe on E002.

- A shared root environment now exists at `../.venv`, created with `uv` and
  CPython 3.10.20.
- Core versions are Torch 2.2.2+cu121, NumPy 1.26.4, h5py 3.16.0, SciPy 1.15.3,
  and einops 0.8.1. `uv pip check` reports all installed packages compatible.
- The environment occupies about 5.0 GiB, primarily because the CUDA 12.1
  runtime libraries matching the competition image are installed locally.
- All eight baseline checkpoints and example data are present locally.
- Track 1's full `smoke_test_kit.py` passed: every checkpoint loaded strictly;
  CNO, FNO, and Transolver produced finite outputs at 32x64 and 64x128; FNO
  fp16 packing round-tripped; bounds validated; and eval-image missing-package
  blocking passed for all three models.
- Track 2's template and agentic streaming evaluations passed end to end on its
  bundled example data. Its copies of all eight checkpoints also loaded
  strictly, and all three model types produced finite 32x64 outputs.
- The host GPU is an NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB, compute
  capability 12.0 / `sm_120`). The NVIDIA 595.84 driver is healthy and reports
  CUDA 13.2 support. The default Codex workspace sandbox hides `/dev/nvidia*`,
  so GPU commands must be run through an approved host-level command.
- The exact competition Torch build, 2.2.2+cu121, can detect the GPU when run
  at host level but cannot execute kernels on it: that wheel contains kernels
  only through `sm_90`, producing `no kernel image is available for execution
  on the device` on `sm_120`. Keep this environment for evaluator-compatible
  CPU validation; use a separate modern CUDA 12.8+ PyTorch environment for
  RTX 5050 development and training.
- The shared root GPU environment `../.venv-gpu` now provides CPython 3.10.20,
  Torch 2.7.1+cu128, NumPy 1.26.4, h5py 3.16.0, SciPy 1.15.3, and einops 0.8.1.
  Activate it with `source .venv-gpu/bin/activate` from the project root. It
  occupies about 6.5 GiB, and `uv pip check` reports no dependency conflicts.
- `realpde_t1_starting_kit_v9/gpu_smoke_test.py` is the reusable host-level GPU
  check. It verified that the wheel contains `sm_120`, completed a CUDA matrix
  multiplication, strictly loaded all eight checkpoints, and produced finite
  32x64 outputs for every checkpoint using the real example input. Peak allocated
  VRAM was 160.3 MiB for CNO, 532.1 MiB for FNO, and 496.4 MiB for Transolver.
- The full CPU-oriented `smoke_test_kit.py` also passes in `.venv-gpu`, including
  finite 32x64 and 64x128 forwards, FP16 checkpoint round-trip, bounds output,
  and evaluation-image missing-package simulation. This establishes development
  compatibility with Torch 2.7.1; final submission checks must still use `.venv`
  or the official Torch 2.2.2 evaluator image.
- Docker Engine 29.7.2 is running. NVIDIA Container Toolkit 1.19.0 is installed,
  and Docker has a configured `nvidia` runtime. Docker socket access is blocked
  only inside the default Codex sandbox; approved host-level Docker commands
  work. The official `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` evaluator
  image is now present locally. A GPU-enabled container still needs a
  Blackwell-compatible PyTorch image—the competition image does not solve the
  `sm_120` incompatibility.
- E001 then passed an end-to-end CPU acceptance run in that image. The image
  reports Python 3.10.14, Torch 2.2.2, CUDA runtime 12.1, and NumPy 1.26.4; SciPy,
  h5py, matplotlib, pandas, and einops are absent. The complete kit smoke loaded
  all eight checkpoints and exercised all model types. The official packed-FNO
  example extracted to 201,847,649 bytes (54,152,351 bytes below a conservative
  256,000,000-byte cap), returned finite float32 outputs with zero pressure, and
  was deterministic across fresh processes. Use
  `scripts/e001_verify_evaluator.py` as the reusable local acceptance gate.
