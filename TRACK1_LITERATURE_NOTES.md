# Track 1 literature notes

Last updated: 2026-08-23

These notes filter the Codex paper recs (`../Research_Papers/Codex_Recs_Conv1.md`,
`../Research_Papers/Codex_Recs_Conv2.md`) and the downloaded PDFs through the
RealPDE Track 1 protocol and the E000–E037 record in `LEDGER.md`. They are
method notes, not a submission recipe. External datasets and external pretrained
weights are forbidden; steal mechanisms, not checkpoints.

The paper-by-paper analysis remains historical research guidance. Current
experiment status, baselines, and submission decisions always come from
`LEDGER.md`; older detailed rows may be in `docs/ledger_archive/`.

Track 2 papers (AdaLED, Unsupervised Adaptation / NSLoRA, CL-PDE surrogates,
PIANO test-time adaptation) are listed only to mark them out of scope.

## How our own results change the ranking

Do not implement a paper in isolation. The following results already constrain
what is worth transplanting:

| Evidence | Constraint on the literature |
|---|---|
| E003 full CNO fine-tune collapsed at Re 3750; E005 recovered it with a scale-balanced scored-`u/v` loss; E034 later showed the 600-update ceiling was also binding | Unrestricted backbone rewrite is fragile, but adequate optimization budget matters once the objective is stable. Treat loss and budget as separate controlled variables. |
| E004: 10.9% parameter drift, BatchNorm-freeze hypothesis rejected | Freeze *more* of the sim backbone (Toma / Yang), not BatchNorm as the first knob. |
| E011 600-update all-public CNO lost on Codabench | More public data under the same budget is not the next quality move. |
| E037 long-budget CNO + single rollout adapter is the scored baseline (77.308) | Persist-first-4, rollout exposure, and longer backbone training have all transferred to the hidden set; evaluate new point models with and without the frozen overlay. |
| E014 zero-shot sim FNO collapsed; persist-first then destroyed TKE (67 → 10) | Persist-first is a patch for a *decent* predictor, not a collapsed one (D021). |
| E015 inference-time window-std matching hurt, worst at Re 3750 (u-ratio 0.15) | Do not overlay scale transfer at inference (D022). Train scale handling if you revisit it. |
| Hidden vs Fold A diagnostic (E037): Rel-L2 −1.62, TKE −1.01, MVPE −2.89, SPS −10.14 | MVPE / mean-wake and uncertainty calibration under regime shift remain the largest local-to-hidden gaps. |
| E036 passed ordinary local checks but failed catastrophically on hidden inputs; E037 restored the proven single-adapter path | Ensemble or calibration complexity must pass localized-outlier and fresh-archive stress checks. Local mean metrics alone do not establish tail robustness. |
| Official scores Rel-L2, TKE, MVPE, time, SPS; unpublished `final_score` | Never invent a local final score. Rank only clean E012 fold rows (D018). |
| 256 MB extract, 5-minute container, kit v9 / Torch 2.2.2 | No RK4 / diffusion / second network at inference. Discriminators and PINN residuals stay in training only. |
| No external pretrained weights | Oommen’s VGG / Med3D perceptual loss is illegal here. Spectral / vorticity / TKE losses are the legal substitute. |

The hidden-set pattern (MVPE and SPS lag most; TKE has a smaller gap) means “add
more turbulence energy” is not automatically the next leaderboard win. A
mean-field / wake-profile term and robust uncertainty calibration are more
aligned with the remaining hole than a GAN.

---

## Paper-by-paper (Track 1)

### 1. Toma, Ganapathisubramani & Symon — mixed-source PINN transfer (2026)

**PDF:** `../Research_Papers/Augmented-PINN.pdf` (arXiv:2601.04921)

**Physical problem is the closest match we have.** Stalled NACA 0012 at
α = 15°, Re 10k and 75k; RANS baseline → transfer onto LES or experimental
PIV; PIV has no pressure, noise, limited FOV, laser-reflection artifacts.

**What they actually do.** A 7×50 FCNN PINN of *time-averaged* RANS
(Spalart–Allmaras + corrective forcing + hard no-slip). Transfer replaces the
data term with subsampled PIV and uses ~1e-4 LR. Transfer improved RMSE by at
least 18%, up to 61% on indirectly inferred (driven) quantities.

No layers are frozen in their reported transfer runs. They simply initialize
from the RANS PINN, reduce the Adam stage to 10k iterations at `1e-4`, then use
L-BFGS to convergence. Thus Toma directly supports **good initialization + a
gentle adaptation budget**; the proposed 80–95% freeze below is our
Yang-inspired competition translation, not a result established by Toma.

Their PIV handling is also informative: they sample at 20 points/chord, remove
clear post-processing errors and near-wall points, train on 586 PIV locations,
and evaluate against the fuller field. The transferable lesson is robust
masking/down-weighting of known-invalid PIV regions—not indiscriminate fitting
of every pixel. We must derive any such mask only from released inputs/training
metadata, never from held-out targets.

**The result that matters for us.** Mismatched-Re transfer worked. They write
that *baseline quality supersedes matching the Reynolds number of the
experiment*. `P75−R10b` (PIV at 75k, initialized from RANS at 10k) was
competitive with the matched-Re transfer. That is exactly the hidden-regime
story.

**Second result.** The PINN sometimes *corrected* PIV artifacts (camera seams,
unphysical shear-layer gaps) instead of imitating them. The 2D NS residual
could not absorb true 3D PIV, and they say so. Do not force the network to
memorize every PIV defect; a sim-regularized residual is a feature.

**Do not copy.** Coordinate PINN, RANS SA, hard-constraint ALM, mean-field
reconstruction. We forecast 20 → 20 instantaneous windows, not a stationary
mean. A PINN residual of 2D incompressible NS on real PIV is the failure mode
they already documented.

**Steal.** Simulation representation first; adapt a small real-data component
at low LR; prefer a strong mismatched-Re sim backbone over a weak matched-Re
one; regularize toward the sim prior rather than fitting PIV noise.

**Our analogue.** Released sim-pretrained CNO (or a later sim operator) as
`F_θ`. Predict `Ŷ_real = F_θ(X) + C_φ(X, F_θ(X))` and train `C_φ` (plus at
most the last 1–2 CNO blocks) on `train_real`. Freeze 80–95% of `F_θ`. This
is I013.

---

### 2. Yang, Lee & Kang — MF-DeepONet + physics-guided subsampling (2025)

**PDF:** `../Research_Papers/DeepONet.pdf` (arXiv:2503.17941)

**What they actually do.** Phase 1 trains branch + trunk + MergeNet on
low-fidelity unsteady flow. Phase 2 transfers all three, *freezes branch and
trunk*, and adapts only MergeNet on high-fidelity points. Inference uses only
the HF net (no LF forward). Fine-tuning (MergeNet only) beat full-tuning and
linear probing, +43.7% vs single-fidelity, up to 76% vs conventional
coupled MF-DeepONet.

Crucially, their decoupled transfer **does not require corresponding LF/HF
pairs or identical query points**. That makes the parameter-separation result
more applicable to our unpaired CFD/PIV trajectories than the earlier summary
implied. It still assumes that the LF model captures physics useful to HF;
E018 shows our released sim CNO does not meet that condition strongly enough
for a 10k-parameter adapter.

**Physics-guided sampling.** After the LF model exists, score HF locations by
`s(x)=E_t |∂_t ω_LF|`, convert this to `p(x) ∝ s(x)^2`, and sample only 10% of
the selected locations from `p`; the remaining 90% stay uniformly sampled to
preserve coverage. Their 64×64-LF case reduced MSE by 20.73% at equal HF count,
and 60 HF samples matched/slightly beat the uniform 100-HF result. This is a
**mixed coverage/dynamics** rule, not permission to concentrate the whole loss
on the wake.

**Limitation they admit.** Fully freezing branch/trunk can be too strict;
sampling quality depends on a *good* LF model. They suggest selective unfreeze
later.

**Do not copy.** DeepONet, discarding 40% of `train_real` (we already have
it), or treating CFD and PIV as paired same-state multi-fidelity. Our sim and
real trajectories are not aligned snapshots of the same realization.

**Steal two things.**

1. Parameter separation: freeze the sim representation; adapt a small merge /
   residual head. Same family as Toma, more implementable on a CNO.
2. Reinterpret sampling as a *bounded mixed loss weight*, not a deletion:

   `w = (1-r) + r * normalize(clip(|∂_t ω|, q_lo, q_hi)^2)`, with small `r`

   That puts pressure on the shear layer, shedding, and wake — the generators
   of TKE and MVPE error while retaining broad spatial coverage. E017's
   everywhere-linear α=1 weighting was therefore not the paper's sampling
   rule; it tested only a rough analogue. This is I012.

**Small architecture note.** Temporal-only Fourier features improved their
DeepONet by 7.57%, whereas spatial or joint space-time encoding was consistently
harmful and sometimes divergent. If we add an explicit lead-time coordinate,
encode time only first; do not add high-frequency spatial PE to the CNO grid.

**Phase schedule they imply, which we should keep:** sim pretrain (already
have) → freeze + adapters → optional last-block unfreeze at tiny LR. E004’s
broad drift is the thing this schedule is designed to prevent.

---

### 3. Oommen et al. — learning turbulent flows with generative models (2025)

**PDF:** `../Research_Papers/Learning-Turbulent.pdf` (arXiv:2509.08752)

**The diagnosis is the paper.** L2-trained neural operators systematically
undershoot high-k energy because `E(k)` is a decaying power law and MSE is
dominated by large scales. CNO / TC-UNet already beat FNO on fine scales, but
*even they* oversmooth under Euclidean training. That is a training problem,
not an architecture problem.

**Their fix.** Keep a single fast operator at inference. Train it as
adv-NO: L1 + perceptual + RaGAN, `β ≈ 0.1`. Discriminator is training-only.
On Schlieren SR they cut energy-spectrum error 15×; on 3D HIT they forecast
~5 eddy-turnover times at 114× the speed of a diffusion baseline.

**Caveat they report.** Recovering high-k can *raise* pointwise field error
when fine-scale phase is wrong. Do not overweight the adversarial term.

The caveat is task-dependent, not universal. In their super-resolution table,
adv-NO improved spectrum NRMSE `0.1601 → 0.0109` but worsened field NRMSE
`0.0542 → 0.0662`; in their HIT forecast it improved both field
`0.0326 → 0.0265` and spectrum `0.0865 → 0.0235`. A divergence-penalized NO
did **not** cure spectral bias and made spectrum error worse (`0.1262`) because
the PDE residual was itself L2-trained. This is direct evidence against assuming
that an incompressibility penalty will recover RealPDE TKE.

**Illegal for us.** Perceptual loss on VGG-19 / Med3D. Those are external
pretrained weights.

**Legal and high-value.**

- Explicit TKE loss on the scored definition
  `0.5 (Var_t u + Var_t v)`.
- Log-spectrum loss
  `||log(E_pred(k)+ε) − log(E_true(k)+ε)||_1` on `u,v` (and maybe vorticity).
- Gradient / vorticity L1.
- Optional from-scratch PatchGAN on `[u, v, ω]`, never shipped. Start
  `λ_adv ~ 1e-3–1e-2`, not 1.0.

**Filter through E013.** Hidden TKE is already slightly *above* local TKE.
A pure “add energy” term can inflate TKE while wrecking Rel-L2 / MVPE phase.
Pair any spectral/TKE term with an explicit mean-wake / MVPE term
(GeoIncNO). This is I011, not a GAN-first experiment.

---

### 4. Zhang et al. — GeoIncNO (2026)

**PDF:** `../Research_Papers/GeoIncNO.pdf` (arXiv:2608.11237)

**What they actually do.** Long-horizon operator that (i) predicts a *latent
increment* `z_{t+1} = z_t + Δz_t`, (ii) shapes `Δz` with active-band low-rank
projectors in frequency–channel space, (iii) reconstructs with
**mean–fluctuation decoupling (MFDR)**: fuse `ū` and `u'` separately, and
apply phase correction *only* to the zero-mean fluctuation. Motivation is
exactly spectral drift, phase drift, and mean drift.

**Why this maps onto our scorecard.**

| Component | Official subscore |
|---|---|
| Temporal mean `ū` | MVPE / mean wake |
| Zero-mean `u'` | TKE |
| Phase of `u'` | Rel-L2 (and TKE if phase-scrambled energy still matches) |

Hidden MVPE is our largest quality hole. Treating mean and fluctuation as
one Euclidean field is how a model can look fine on Rel-L2 while missing the
wake profile.

**They also report RealPDEBench FSI** (sim training / real training / real
finetune). GeoIncNO (9.8M) beat CNO and Transolver there. That is a different
benchmark and a different protocol; do not treat those numbers as our
leaderboard. The *decomposition* is what transfers.

**The ablation sharpens the recommendation.** On synthetic 2D NS, adding MFDR
reduced temporal mean drift from `1.32e-3` to `1.05e-4`, the largest single
mean-drift improvement. Applying phase correction to the *full* field then
raised mean drift to `3.42e-4`; restricting it to the zero-mean fluctuation
reduced it to `6.21e-5`. Mean/fluctuation separation has direct evidence;
full-field phase manipulation is actively risky.

They also train on Re 1000 and 10000 and test the unseen intermediate Re 5000,
where GeoIncNO leads the reported baselines. This is unusually aligned with
Track 1 regime interpolation, but it remains a synthetic 2D NS experiment, not
Sim2Real PIV evidence. The PDF is a first-version preprint with placeholder
conference metadata, so treat all headline numbers as provisional.

**Do not copy first.** Full GeoIncNO (latent increments + ABP + phase
corrector) is a new architecture, a new checkpoint, and a runtime risk.
CNO + persist-first already has a 5-minute / 32 MB envelope we understand.

**Steal first, as losses not as a new net.** The paper did not ablate the loss
below on a fixed CNO; this is our inexpensive score-aligned approximation of
MFDR, not a reproduced GeoIncNO result.

```text
Ŷ = Ŷ_mean + Ŷ_fluct
L = L_field
  + λ_m ||mean_t Ŷ − mean_t Y||²          # MVPE
  + λ_f ||(Ŷ − mean Ŷ) − (Y − mean Y)||²  # fluctuations
  + λ_TKE |TKE(Ŷ) − TKE(Y)|
  + λ_spec ||log E_Ŷ − log E_Y||_1
```

If that helps Fold A *and* Fold B, *then* consider a two-head decoder. This
is the top of I011.

---

### 5. Reza & Faraji — RECAST (2026)

**PDF:** `../Research_Papers/RECAST.pdf` (arXiv:2608.11572)

**What they actually do.** Hybrid coarse solver + SHRED-delta corrector inside
the time loop + SHRED-SR reconstructor. Train the corrector first one-step,
then solver-in-the-loop so it sees its own corrected states. 50–92% error
drop vs uncorrected coarse 1D PDE solvers; held-out parameters still improved.

Their key separation is **evolution error vs representation error**. Offline
super-resolution of an already drifted coarse trajectory stayed near the
uncorrected solver's error; the correction had to act before its output was fed
back. The solver-in-loop curriculum was horizons 2 → 4 → 8 (400 updates each),
and horizons beyond 8 did not help. Checkpoint selection always used a fixed
100-step validation rollout rather than the current curriculum horizon.

**We do not have a coarse CFD solver at inference.** The transplant is the
*shape*, not SHRED or 8× coarsening:

```text
baseline(X) + corrector(X, baseline(X))
```

E013 is already a crude, non-learned version of this: copy last input frame
for k=4, then trust the CNO. RECAST says: *learn* the residual on top of a
frozen baseline, and expose the corrector to its own outputs during training.

**Important restriction from E014.** The baseline must already be in the
right ballpark. Correcting a collapsed FNO is how TKE died. Candidate
baselines, in order: last-frame persistence, E005/E013 CNO, not sim-only FNO.

**Steal.** Residual-on-frozen-predictor (I014), and a small amount of
model-rollout / persist-mixed history during training (PIANO / Temporal-NO).
Do not ship a second recurrent net.

**Boundary of the analogy.** RECAST has paired fine/coarse states and an actual
solver transition; our CFD and PIV trajectories are unpaired and E005 is a
direct 20-frame block predictor. For I014, first learn `Y - E005(X)` on clean
real pairs. Only if that works should we construct a second 20-frame block from
the model's own predicted history. Do not assume the paper's 1D solver-in-loop
gain transfers automatically.

---

### 6. Lai, Chen & Xu — DyMixOp (2025)

**PDF:** `../Research_Papers/DyMixOp.pdf` (arXiv:2508.13490)

**Claim.** Local-global mixing (LGM) layers, motivated by convection /
inertial-manifold theory, cut error up to 86.7% on convection-dominated PDEs
versus global-only operators.

The actual nonlinear LGM unit is a Hadamard product of a local convolutional
path and a global Fourier path, not a simple sum. Local-only and global-only
ablations were worse; nonlinear-only training became unstable around epoch 340,
so the authors retain both linear and nonlinear routes. Also, each table entry
is the best of roughly 16 model configurations. The 86.7% headline is therefore
a heavily tuned synthetic-benchmark result, not an expected RealPDE gain.

**Relevance.** Airfoil wake is convection-dominated. Pure Fourier / global
operators smear shear layers. CNO is already a local (wavelet/conv) operator,
which is why it is our backbone rather than FNO.

**Do not start here.** Replacing CNO with DyMixOp is a new architecture, a
new sim-pretrain, and a kit-v9 compatibility project. E014 already showed
that “maybe FNO is better” is not free. Revisit only if residual + split
losses saturate and the failure is visibly a missing local-global path.

---

### 7. Diab & Al Kobaisi — Temporal Neural Operator (2025)

**PDF:** `../Research_Papers/Temporal-NO.pdf`

**Steal the training protocol, not TNO.** They combine Markov assumption,
teacher forcing, and temporal bundling (`L` past → `K` future). Multi-step
bundled DeepONet beat one-step DeepONet on their weather-like task.

The paper explicitly warns that setting `K` to *all available future steps* can
prevent the operator from learning temporal dynamics. In its concrete weather
schedule it uses 60 teacher-forced epochs followed by 40 epochs conditioned on
the model's own outputs. Its demonstrated bundles are small (`K=3` or `4`), so
our proposed 5/10/20 mix is an extrapolation, not a reported ablation.

**Our mapping.** The official task is already bundled 20 → 20. Extra value is
*mixed horizons* drawn from the same trajectories: 5 → 5, 10 → 10, 20 → 20,
and histories that mix ground truth with the model’s own earlier frames.
That attacks the E013 finding that frames 1–4 lose to persistence — a model
trained only on clean 20-frame teacher forcing never sees a persist-like
near-term target.

**Keep persist-first frozen at k=4.** Do not retune k to harvest a mixed-
horizon gain (D020). Score the trained model with the frozen overlay.

This is I015.

---

### 8. Hou, Huang & Perdikaris — CFO (2025)

**PDF:** `../Research_Papers/CFO.pdf` (arXiv:2512.05297)

**What they actually do.** Fit temporal splines, treat spline velocity as the
flow-matching target, train a neural operator to predict `du/dt`, integrate
with an ODE solver at inference. Strong long-horizon numbers; 25% irregular
time samples beat full-data AR baselines (up to 87% relative-error drop).

**Do not ship RK4 first.** CFO can trade step size against function evaluations
and reports competitive cases at 50% of its AR baseline's NFE, so “always 4×”
is too crude. But its baseline is a one-step AR rollout, whereas our CNO emits
20 frames in one pass; multi-evaluation integration remains an unjustified
`time_score` risk under the five-minute cap.

**Steal as a training head.** Same 20 → 20 CNO, plus
`L_dot = ||∂̂_t Ŷ − D_t Y||²` on a finite-difference / spline target. No
extra eval at inference. This is **our approximation**, not a CFO ablation: CFO
trains on spline-interpolated states and analytic spline velocities, not merely
an auxiliary derivative head. Because PIV is noisy, raw first differences can
amplify measurement error; use a smoothed central/quintic target and a small
weight, or skip I016. This remains after I011–I013.

---

### 9. Mandl et al. — PITI-DeepONet (2025)

**PDF:** `../Research_Papers/PITI-DeepONet.pdf` (arXiv:2508.05190)

Same family as CFO: learn a temporal tangent and integrate. Also reconstructs
the current state and treats a large reconstruction residual as an OOD flag.

Their supervised tangent targets are evaluated from the *known PDE right-hand
side* at stored states, and their hybrid objective also uses a trustworthy PDE
residual. RealPDE provides neither for experimental PIV. That makes a PITI-style
`∂t` head materially weaker here unless we denoise finite differences first.

**Track 1 use.** Auxiliary `∂_t` head (with CFO) and, if we ever train a
reconstruction head, a cheap train-time regularizer. The OOD trigger is a
Track 2 idea.

**Do not** replace the 20 → 20 map with Euler/RK4 integration.

---

### 10. Nagda et al. — PIANO (2025)

**PDF:** `../Research_Papers/PIANO.pdf` (arXiv:2508.16235)

**Steal one sentence.** Train on states the model itself produces. Teacher-
forced 20 → 20 never sees persist-like or self-rolled histories.

PIANO itself is trained self-supervised with a known PDE/boundary residual
computed over its rollout. Our supervised noisy/persist/model-filled history
mix below is a competition translation, not PIANO's tested recipe.

A cheap Track 1 mix, after the loss terms exist:

- 70% clean history
- 20% noisy history
- 10% persist-filled or model-filled early frames

Do not meta-train a TTT loop. That is Track 2.

---

## Track 2 papers (do not implement here)

| Paper | Why it is not a Track 1 experiment |
|---|---|
| Song et al., Unsupervised Adaptation / NSLoRA | Test-time LoRA with a PDE residual. Track 1 has no `ttt_step` and no trustworthy 2D NS residual on PIV. |
| AdaLED | Online controller that falls back to a simulator we do not have. The *adapt-or-not* controller is Track 2. |
| Hemati et al., CL-PDE | Continual updates under regime drift. Track 1 is a single frozen `predict`. |
| PIANO / PITI TTA triggers | Need revealed `y_{t-1}`. |

If we later share a backbone with Track 2, keep adapters tiny and delay-
supervised; do not put PDE-residual TTT on PIV.

---

## Ranked Track 1 transplants

Effort is relative to the current E005 CNO + frozen E013 wrapper. Every
trained idea is scored on E012 Fold A (and Fold B before any submission),
with persist-first-4 *and* raw CNO rows. Rank only clean fold rows.

| Rank | Idea | Source | Effort | Why now | Known failure mode |
|---:|---|---|---|---|---|
| 1 | Split loss: field + mean + fluct + TKE + log-spectrum | GeoIncNO + Oommen | Low | Hidden hole is MVPE, not TKE; TKE term still needed so spectrum does not collapse while the mean is fitted | Over-weighted TKE/spectrum can raise Rel-L2 (Oommen’s phase caveat) |
| 2 | Wake / vorticity / `∂_t` loss weights | Yang MF-DeepONet | Low | Puts the existing E005 relative loss on the shear layer and wake | E017's everywhere-linear weighting was only an analogue; a paper-faithful mix must keep most coverage uniform and bound the dynamic term |
| 3 | Frozen sim CNO + residual adapter (then optional last-block unfreeze) | Toma + Yang | Medium | E003/E011 show full FT is the fragile path; Toma says source quality > Re match | E018 confirms the released sim CNO is too weak for the tiny adapter; do not unfreeze from that Fold A result |
| 4 | Learned residual on persist or frozen CNO, with mixed self-rollout histories | RECAST + PIANO + Temporal-NO | Medium | E013’s persist-first is the unlearned version of this | First prove a residual on frozen E005; only then add a short 2/4/8 rollout curriculum with fixed long-horizon selection |
| 5 | Mixed-horizon / teacher-forced curriculum | Temporal-NO | Low–medium | TNO warns that predicting every available future step can hide temporal dynamics | Its evidence is K=3/4, not our 5/10/20 proposal; fixed-shape CNO implementation and Fold-A overfit are risks |
| 6 | Auxiliary `∂_t` head, no ODE at inference | CFO / PITI | Medium | Encodes motion direction without RK4 | This head is our approximation; raw PIV differences amplify noise and PITI used known-RHS targets |
| 7 | From-scratch PatchGAN, training only | Oommen | Medium | Legal if no VGG; inference cost unchanged | Unstable; can trade Rel-L2 for fake energy; do after (1) |
| 8 | DyMixOp / new operator | DyMixOp | High | CNO already is the local operator; architecture last | New sim-pretrain, kit compatibility, time_score unknown |
| — | PINN / 2D NS residual at train or test | Toma (as architecture) | — | Rejected | 3D PIV ≠ 2D NS (Toma themselves) |
| — | RK4 / Neural-ODE / diffusion inference | CFO, Oommen diffusion | — | Rejected | time_score / 5-minute cap |
| — | Inference window-std match | (our E015, not a paper) | — | Already rejected (D022) | Re 3750 blow-up |
| — | Persist-first on a collapsed model | (our E014) | — | Already rejected (D021) | TKE collapse |
| — | Retune persist k or `(α,β)` on the leaderboard | — | — | Frozen (D014, D020) | One-point overfit |

---

## E016 result (I011, 2026-08-18)

Ran as planned: E005 controls, `λ_mean=λ_fluct=0.3`, `λ_tke=0.1`, no
spectrum, one Fold A look.

| Row | Rel-L2 | TKE | MVPE | cal-SPS |
|---|---:|---:|---:|---:|
| E013 persist-first-4 | 94.541 | 72.254 | 95.205 | 38.086 |
| E016 persist-first-4 | 92.009 | 60.788 | 92.537 | 29.365 |
| E016 raw | 91.368 | 72.385 | 91.275 | 28.105 |
| E005 raw | 94.236 | 70.381 | 94.828 | 36.484 |

Last-25 training parts were field/mean/fluct/tke `0.043/0.037/0.864/0.741`.
`L_fluct` and `L_TKE` use smaller denominators than `L_rel`, so they dominated.
Raw TKE beat E005, but Rel-L2/MVPE and Re 3750 collapsed; persist-first then
destroyed TKE at Re 3750 (D021 pattern). Hypothesis rejected (D024). Do not
retune these lambdas on Fold A.

## E017 result (I012, 2026-08-18)

Ran as planned: E005 controls, target vorticity-change weights, α=1, one Fold A
look. Last-25 unweighted loss 0.017 with mean weight 2.0 by construction.

| Row | Rel-L2 | TKE | MVPE | cal-SPS |
|---|---:|---:|---:|---:|
| E013 persist-first-4 | 94.541 | 72.254 | 95.205 | 38.086 |
| E017 persist-first-4 | 94.493 | 72.078 | 95.046 | 37.772 |
| E017 raw | 94.217 | 70.430 | 94.666 | 36.298 |
| E005 raw | 94.236 | 70.381 | 94.828 | 36.484 |

No collapse (Re 3750 and TKE gates passed), but Rel-L2/MVPE did not beat E013.
α=1 is an E005 twin, not a better overlay. Hypothesis rejected (D025). Do not
retune α on Fold A.

## E018 result (I013, 2026-08-18)

Ran as planned: frozen sim CNO, zero-init residual Conv3d adapter (10,835
trainable params), E005 relative loss, adapter-only, no last-block unfreeze.

| Row | Rel-L2 | TKE | MVPE | cal-SPS |
|---|---:|---:|---:|---:|
| E013 persist-first-4 | 94.541 | 72.254 | 95.205 | 38.086 |
| E018 persist-first-4 | 92.172 | 61.648 | 93.551 | 31.242 |
| E018 raw | 91.232 | 59.286 | 92.666 | 28.330 |
| E005 raw | 94.236 | 70.381 | 94.828 | 36.484 |

Toma/Yang freeze needs a useful source. Our sim CNO is too far from real PIV
for a 10k-param residual (same lesson as E014). Hypothesis rejected (D026).
Do not unfreeze last blocks on this Fold A look. Keep E013.

## Concrete next experiment (not pre-registered)

I011–I013 as specified are closed. A residual adapter is only promising on a
**decent** base (D021/D026). The cleanest next isolation is I014: freeze E005,
zero-initialize a small correction head, and train directly on `Y−E005(X)`
with clean real histories. Do **not** add rollout mixing until that simple row
beats frozen E005/E013. If it does, add a short 2→4→8 self-rollout curriculum
and select every stage on the same fixed long-horizon criterion, following
RECAST's discipline.

I015 is the alternative if we want to change temporal training rather than add
parameters: preserve the fixed 20-frame interface, vary the scored prefix or
construct carefully masked 5/10/20 examples, then use a teacher-forced phase
before model-history fine-tuning. TNO's actual evidence is only for small
bundles, so this needs an explicit implementation and leakage plan before it is
registered. Do not attach a larger adapter to sim CNO or retune either idea on
Fold A.

---

## Source files

| File | Role |
|---|---|
| `../Research_Papers/Codex_Recs_Conv1.md` | First-pass shortlist |
| `../Research_Papers/Codex_Recs_Conv2.md` | Deeper ranking after protocol constraints |
| `../Research_Papers/Augmented-PINN.pdf` | Toma Sim2Real transfer |
| `../Research_Papers/DeepONet.pdf` | Yang MF-DeepONet + sampling |
| `../Research_Papers/Learning-Turbulent.pdf` | Oommen spectral bias / adv-NO |
| `../Research_Papers/GeoIncNO.pdf` | Mean/fluctuation + increments |
| `../Research_Papers/RECAST.pdf` | Residual corrector |
| `../Research_Papers/DyMixOp.pdf` | Local-global mixing |
| `../Research_Papers/Temporal-NO.pdf` | Bundling / teacher forcing |
| `../Research_Papers/CFO.pdf` | Continuous dynamics, training-only use |
| `../Research_Papers/PITI-DeepONet.pdf` | Tangent operator, training-only use |
| `../Research_Papers/PIANO.pdf` | Self-rollout training |
| `LEDGER.md` | Binding experiment/decision record |
