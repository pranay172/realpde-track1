# Method — Track 1

## Problem and evaluation discipline

Forecast 20 real PIV frames from 20 observed frames at 32×64 resolution, channels
[u,v,p]. Real pressure is represented as zero; velocity channels carry the scored
measurements. The competition supplies simulation-only pretrained baselines.
We split real trajectories by whole physical regimes rather than overlapping
windows, and exclude the duplicate/mislabeled `7575_0.h5`.

The backbone is an organizer-provided CNO. Our work concerns its training recipe,
lightweight residual adaptation, persistence blending, uncertainty calibration,
and evaluation/packaging discipline—not a newly invented neural operator.

## Best officially scored recipe: E037

1. **E035 backbone:** initialize the released simulation CNO; fine-tune on all
   81 valid public real trajectories for 3000 updates. Fit Gaussian normalization
   on training windows only. Use batch 4, Adam 3e-4, cosine decay and float32.
2. **Scale-balanced loss:** mean over windows of squared physical u/v error
   divided by that window's target u/v energy (with denominator floor). This
   avoids letting high-velocity examples dominate a global MSE.
3. **Single rollout adapter:** freeze the backbone and learn a small residual
   Conv3d adapter on concatenated inputs and backbone predictions. The selected
   E036 member-1 recipe uses 600 updates, transitioning from teacher forcing to
   a two-step rollout curriculum. E037 retains only this member.
4. **Persistence:** replace the first four forecast frames with the last input
   frame; preserve zero pressure.
5. **Intervals:** symmetric half-width `0.025*abs(prediction)+0.15*SIGMA_GLOBAL`,
   with `SIGMA_GLOBAL=0.0563870259`. E037 does not use E041's conditional rule.
6. **Inference:** evaluate samples independently, without borrowing evaluation
   windows or caching predictions across samples.

Controls: `configs/e035_all_data_long_backbone.json`,
`configs/e036_long_backbone_ensemble.json`, and `submission/e037/submission.py`.
Do not deploy the rejected multi-member E036 wrapper.

## Why these choices

E034 and E038 established clean whole-regime evidence for longer backbone
training on separate folds. The rollout adapter produced small gains on A;
its later B replication (E039B) failed the registered TKE-preservation gate.
Persistence improved the aggregate tradeoff without changing architecture.
Model/runtime/uncertainty improvements were evaluated separately; the hidden
final-score aggregation was not treated as a known linear formula.

The ensemble branch failed badly on the server. Local outlier probes exposed
a failure signature, but the precise causal attribution should not be reduced
to "averaging causes amplification": arithmetic averaging cannot exceed all
member magnitudes. Historical conclusions are retained with that qualification.

## Later uncertainty work, not the official best model

E041 selected channel/horizon width multipliers, optionally conditioned on a
per-window input temporal-variability statistic. Leave-one-Re-group-out fits
were done separately within A and B, never by naive cross-fold residual exchange.
The point model remained fixed. Development-fold reuse means this was exploratory
cross-fitting, not a pristine final audit.

E043 used the E038 backbone alone (62 training trajectories), persist-first-4,
and one conditional rule fitted on all 19 Fold B calibration trajectories.
It passed local checks but **failed official evaluation**, as reported by the
user on 2026-09-28. The server error log is unavailable; no cause is established.
E040 scaling was paused; E042's fresh three-way audit was registered but not run.

## Limits

All-public-data fits have no clean public holdout. Local/calibration scores are
not leaderboard estimates; previous local container checks did not guarantee
server success. The deleted weights and missing historical assembly step limit
artifact-level reconstruction. See [REPRODUCING.md](REPRODUCING.md).
