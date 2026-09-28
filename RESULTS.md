# Results — Track 1

## Best recorded development-leaderboard result

User-supplied Codabench result for **E037**, not independently fetched server logs:

| Rel-L2 | TKE | MVPE | Time | SPS | final_score |
|---:|---:|---:|---:|---:|---:|
| 94.023891 | 73.827351 | 93.166385 | 85.988280 | 32.361125 | **77.307902** |

The prior E029 final_score was 75.878454. E037 improved it by 1.429448.
No final private-test ranking or score above 80 is claimed.

## Selected local evidence (not leaderboard scores)

| Experiment | Split / evidence | Rel-L2 | TKE | MVPE | SPS |
|---|---|---:|---:|---:|---:|
| E034 | A, clean backbone | 95.616 | 74.795 | 96.028 | 42.507 |
| E038 | B, clean backbone before calibration | 95.560 | 75.387 | 96.064 | 41.265 |
| E039A | A, single rollout adapter | 95.648 | 75.048 | 96.078 | 42.721 |
| E039B | B, single rollout adapter | 95.602 | 75.251 | 96.136 | 41.545 |

E039B failed its TKE delta threshold (-0.136 versus allowed -0.100). E041's
conditional-width cross-fit raised SPS to 44.271 on A and 42.258 on B, with fixed
points, but the development folds had already informed model choices.

## Failures and unfinished work

- E036: catastrophic official quality, final_score 7.957; quarantined along
  with the related E033 ensemble. Local verification had been insufficient.
- E043: **submitted and failed evaluation**, user-confirmed 2026-09-28.
  No error log, failure phase, or numeric result is available. Do not infer
  a timeout, format bug, or instability from this fact alone.
- E040: 6000-update scaling paused; update-400 recovery was later deleted.
- E042: disjoint calibration/audit protocol registered, never trained/audited.

Historical E043 archive SHA:
`651b7e74af1af1140acd392593c4b4b10e2c4ef3d36050518a86f6714aba5bd3`.
Best E037 archive SHA:
`0e799edde8660f1fd85e3f2eb17fad020788e18bfa3380465968cb1dcdc2236a`.
Both archives were deleted; hashes identify historical evidence, not downloads.

See [LEDGER.md](LEDGER.md), [VALIDATION.md](VALIDATION.md) and
[the E043 record](docs/e043_submission.md). The aggregate scoring formula was not
published in the competition kit; do not compute an unofficial "final score"
from these local component values.
