# External dependencies — Track 1

This public snapshot does **not** contain the organizer starting kit, copied
competition pages, example arrays, pretrained/trained weights, or datasets.
The experiment code still depends on those external resources. No download,
account action, or model execution happens automatically.

## Obtain the exact organizer kit

1. Open [Track 1 on Codabench](https://www.codabench.org/competitions/17363/)
   using your own authorized account. Historically, the Files tab required
   sign-in and approved participation. Obtain **starting kit v9**, not an older
   scorer and not a generic checkout of RealPDEBench.
2. Inspect the archive and its terms, then extract it locally to
   `realpde_t1_starting_kit_v9/` under this repository. This directory is
   deliberately ignored by Git. Do not commit it to the public repository.
3. Verify the required layout below before running model imports. Record the
   downloaded archive's SHA-256 locally. No independently verified original
   kit-archive checksum is available in this public snapshot.
4. If the exact historical kit is unavailable, ask the organizers for authorized
   access. Do not scrape protected downloads or silently substitute a newer kit.

Required external layout (not bundled):

```text
realpde_t1_starting_kit_v9/
  load_baseline.py
  scoring.py
  rpde_baselines/
  _vendor/einops/
```

Current kit download availability was **not verified** on 2026-09-28: the web
reader did not return usable competition-page content. The URLs above identify
the historical authorized source, not a guarantee of present download access.

The private archive also contained participant-added smoke helpers inside the
kit and a lazy SciPy import in CNO's `filtered_networks.py`. Those kit-local
changes are not redistributed here. Use the package/CUDA smoke commands in
`environment-locks/README.md`. The restored evaluator environment includes
SciPy, but exact organizer-container import parity still requires testing;
installing the upstream kit is not proof that old package checks will pass.

## Data and initial weights

Obtain [the competition data release](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data)
under its terms. Restore the historical sibling directory
`../RealPDE-Competition-Data/`, including `train_real/`,
`baseline_checkpoints/sim_pretrain/`, and any other files required by the
chosen experiment configuration. Exclude `train_real/7575_0.h5`.
Only restore simulation trajectories for experiments that actually use them.
Never obtain or redistribute hidden evaluation data.

The data and baseline checkpoints have their own terms; the root MIT license
does not cover them. Small example assets are dependencies too, not included
exceptions. Historical model/archive hashes identify deleted artifacts; they
are not download locations.

## Environment and runtime

Restore the shared environments using [environment-locks/README.md](environment-locks/README.md).
Then follow [REPRODUCING.md](REPRODUCING.md) for the selected recipe. Missing kit,
data, or weights will cause model/packaging tests to fail; do not count those
failures or skips as successful reproduction.

Packaging scripts copy organizer modules into local evaluation bundles.
Permission to obtain a kit or upload a competition submission is not a blanket
grant to publish those bundles on GitHub. Keep generated bundles private unless
their redistribution terms have been separately established.
