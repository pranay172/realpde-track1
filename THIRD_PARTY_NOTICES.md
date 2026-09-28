# Third-party attribution — Track 1

Reviewed 2026-09-28. This is a participant-maintained research snapshot, not an
official or endorsed organizer release.

- RealPDE organizers / RealPDEBench contributors: benchmark, datasets, baselines,
  task specification, scoring definitions and external starting kit.
- Zongyi Li and the FNO authors: Fourier neural operator implementation.
- CNO and Transolver authors: their respective architectures and external code.
- NVIDIA Corporation & affiliates: StyleGAN3-derived CNO filter utilities in the
  external dependency, not bundled here.
- Alex Rogozhnikov: einops, an external dependency with its own MIT notice.

Our participant-side adapters, calibration and experiment tooling use the
external organizer models. The upstream CNO/FNO/Transolver implementations are
not bundled or claimed as our original work.

The [preserved benchmark license notice](third_party_licenses/RealPDEBench-LICENSE.txt)
links to the full [CC BY-NC 4.0 legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
[Benchmark source](https://github.com/AI4Science-WestlakeU/RealPDEBench).

Numerical normalization/scoring constants and protocol descriptions retain their
competition provenance. No claim is made that the task, metrics, architectures,
dataset, or simulation-pretrained weights originated in this project.

The organizer kits, original page snapshots, example data and private Git
history are absent. When restoring external dependencies, preserve all original
copyright notices and obtain the exact kit under its terms; our license notice
does not grant access or redistribution rights. See
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md) and
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
