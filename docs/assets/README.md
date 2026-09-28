# README artwork

## `sim-vs-sensor.svg`

An original illustration of the sim-to-real gap, generated from scratch by
[`make_flow_animation.py`](make_flow_animation.py). No competition data, organizer
imagery or model output went into it.
It is documentation-only artwork, not training, calibration, or evaluation data
for the competition models.

**Left, "clean simulation".** Vorticity from a small lattice-Boltzmann
simulation (D2Q9, BGK collisions with a Smagorinsky term) of flow past a
NACA4418 airfoil, the same profile used in the competition. The wing sits at
15° angle of attack, at a Reynolds number of about 1000, on a 560 × 210 lattice.
The view starts just behind the trailing edge and is area-averaged onto a
coarse 86 × 54 grid, like the organizers' website animations. In the displayed
image coordinates, warm colours indicate clockwise rotation and cool colours
counter-clockwise. The loop spans about two estimated vortex-shedding cycles
for a visually smooth repeat.

**Right, "sensor view".** The same simulated flow at a different moment,
degraded the way particle image velocimetry (PIV) data tends to be: slightly
blurred by finite interrogation windows, and speckled with fresh measurement
noise every frame.

This is a cartoon of the gap, not a model of any real experiment. Real PIV
differs from simulation in more ways, including physics the simulation leaves out.
For the actual paired data, see the
[RealPDE competition website](https://realpdecompetition.github.io/#about).

## Format

The file is a single SVG. Each panel is a sprite sheet of 96 small PNG frames,
embedded as base64, stepped with a CSS animation and scaled up without
smoothing. There are no scripts or external requests. Animation stops when the
viewer prefers reduced motion.

## Rebuilding

```bash
pip install numpy pillow
python3 docs/assets/make_flow_animation.py
```

The simulation takes roughly 15–25 minutes on a laptop CPU. Its result is cached
in `docs/assets/.flow_cache.npz` (git-ignored), so tweaking colours or layout
afterwards re-renders in seconds.

The artwork and script are original contributions under the repository's
[MIT license](../../LICENSE).
