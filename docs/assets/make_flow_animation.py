#!/usr/bin/env python3
"""Generate docs/assets/sim-vs-sensor.svg: an original, illustrative animation.

A small D2Q9 lattice-Boltzmann simulation (BGK + Smagorinsky) of flow past a
NACA4418 airfoil. Both panels show vorticity in the wake just behind the wing,
on an 86 x 54 grid. The left panel shows the simulated field as-is. The right
panel shows the same simulated flow at a different moment, degraded the way
particle image velocimetry (PIV) data is: slightly blurred by finite
interrogation windows and speckled with measurement noise.

It is an illustration of the sim-to-real gap, not competition data, a
measurement, or a prediction from any model in this repository.

Requires numpy and Pillow (not part of the research environment):
    python3 docs/assets/make_flow_animation.py            # ~15-25 min on a CPU
The simulation is cached next to the output, so re-rendering is instant.
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

# Lattice and flow
NX, NY = 560, 210
CHORD, AOA_DEG = 96, 15.0
U0 = 0.08
TAU0 = 0.523  # nu = 0.0077, Re = U0 * CHORD / nu ~ 1000
CS = 0.10  # Smagorinsky constant, keeps BGK stable at this Re
WARMUP, PROBE, SAMPLE = 26000, 9000, 10  # steps; record every SAMPLE steps
# Recorded region (lattice cells) and the displayed wake behind the trailing edge
CROP_X, CROP_Y, CROP_W, CROP_H = 64, 18, 432, 176
VIEW_X = (142, 374)  # columns of the recorded region; the wing ends at column 139
VIEW_Y = (30, 176)  # rows framing the wake (same 86:54 aspect)
GRID_W, GRID_H = 86, 54  # coarse display grid, like the organizers' website animations
PIXEL = 5  # screen pixels per grid cell
PANEL_W, PANEL_H = GRID_W * PIXEL, GRID_H * PIXEL
NOISE = 0.12  # sensor-view noise, in units of the colour scale
FRAMES, DELAY_MS, GAP = 96, 80, 20

C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float32)
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]


def naca4418(x0: float, y0: float) -> np.ndarray:
    """Solid mask; image rows grow downward, positive AoA pitches the nose up."""
    m, p, t = 0.04, 0.4, 0.18
    y, x = np.mgrid[0:NY, 0:NX].astype(np.float64)
    a = np.radians(AOA_DEG)
    dx, dy = (x - x0) / CHORD, (y0 - y) / CHORD
    xc = dx * np.cos(a) - dy * np.sin(a)
    yc = dx * np.sin(a) + dy * np.cos(a)
    xs = np.clip(xc, 0, 1)
    half = 5 * t * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs**2
                    + 0.2843 * xs**3 - 0.1015 * xs**4)
    camber = np.where(xs < p, m / p**2 * (2 * p * xs - xs**2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xs - xs**2))
    return (xc >= 0) & (xc <= 1) & (np.abs(yc - camber) <= half)


def equilibrium(rho, ux, uy):
    cu = 3 * (C[:, 0, None, None] * ux + C[:, 1, None, None] * uy)
    return rho * W[:, None, None] * (1 + cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))


def vorticity(ux, uy):
    return 0.5 * ((np.roll(uy, -1, 1) - np.roll(uy, 1, 1)) - (np.roll(ux, -1, 0) - np.roll(ux, 1, 0)))


def simulate(cache: Path) -> None:
    solid = naca4418(NX * 0.2, NY * 0.5)
    rng = np.random.default_rng(4418)
    ux = np.full((NY, NX), U0, np.float32)
    uy = (1e-3 * rng.standard_normal((NY, NX))).astype(np.float32)
    f = equilibrium(np.ones((NY, NX), np.float32), ux, uy).astype(np.float32)
    inlet = equilibrium(np.ones((NY, 1), np.float32), np.full((NY, 1), U0, np.float32),
                        np.zeros((NY, 1), np.float32)).astype(np.float32)
    cxx = (C[:, 0] ** 2).astype(np.float32)[:, None, None]
    cyy = (C[:, 1] ** 2).astype(np.float32)[:, None, None]
    cxy = (C[:, 0] * C[:, 1]).astype(np.float32)[:, None, None]
    probe = (NY // 2, int(NX * 0.2 + 2.2 * CHORD))
    frames, signal = [], []
    total = WARMUP + PROBE
    for step in range(total):
        for i in range(1, 9):
            f[i] = np.roll(f[i], shift=(C[i, 1], C[i, 0]), axis=(0, 1))
        f[:, solid] = f[:, solid][OPP]
        rho = f.sum(0)
        ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
        uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
        ux[solid] = 0
        uy[solid] = 0
        feq = equilibrium(rho, ux, uy)
        neq = f - feq
        q = np.sqrt((neq * cxx).sum(0) ** 2 + (neq * cyy).sum(0) ** 2 + 2 * (neq * cxy).sum(0) ** 2)
        tau = 0.5 * (TAU0 + np.sqrt(TAU0**2 + 18 * CS**2 * q / rho))
        f -= neq / tau
        f[:, :, :1] = inlet
        f[:, :, -1] = f[:, :, -2]
        if step >= WARMUP:
            signal.append(float(uy[probe]))
            if (step - WARMUP) % SAMPLE == 0:
                v = vorticity(ux, uy)[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
                frames.append(v.astype(np.float16))
        if step % 2000 == 0:
            print(f"step {step}/{total}", flush=True)
    np.savez_compressed(cache, frames=np.stack(frames), signal=np.array(signal),
                        solid=solid[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W])


def shedding_period(signal: np.ndarray) -> float:
    s = signal - signal.mean()
    ups = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    return float(np.median(np.diff(ups)))


# Image rows increase downward: teal (counter-clockwise) - ink - amber (clockwise).
STOPS = np.array([
    (-1.00, (120, 232, 240)), (-0.55, (32, 150, 186)), (-0.18, (20, 58, 92)),
    (0.00, (11, 19, 32)),
    (0.18, (104, 44, 38)), (0.55, (226, 110, 64)), (1.00, (255, 214, 150)),
], dtype=object)


def colorize(s: np.ndarray, dead: float = 0.05) -> np.ndarray:
    xs = np.array([p for p, _ in STOPS], float)
    cols = np.array([c for _, c in STOPS], float)
    # Small dead zone so near-zero numerical ripples stay background-coloured.
    s = np.sign(s) * np.clip((np.abs(s) - dead) / (1 - dead), 0, 1)
    return np.stack([np.interp(s, xs, cols[:, k]) for k in range(3)], -1)


def png_data_uri(rgb: np.ndarray, colors: int = 256) -> str:
    im = Image.fromarray(rgb.astype(np.uint8)).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render(cache: Path, out: Path, offset: float) -> None:
    data = np.load(cache)
    (x0, x1), (y0, y1) = VIEW_X, VIEW_Y
    raw = data["frames"][:, y0:y1, x0:x1].astype(np.float32)
    if data["solid"][y0:y1, x0:x1].any():
        raise ValueError("The view should start behind the airfoil's trailing edge")
    period = shedding_period(data["signal"]) / SAMPLE  # in recorded frames
    span = 2 * period  # two shedding cycles -> seamless loop
    if span + offset * period + 1 >= len(raw):
        raise ValueError("Recording too short for two shedding periods; raise PROBE")
    scale = 3.2 * raw.std()
    rng = np.random.default_rng(7)

    def frame_at(pos: float) -> np.ndarray:
        i0 = int(np.floor(pos)) % len(raw)
        i1 = (i0 + 1) % len(raw)
        a = pos - np.floor(pos)
        return (1 - a) * raw[i0] + a * raw[i1]

    def to_grid(field: np.ndarray) -> np.ndarray:
        """Area-average the lattice field onto the coarse display grid."""
        return np.asarray(Image.fromarray(field, mode="F").resize((GRID_W, GRID_H), Image.BOX))

    sim_rgb, piv_rgb = [], []
    for k in range(FRAMES):
        pos = span * k / FRAMES
        sim_rgb.append(colorize(to_grid(frame_at(pos)) / scale))
        # Sensor view: another moment of the same flow, slightly blurred (finite
        # interrogation windows) with fresh per-pixel measurement noise each frame.
        w = to_grid(frame_at(pos + offset * period)) / scale
        w = 0.4 * w + 0.15 * (np.roll(w, 1, 0) + np.roll(w, -1, 0) + np.roll(w, 1, 1) + np.roll(w, -1, 1))
        w = 1.25 * w + NOISE * rng.standard_normal(w.shape)
        piv_rgb.append(colorize(w, dead=0.0))
    sim_sheet = np.concatenate(sim_rgb, 0)
    piv_sheet = np.concatenate(piv_rgb, 0)
    width = 2 * PANEL_W + GAP
    dur = FRAMES * DELAY_MS / 1000
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{PANEL_H}" viewBox="0 0 {width} {PANEL_H}" role="img" aria-labelledby="t d">
<title id="t">Clean simulation versus a sensor's view of the same flow</title>
<desc id="d">Original illustration generated by make_flow_animation.py: vorticity in the wake of a NACA4418 airfoil from a small lattice-Boltzmann simulation, on an {GRID_W} by {GRID_H} grid. Left, the clean simulated field. Right, the same simulated flow at a different moment, blurred and noisy like PIV data. Not competition data and not a model prediction. Motion stops when reduced motion is preferred.</desc>
<style>
.reel{{animation:play {dur:.2f}s steps({FRAMES}) infinite;image-rendering:optimizeSpeed;image-rendering:pixelated}}
@keyframes play{{to{{transform:translateY(-{FRAMES * PANEL_H}px)}}}}
@media (prefers-reduced-motion:reduce){{.reel{{animation:none}}}}
text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-size:11px;font-weight:700;letter-spacing:.16em;fill:#fff}}
</style>
<defs>
<clipPath id="a"><rect width="{PANEL_W}" height="{PANEL_H}" rx="10"/></clipPath>
<clipPath id="b"><rect x="{PANEL_W + GAP}" width="{PANEL_W}" height="{PANEL_H}" rx="10"/></clipPath>
</defs>
<g clip-path="url(#a)"><image class="reel" width="{PANEL_W}" height="{FRAMES * PANEL_H}" preserveAspectRatio="none" href="{png_data_uri(sim_sheet)}"/></g>
<g clip-path="url(#b)"><image class="reel" x="{PANEL_W + GAP}" width="{PANEL_W}" height="{FRAMES * PANEL_H}" preserveAspectRatio="none" href="{png_data_uri(piv_sheet)}"/></g>
<g><rect x="10" y="10" width="160" height="24" rx="12" fill="#0b1320" fill-opacity=".72"/><text x="22" y="26">CLEAN SIMULATION</text></g>
<g><rect x="{PANEL_W + GAP + 10}" y="10" width="112" height="24" rx="12" fill="#0b1320" fill-opacity=".72"/><text x="{PANEL_W + GAP + 22}" y="26">SENSOR VIEW</text></g>
</svg>
"""
    out.write_text(svg)
    print(f"{out.name}: {len(svg) / 1e6:.2f} MB, period {period * SAMPLE:.0f} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "sim-vs-sensor.svg")
    parser.add_argument("--cache", type=Path, default=HERE / ".flow_cache.npz")
    parser.add_argument("--offset", type=float, default=0.37,
                        help="phase offset of the sensor view, in shedding periods")
    args = parser.parse_args()
    if not args.cache.exists():
        simulate(args.cache)
    render(args.cache, args.out, args.offset)


if __name__ == "__main__":
    main()
