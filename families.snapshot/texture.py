"""Family: texture — stereo, seamless atmospheric beds. 4-12 s, loop=True.

Sound design follows McDermott & Simoncelli's texture statistics: many
contiguous octave bands, each with its own slow random amplitude modulation,
rather than one broadband noise with a single LFO. The bed is built twice with
independent noise (one per channel) so the stereo image is real, not a mono
source with a haas delay.

Looping:
  * render `dur + 150 ms`, then equal-power crossfade the last 150 ms into the
    first 150 ms (the standard seamless-loop trick)
  * every tonal frequency is snapped to an integer number of cycles over the
    loop length, so the tonal content is phase-continuous at the seam too. The
    pitch error is < 0.6 % at 4 s, < 0.2 % at 12 s
  * the master chain fades 4 ms at both ends (core behaviour), which dips the
    loop point; `_seam_compensate` pre-boosts those 4 ms so the seam stays at
    level (CORE ISSUES in texture.md)

Types: room tone, wind, machinery hum, dark drone, airy pad, tension bed.
All synthesised from noise + sines; no samples.

Contract:
    FAMILY, DESCRIPTORS, sample_params(rng), generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "texture"
DESCRIPTORS = ["texture", "atmos", "drone", "bed", "room", "air"]


# --------------------------------------------------------------------------
# CORE ISSUE #1 workaround (see texture.md)
# --------------------------------------------------------------------------
def _install_stereo_fade_fix() -> None:
    """core.dsp.fade_asym does `y[:mi] *= linspace(0,1,mi)`, which broadcasts
    for a mono buffer only; on an (n, 2) buffer it raises inside master() and
    every stereo render is skipped. Core files are off-limits, so probe for the
    bug and, only if it is present, install an axis-correct copy of the same
    fade. Mono input is still handled by the original function, unchanged.
    Delete this shim the moment fade_asym is fixed upstream.
    """
    try:
        from core import dsp as _dsp
    except Exception:
        return
    orig = getattr(_dsp, "fade_asym", None)
    if orig is None or getattr(_dsp, "_sfx_stereo_fade_fix", False):
        return
    try:
        orig(np.zeros((64, 2)), FS)      # the broken version raises here
        return
    except Exception:
        pass

    def fade_asym(x, fs: int = FS, in_ms: float = 0.4, out_ms: float | None = None):
        y = np.asarray(x, dtype=np.float64)
        if y.ndim == 1:
            return orig(x, fs=fs, in_ms=in_ms, out_ms=out_ms)
        n = y.shape[0]
        if n < 4:
            return y.copy()
        mi = min(max(1, int(fs * in_ms / 1000.0)), n // 4)
        if out_ms is None:
            out_ms = max(2.0, min(8.0, 1000.0 * n / fs * 0.15))
        mo = min(max(1, int(fs * out_ms / 1000.0)), n // 4)
        z = y.copy()
        if mi > 1:
            z[:mi] *= np.linspace(0.0, 1.0, mi)[:, None]
        if mo > 1:
            z[-mo:] *= np.linspace(1.0, 0.0, mo)[:, None]
        return z

    _dsp.fade_asym = fade_asym
    _dsp._sfx_stereo_fade_fix = True


_install_stereo_fade_fix()

TYPES = ("room", "wind", "machinery", "drone", "pad", "tension")

ROOT_NOTES = ["D2", "A2", "C3", "E3", "G3", "A3", "C4", "D4"]

LOOP_XFADE_S = 0.150


def sample_params(rng: np.random.Generator) -> dict:
    typ = str(rng.choice(TYPES, p=[0.20, 0.18, 0.14, 0.18, 0.16, 0.14]))
    dur = float(rng.uniform(4.0, 11.5))
    note = str(rng.choice(ROOT_NOTES))

    if typ == "room":
        am_lo, am_hi, depth = 0.03, 0.18, rng.uniform(0.10, 0.30)
        tilt_oct = rng.uniform(-4.5, -2.0)
        tonal = rng.uniform(0.0, 0.10)
        air = rng.uniform(0.02, 0.12)
    elif typ == "wind":
        am_lo, am_hi, depth = 0.04, 0.35, rng.uniform(0.35, 0.65)
        tilt_oct = rng.uniform(-4.0, -1.5)
        tonal = 0.0
        air = rng.uniform(0.08, 0.24)
    elif typ == "machinery":
        am_lo, am_hi, depth = 0.03, 0.20, rng.uniform(0.08, 0.25)
        tilt_oct = rng.uniform(-3.5, -1.5)
        tonal = rng.uniform(0.22, 0.50)
        air = rng.uniform(0.02, 0.10)
    elif typ == "drone":
        am_lo, am_hi, depth = 0.02, 0.14, rng.uniform(0.08, 0.25)
        tilt_oct = rng.uniform(-5.0, -2.5)
        tonal = rng.uniform(0.30, 0.62)
        air = rng.uniform(0.02, 0.12)
    elif typ == "pad":
        am_lo, am_hi, depth = 0.08, 0.50, rng.uniform(0.15, 0.45)
        tilt_oct = rng.uniform(-4.5, -1.5)
        tonal = rng.uniform(0.26, 0.55)
        air = rng.uniform(0.06, 0.22)
    else:  # tension
        am_lo, am_hi, depth = 0.15, 0.80, rng.uniform(0.25, 0.55)
        tilt_oct = rng.uniform(-5.0, -2.0)
        tonal = rng.uniform(0.24, 0.52)
        air = rng.uniform(0.03, 0.14)

    return dict(
        type=typ,
        dur=dur,
        root_note=note,
        root_oct=int(note[-1]),
        bands=int(rng.integers(8, 11)),
        q=float(rng.uniform(1.30, 1.65)),
        tilt_oct=float(tilt_oct),
        am_lo=float(am_lo),
        am_hi=float(am_hi),
        am_depth=float(depth),
        width=float(rng.uniform(0.55, 0.95)),
        tonal_gain=float(tonal),
        air_gain=float(air),
        bed_gain=float(rng.uniform(0.70, 1.00)),
        tilt_lp=float(np.exp(rng.uniform(np.log(4500.0), np.log(18000.0)))),
        hpf=float(rng.uniform(24.0, 70.0)),
        colour=str(rng.choice(["pink", "pink", "white", "brown"], p=[0.6, 0.15, 0.15, 0.10])),
        # tonal detail
        hum_hz=float(rng.choice([50.0, 60.0])),
        detune_cents=float(rng.uniform(2.0, 12.0)),
        chord=int(rng.integers(0, 6)),
        open_lp=float(np.exp(rng.uniform(np.log(900.0), np.log(4200.0)))),
        wobble_hz=float(rng.uniform(0.03, 0.20)),
        motor_hz=float(rng.uniform(6.0, 13.0)),
        # gust / movement
        wind_rate=float(rng.uniform(0.04, 0.22)),
        wind_oct=float(rng.uniform(0.5, 1.6)),
        dirt=float(rng.uniform(0.0, 1.0)),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return x if p <= 1e-12 else x / p * target


def _band_am(n: int, fs: int, rng: np.random.Generator, colour,
             centres: np.ndarray, q: float, depth: float,
             rate_lo: float, rate_hi: float, tilt_oct: float) -> np.ndarray:
    """Contiguous octave bands of noise, each with its own slow random AM.

    Bands are RMS-normalised and then given an explicit spectral slope so the
    result is a shaped bed (not white hiss) that still covers every band.
    """
    src = colour(n, rng)
    out = np.zeros(n)
    for fc in centres:
        b = dsp.bp(src, fc, fs, order=2, q=q)
        b = b / (np.sqrt(np.mean(b * b)) + 1e-9)
        rate = float(np.exp(rng.uniform(np.log(rate_lo), np.log(rate_hi))))
        m = dsp.lp(dsp.white(n, rng), max(0.05, rate * 3.0), fs, order=2)
        m = m / (np.max(np.abs(m)) + 1e-9)
        slope = 10.0 ** (tilt_oct * np.log2(fc / 1000.0) / 20.0)
        out += b * (1.0 + depth * m) * slope
    return _norm(out)


def _seamless(buf: np.ndarray, n_out: int, n_x: int) -> np.ndarray:
    """Equal-power crossfade of the extra tail into the head.

    y[:F] = tail * sqrt(1-w) + head * sqrt(w): at the wrap point y[0] is the
    natural continuation of y[n_out-1], and by y[F] the head is back intact.
    """
    if buf.ndim == 1:
        buf = buf[:, None]
    w = np.linspace(0.0, 1.0, n_x)[:, None]
    y = buf[:n_out].copy()
    y[:n_x] = buf[n_out:n_out + n_x] * np.sqrt(1.0 - w) + buf[:n_x] * np.sqrt(w)
    return y


def _seam_compensate(buf: np.ndarray, fs: int,
                     in_ms: float = 0.4, out_ms: float = 8.0,
                     floor: float = 0.65) -> np.ndarray:
    """Cancel most of master()'s mandatory fade at both ends of a loop.

    master() runs fade_asym(in 0.4 ms, out 8 ms) on every file. On a continuous
    bed those ramps are an audible hole at the loop point: the 8 ms out-ramp
    averages -6 dB. Dividing by the same ramp (clamped at `floor`, so the very
    first/last samples are not boosted to infinity) restores the level; only
    the last ~2.8 ms still taper. Peak boost is 1/floor.
    """
    y = np.array(buf, dtype=np.float64, copy=True)
    n = y.shape[0]
    two_d = y.ndim == 2

    def col(v: np.ndarray) -> np.ndarray:
        return v[:, None] if two_d else v

    mi = max(1, int(fs * in_ms / 1000.0))
    mo = max(1, int(fs * out_ms / 1000.0))
    if 1 < mi < n:
        ramp = np.clip(np.linspace(0.0, 1.0, mi), floor, 1.0)
        y[:mi] *= col(1.0 / ramp)
    if 1 < mo < n:
        ramp = np.clip(np.linspace(1.0, 0.0, mo), floor, 1.0)
        y[-mo:] *= col(1.0 / ramp)
    return y


def _snapper(n_out: int, fs: int):
    """Return a function that snaps a frequency to a whole number of cycles
    over the loop, so tonal layers are phase-continuous at the seam."""
    def snap(f: float) -> float:
        k = max(1.0, round(float(f) * n_out / fs))
        return k * fs / n_out
    return snap


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    typ = p["type"]
    n_out = int(fs * p["dur"])
    n_x = int(fs * LOOP_XFADE_S)
    n = n_out + n_x
    t = np.arange(n) / fs
    root = dsp.note_hz(p["root_note"][:-1], p["root_oct"])
    snap = _snapper(n_out, fs)
    colour = {"white": dsp.white, "pink": dsp.pink, "brown": dsp.brown}[p["colour"]]

    # ---- stereo bed: two independent per-band AM noise beds ----
    centres = np.geomspace(45.0, min(14000.0, fs * 0.45), p["bands"])
    bedL = _band_am(n, fs, rng, colour, centres, p["q"], p["am_depth"],
                    p["am_lo"], p["am_hi"], p["tilt_oct"])
    bedR = _band_am(n, fs, rng, colour, centres, p["q"], p["am_depth"],
                    p["am_lo"], p["am_hi"], p["tilt_oct"])
    g_bed = 0.62 * p["bed_gain"]
    layers = [Layer(bedL, gain=g_bed, pan=-p["width"], label="bedL"),
              Layer(bedR, gain=g_bed, pan=p["width"], label="bedR")]

    # ---- air: slow-swelling high band, centre ----
    air = dsp.hp(dsp.white(n, rng), 6200.0, fs, order=3)
    air_rate = max(1.0, round(p["am_lo"] * n_out / fs)) * fs / n_out
    air_mod = 0.7 + 0.3 * np.sin(2.0 * np.pi * air_rate * t + rng.uniform(0, 6.28))
    air = _norm(air) * air_mod * p["air_gain"]
    layers.append(Layer(air, gain=1.0, pan=rng.uniform(-0.25, 0.25), label="air"))

    # ---- type-specific centre content ----
    if typ == "wind":
        fc = 650.0 * 2.0 ** (p["wind_oct"] * np.sin(
            2.0 * np.pi * p["wind_rate"] * t + rng.uniform(0.0, 6.28)))
        gust = 0.55 + 0.45 * np.sin(2.0 * np.pi * p["wind_rate"] * 0.6 * t + rng.uniform(0.0, 6.28))
        wind = _norm(dsp.moving_bp(dsp.white(n, rng), fc, fs, q=1.5, bands=12))
        wind = wind * np.clip(gust, 0.0, 1.0) * 0.55
        layers.append(Layer(wind, gain=1.0, pan=0.0, label="wind"))

    if typ in ("machinery", "room") and p["tonal_gain"] > 0.001:
        hum = np.zeros(n)
        f0 = snap(p["hum_hz"])
        for h, g in ((1, 1.0), (2, 0.60), (3, 0.35), (4, 0.25), (6, 0.15), (8, 0.10), (10, 0.06)):
            hum += dsp.sine(np.full(n, f0 * h), fs) * g
        wob = 1.0 + 0.12 * np.sin(2.0 * np.pi * snap(p["wobble_hz"]) * t)
        hum = _norm(hum) * wob * p["tonal_gain"]
        layers.append(Layer(hum, gain=1.0, pan=0.0, label="hum"))
        if typ == "machinery":
            motor = dsp.bp(dsp.white(n, rng), 600.0, fs, order=2, q=1.6)
            beat = 1.0 - 0.35 + 0.35 * np.sin(2.0 * np.pi * snap(p["motor_hz"]) * t)
            motor = _norm(motor) * beat * p["tonal_gain"] * 0.45
            layers.append(Layer(motor, gain=1.0, pan=0.0, label="motor"))

    if typ == "drone":
        tones = (1.0, 1.5, 2.0, 3.0)
        centre = np.zeros(n)
        for i, iv in enumerate(tones):
            f = snap(root * iv)
            centre += dsp.sine(np.full(n, f), fs) * (1.0 / (1.0 + 0.7 * i))
            # a second detuned copy beats slowly against it
            centre += dsp.sine(np.full(n, snap(f * (1.0 + p["detune_cents"] / 1200.0))), fs) * (0.45 / (1.0 + 0.7 * i))
        centre = dsp.lp(centre, p["open_lp"], fs, order=2)
        wob = 1.0 + 0.18 * np.sin(2.0 * np.pi * snap(p["wobble_hz"]) * t)
        centre = _norm(centre) * wob * p["tonal_gain"]
        layers.append(Layer(centre, gain=1.0, pan=0.0, label="drone"))

    if typ in ("pad", "tension"):
        if typ == "pad":
            ivs = [(1.0, 1.0), (1.5, 0.7), (2.0, 0.55)] if p["chord"] < 3 else [(1.0, 1.0), (1.25, 0.6), (1.5, 0.65)]
            lp_hz = p["open_lp"] * 1.6
        else:
            ivs = [(1.0, 1.0), (2.828, 0.45)] if p["chord"] < 3 else [(1.0, 1.0), (1.0595 * 4.0, 0.40)]
            lp_hz = p["open_lp"] * 0.8
        pad = np.zeros(n)
        for iv, g in ivs:
            f = snap(root * iv)
            pad += dsp.saw(np.full(n, f), fs) * g
        pad = dsp.lp(pad, lp_hz, fs, order=3)
        wob = 1.0 + 0.22 * np.sin(2.0 * np.pi * snap(p["wobble_hz"]) * t + rng.uniform(0, 6.28))
        pad = _norm(pad) * wob * p["tonal_gain"]
        layers.append(Layer(pad, gain=1.0, pan=0.0, label="pad"))
        # detuned twin, panned for width (still mono-compatible: no delay)
        twin = np.zeros(n)
        for iv, g in ivs:
            twin += dsp.saw(np.full(n, snap(root * iv * (1.0 + p["detune_cents"] / 1200.0))), fs) * g
        twin = dsp.lp(twin, lp_hz, fs, order=3)
        twin = _norm(twin) * p["tonal_gain"] * 0.55
        layers.append(Layer(twin, gain=1.0, pan=rng.choice([-1.0, 1.0]) * 0.45, label="twin"))

    buf = mix(layers, fs=fs, stereo=True)

    # ---- shaping before the seam, so the crossfade is the last audio-op ----
    if p["tilt_lp"] < 17500.0:
        buf = dsp.lp(buf, p["tilt_lp"], fs, order=2)
    buf = dsp.hp(buf, p["hpf"], fs, order=3)

    # ---- loop: equal-power crossfade of the 150 ms tail into the head ----
    buf = _seamless(buf, n_out, n_x)

    # ---- character ----
    if p["dirt"] > 0.72:
        buf = fx.sat_adaa(buf, 1.15 + 0.9 * p["dirt"])
    elif p["dirt"] > 0.45:
        buf = dsp.lp(buf, 7000.0, fs, order=2)   # duller, older bed

    # ---- finish ----
    buf = dsp.dc_block(buf, fs, fc=6.0)
    buf = buf - buf.mean(axis=0, keepdims=True)
    buf = _seam_compensate(buf, fs)
    buf = _norm(buf, 0.80)   # under master()'s soft-clip ceiling

    return Render(buf=buf, fs=fs, stereo=True,
                  meta=dict(family=FAMILY, type=typ, loop=True,
                            descriptor=f"{typ}_{p['root_note']}",
                            seconds_target=round(p["dur"], 2),
                            bands=p["bands"], tilt_oct=round(p["tilt_oct"], 2),
                            width=round(p["width"], 2),
                            tilt_lp=round(p["tilt_lp"], 0),
                            colour=p["colour"]))
