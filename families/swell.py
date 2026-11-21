"""Family: swell — reverse swells, whooshes, transitions, air.

Five gestures, all mono one-shots:
  reverse     reverse-cymbal / reverse-noise: dense bright noise + a metallic
              partial cluster, exponential rise, hard stop at full level
  revsynth    reverse synth: a chord whose lowpass "opens" (dark copy
              crossfaded into a bright copy, since the core has no time-varying
              filter) over a rising amplitude, hard stop
  whoosh      band-passed pass: moving centre, q 1.6-3.0, skewed bell envelope
              peaking at 75-90 % of the file; some passes sweep down
  transition  noise rise into a bright transient in the last ~100 ms, which
              then decays out to the end (the "rise into a hit" form)
  air         airy reverse swell: high-passed noise, gentle landing

Every render carries a low-level broadband "bed" so the LTAS stays continuous
(the audit's spectral-hole check) and a dc/edge-safe finish.

Contract:
    FAMILY, DESCRIPTORS, sample_params(rng), generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "swell"
DESCRIPTORS = ["swell", "whoosh", "reverse", "transition", "air"]

TYPES = ("reverse", "revsynth", "whoosh", "transition", "air")

# intervals for the reverse-synth chord
CHORDS = (
    (1.0,),
    (1.0, 1.5),
    (1.0, 1.25, 1.5),
    (1.0, 1.2, 1.5),
    (1.0, 1.4142),
    (1.0, 2.0),
    (1.0, 1.25, 1.5, 2.0),
    (1.0, 1.5, 2.0),
)

# inharmonic ratios for the reverse-cymbal shimmer (metal-ish)
SHIMMER = (1.0, 2.76, 5.40, 8.93, 13.34)

ROOTS = ["C3", "E3", "G3", "A3", "C4", "E4", "G4", "A4"]


def sample_params(rng: np.random.Generator) -> dict:
    typ = str(rng.choice(TYPES, p=[0.28, 0.18, 0.24, 0.16, 0.14]))
    if typ == "reverse":
        dur, f0 = rng.uniform(0.6, 2.5), rng.uniform(1800.0, 7000.0)
        f1, k, q, tail = f0 * rng.uniform(0.85, 1.6), rng.uniform(2.2, 4.0), rng.uniform(1.2, 2.2), rng.uniform(4.0, 14.0)
        colour_p = [0.75, 0.22, 0.03]
    elif typ == "revsynth":
        dur, f0 = rng.uniform(0.6, 2.2), rng.uniform(120.0, 520.0)
        f1, k, q, tail = f0, rng.uniform(2.0, 3.5), 2.0, rng.uniform(4.0, 16.0)
        colour_p = [0.10, 0.55, 0.35]
    elif typ == "whoosh":
        dur = rng.uniform(0.4, 1.6)
        f0, f1 = rng.uniform(200.0, 1200.0), rng.uniform(3500.0, 12000.0)
        if rng.random() < 0.40:                       # a pass can sweep down
            f0, f1 = f1, f0
        k, q, tail = rng.uniform(1.4, 2.6), rng.uniform(1.6, 3.0), rng.uniform(10.0, 70.0)
        colour_p = [0.35, 0.55, 0.10]
    elif typ == "transition":
        dur, f0 = rng.uniform(0.8, 2.5), rng.uniform(250.0, 800.0)
        f1, k, q, tail = rng.uniform(7000.0, 16000.0), rng.uniform(1.8, 3.2), rng.uniform(1.2, 2.2), 0.0
        colour_p = [0.45, 0.45, 0.10]
    else:  # air
        dur, f0 = rng.uniform(0.8, 2.5), rng.uniform(1200.0, 4000.0)
        f1, k, q, tail = f0 * rng.uniform(1.0, 2.5), rng.uniform(1.2, 2.4), 1.2, rng.uniform(60.0, 250.0)
        colour_p = [0.55, 0.45, 0.0]

    colour = str(rng.choice(["white", "pink", "brown"], p=colour_p))
    note = str(rng.choice(ROOTS))
    hit_hz = float(rng.uniform(1400.0, 4200.0)) if typ == "transition" else 0.0
    bell_peak = float(rng.uniform(0.72, 0.90)) if typ == "whoosh" else 0.0
    return dict(
        type=typ,
        dur=float(dur),
        fc0=float(f0),
        fc1=float(f1),
        k=float(k),
        q=float(q),
        tail_ms=float(tail),
        bell_peak=bell_peak,
        hit_hz=hit_hz,
        hit_len=float(rng.uniform(0.06, 0.14)),
        colour=colour,
        # spectral character
        tilt_lp=float(np.exp(rng.uniform(np.log(2600.0), np.log(18000.0)))),
        presence_hz=float(np.exp(rng.uniform(np.log(600.0), np.log(5000.0)))),
        presence_gain=float(rng.uniform(0.0, 0.35)),
        presence_q=float(rng.uniform(1.2, 3.0)),
        air_gain=float(rng.uniform(0.05, 0.35)),
        air_pow=float(rng.uniform(2.5, 6.0)),
        bed_gain=float(rng.uniform(0.04, 0.20)),
        low_gain=float(rng.uniform(0.05, 0.40)),
        hpf=float(25.0 * (1500.0 / 25.0) ** (rng.uniform(0.0, 1.0) ** 1.6)),
        # tonal / shimmer content
        chord=int(rng.integers(0, len(CHORDS))),
        open_frac=float(rng.uniform(0.25, 0.95)),
        shimmer_gain=float(rng.uniform(0.0, 0.45)),
        shimmer_base=float(rng.uniform(280.0, 900.0)),
        detune_cents=float(rng.uniform(0.0, 14.0)),
        env_shape=str(rng.choice(["pow", "exp", "staged"], p=[0.50, 0.30, 0.20])),
        dirt=float(rng.uniform(0.0, 1.0)),
        root_note=note,
        root_oct=int(note[-1]),
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return x if p <= 1e-12 else x / p * target


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f^2 noise by spectral shaping.

    NOT core.dsp.brown: that helper does `out -= detrend(out)`, which puts the
    linear trend back, so every seed returns the same ramp (CORE ISSUES in
    swell.md).
    """
    w = rng.standard_normal(int(n))
    spec = np.fft.rfft(w)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    spec /= f
    out = np.fft.irfft(spec, int(n))
    return out / (np.max(np.abs(out)) + 1e-12)


def _flat_rms(x: np.ndarray, fs: int, win_ms: float = 60.0) -> np.ndarray:
    """Equalise the local RMS so a moving bandpass does not get louder just
    because it is parked on a denser part of the noise spectrum."""
    n = len(x)
    if n == 0:
        return x
    k = min(n, max(8, int(fs * win_ms / 1000.0)))
    c = np.cumsum(np.concatenate(([0.0], x * x)))
    idx = np.arange(n)
    half = k // 2
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx - half + k, 0, n)
    win = (c[hi] - c[lo]) / np.maximum(1, hi - lo)
    return x / (np.sqrt(win) + 1e-9)


def _release(env: np.ndarray, fs: int, tail_ms: float) -> np.ndarray:
    m = int(fs * tail_ms / 1000.0)
    if 1 < m < len(env):
        env = env.copy()
        env[-m:] *= np.linspace(1.0, 0.0, m) ** 1.4
    return env


def _rise(n: int, fs: int, k: float, tail_ms: float, shape: str = "pow") -> np.ndarray:
    t = np.arange(n) / fs
    T = max(1e-6, n / fs)
    u = np.clip(t / T, 0.0, 1.0)
    if shape == "exp":
        a = 2.0 + k
        e = (np.exp(a * u) - 1.0) / (np.exp(a) - 1.0)
    elif shape == "staged":
        e = 0.55 * u ** k + 0.45 * np.clip((u - 0.55) / 0.45, 0.0, 1.0) ** (k * 0.8)
    else:
        e = u ** k
    return _release(e, fs, tail_ms)


def _bell(n: int, peak: float, fs: int, tail_ms: float) -> np.ndarray:
    """Rise-then-fall envelope whose maximum sits at `peak` (0..1) of the file."""
    u = np.linspace(0.0, 1.0, n)
    a = max(0.2, 2.2 * peak / max(1e-3, 1.0 - peak))
    b = 2.2
    e = (u ** a) * ((1.0 - u) ** b)
    e = e / (np.max(e) + 1e-12)
    return _release(e, fs, tail_ms)


def _chord_sines(f: np.ndarray, intervals, fs: int) -> np.ndarray:
    out = np.zeros_like(f)
    for iv in intervals:
        out += dsp.sine(np.clip(f * iv, 10.0, fs * 0.45), fs) * (1.0 / (1.0 + 0.5 * (iv - 1.0)))
    return out


def _hit(n: int, fs: int, rng: np.random.Generator,
         at: int, length_s: float, tone_hz: float) -> np.ndarray:
    """Bright transient placed at `at` and allowed to decay out to the end."""
    k = min(n - at, int(fs * length_s))
    out = np.zeros(n)
    if k <= 8:
        return out
    snap = dsp.hp(dsp.white(k, rng), 3200.0, fs, order=2)
    snap *= dsp.env_exp(k, fs, attack=0.0002, tau=0.0045)
    ping = dsp.sine(np.full(k, tone_hz), fs) * dsp.env_exp(k, fs, attack=0.0003, tau=0.030)
    thump = dsp.sine(np.full(k, tone_hz * 0.05), fs) * dsp.env_exp(k, fs, attack=0.001, tau=0.045)
    burst = _norm(snap + 0.55 * ping + 0.45 * thump)
    out[at:at + k] = burst
    return out


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    typ = p["type"]
    n = int(fs * p["dur"])
    esh = p["env_shape"]
    t = np.arange(n) / fs
    root = dsp.note_hz(p["root_note"][:-1], p["root_oct"])
    colour = {"white": dsp.white, "pink": dsp.pink, "brown": _brown}[p["colour"]]

    if typ == "whoosh":
        env = _bell(n, p["bell_peak"], fs, p["tail_ms"])
    elif typ == "transition":
        # build to full level exactly when the transient lands, then clear out
        u = np.clip(t / max(t[-1], 1e-6), 0.0, 1.0)
        u_land = max(0.10, 1.0 - p["hit_len"] / (n / fs))
        env = np.clip(u / u_land, 0.0, 1.0) ** max(1.0, p["k"] * 0.6)
        m = int(fs * p["hit_len"])
        if 1 < m < n:
            env[-m:] *= np.linspace(1.0, 0.0, m) ** 1.4
    else:
        env = _rise(n, fs, p["k"], p["tail_ms"], esh)

    # ---- bed: broadband floor, keeps the LTAS continuous ----
    bed = dsp.hp(dsp.lp(colour(n, rng), 16000.0, fs, order=2), 30.0, fs, order=2)
    bed = _norm(bed) * env * p["bed_gain"]

    # ---- band layer: the moving pass / the body of the swell ----
    fc = dsp.sweep(n, p["fc0"], p["fc1"], "exp")
    band = _norm(_flat_rms(dsp.moving_bp(colour(n, rng), fc, fs, q=p["q"], bands=20), fs))
    band = band * env * 0.95

    # ---- fixed presence band ----
    pres = np.zeros(n)
    if p["presence_gain"] > 0.01:
        pres = dsp.bp(dsp.white(n, rng), p["presence_hz"], fs, order=2, q=p["presence_q"])
        pres = _norm(pres) * env * p["presence_gain"]

    # ---- air: opens only near the top ----
    air = dsp.hp(dsp.white(n, rng), 5200.0, fs, order=3)
    air = _norm(air) * _rise(n, fs, p["air_pow"], max(4.0, p["tail_ms"] * 0.5)) * p["air_gain"]

    # ---- low "suck": a rising sub-bass swell under the pass ----
    f_low = max(34.0, root / 4.0)
    low = dsp.sine(dsp.sweep(n, f_low * 0.65, f_low, "exp"), fs)
    low = low * _rise(n, fs, max(1.0, p["k"] * 0.9), p["tail_ms"], esh) * p["low_gain"]

    layers = [Layer(bed, gain=1.0, label="bed"), Layer(band, gain=1.0, label="band"),
              Layer(pres, gain=1.0, label="presence"), Layer(air, gain=1.0, label="air"),
              Layer(low, gain=1.0, label="low")]

    # ---- type-specific tonal / transient content ----
    if typ == "reverse" and p["shimmer_gain"] > 0.01:
        sh = np.zeros(n)
        for i, r in enumerate(SHIMMER):
            sh += dsp.sine(np.full(n, p["shimmer_base"] * r), fs) * (0.75 ** i)
        sh = _norm(sh) * _rise(n, fs, p["k"] * 1.25, p["tail_ms"], esh) * p["shimmer_gain"]
        layers.append(Layer(sh, gain=1.0, label="shimmer"))

    if typ == "revsynth":
        # filter opening emulated by crossfading a dark and a bright copy
        chord = np.zeros(n)
        for iv in CHORDS[p["chord"]]:
            chord += dsp.saw(np.full(n, root * iv), fs)
        if p["detune_cents"] > 0.1:
            chord += dsp.saw(np.full(n, root * (2.0 ** (p["detune_cents"] / 1200.0))), fs) * 0.5
        chord = dsp.lp(chord, 7000.0, fs, order=3)
        dark = dsp.lp(chord, 550.0, fs, order=3)
        bright = dsp.lp(chord, 6500.0, fs, order=3)
        u = np.clip(t / max(t[-1], 1e-6) / max(p["open_frac"], 1e-3), 0.0, 1.0)
        w = u ** 2
        synth = _norm(dark * (1.0 - w) + bright * w) * env * 0.55
        layers.append(Layer(synth, gain=1.0, label="synth"))

    if typ == "transition":
        at = n - int(fs * p["hit_len"])
        hit = _hit(n, fs, rng, at, p["hit_len"] + 0.02, p["hit_hz"])
        layers.append(Layer(hit, gain=1.0, label="hit"))

    buf = mix(layers, fs=fs, stereo=False)

    # ---- spectral tilt ----
    if p["tilt_lp"] < 17500.0:
        buf = dsp.lp(buf, p["tilt_lp"], fs, order=2)

    # ---- character ----
    if p["dirt"] > 0.65:
        buf = fx.sat_adaa(buf, 1.3 + 1.2 * p["dirt"])
    elif p["dirt"] > 0.35:
        buf = fx.sat_adaa(buf, 1.0 + 0.6 * p["dirt"])

    # ---- finish ----
    buf = dsp.dc_block(buf, fs, fc=12.0)
    buf = dsp.hp(buf, max(30.0, p["hpf"]), fs, order=4)
    buf = dsp.fade(buf, fs, ms=4.0)
    buf = buf - float(np.mean(buf))
    buf = dsp.fade(buf, fs, ms=4.0)
    buf = _norm(buf, 0.85)   # stay under master()'s soft-clip ceiling

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, type=typ, loop=False,
                            descriptor=f"{typ}_{p['root_note']}_{p['colour']}",
                            root_hz=round(root, 2), fc0=round(p["fc0"], 1),
                            fc1=round(p["fc1"], 1), k=round(p["k"], 2),
                            tail_ms=round(p["tail_ms"], 1), colour=p["colour"],
                            tilt_lp=round(p["tilt_lp"], 0), hpf=round(p["hpf"], 0),
                            env=p["env_shape"], chord=p["chord"]))
