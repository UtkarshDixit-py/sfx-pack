"""Family: riser — build/tension rises, sweeps and uplifts.

A sellable riser is a texture that accelerates. Every render has a moving
bandpass backbone, but the *gesture* is one of five structures:

  noise    the classic: band-passed noise sweeping up + a tonal stack that
           locks on a root note for the last 12-32 %
  shepard  octave-spaced glides with Gaussian windows: the endless-rise illusion
  roll     accelerating percussive hit train (1.2 Hz -> 14-28 Hz)
  grain    accelerating granular chime cloud, grains rising in pitch
  siren    a dominant tonal glide over 2+ octaves with deep vibrato

Layers for the noise gesture:
  bed        broadband floor: keeps the LTAS continuous (no spectral holes)
  band       moving bandpass, short-time-RMS flattened so the top of the sweep
             is as loud as the bottom (peak-normalising alone buried it)
  band2      optional second sweep band, 1.5-4x up
  fill       fixed mid band, the body
  low        root/4 sine glide
  tone       additive chord, glides to the root, locks, vibrato deepens
  presence   fixed resonant formant (the voice of the riser)
  air        hiss that only opens in the last third
  ticks      optional accelerating click train
  sub        optional 38-56 Hz rumble

Contract:
    FAMILY, DESCRIPTORS, sample_params(rng), generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "riser"
DESCRIPTORS = ["riser", "build", "sweep", "tension", "uplift"]

ARCHETYPES = ("build", "sweep", "tension", "uplift")
STRUCTS = ("noise", "shepard", "roll", "grain", "siren")

# interval stacks: (partial ratio, gain). Picked per candidate, so two "tonal"
# risers are not the same chord with a different root.
CHORDS = (
    ((1.0, 1.00),),
    ((1.0, 1.00), (2.0, 0.50)),
    ((1.0, 1.00), (1.5, 0.50)),
    ((1.0, 1.00), (1.25, 0.48), (1.5, 0.44)),
    ((1.0, 1.00), (1.2, 0.45), (1.5, 0.40)),
    ((1.0, 1.00), (1.4142, 0.46)),
    ((1.0, 1.00), (2.0, 0.40), (3.0, 0.20), (4.0, 0.10)),
    ((1.0, 1.00), (0.5, 0.42)),
)

ROOT_NOTES = {
    "build":   ["A2", "C3", "E3", "G3", "A3", "C4"],
    "sweep":   ["C3", "E3", "G3", "A3", "C4", "E4"],
    "tension": ["A2", "D3", "F3", "G#3", "A3", "D4"],
    "uplift":  ["C3", "E3", "G3", "A3", "C4", "E4"],
}

# How far the structure layer is allowed to duck the noise gesture.
STRUCT_NOISE_MUL = {"noise": 1.0, "shepard": 0.45, "roll": 0.45, "grain": 0.35, "siren": 0.55}
STRUCT_TONE_MUL = {"noise": 1.00, "shepard": 0.00, "roll": 0.50, "grain": 0.40, "siren": 1.15}

# Discrete spectral "voices". Continuous ranges alone re-created the same riser
# over and over, so timbre is picked from corners of the space.
VOICES = ("air", "hiss", "dark", "rumble", "tonal", "metal")


def _voice(rng: np.random.Generator) -> dict:
    voice = str(rng.choice(VOICES, p=[0.18, 0.18, 0.18, 0.13, 0.17, 0.16]))
    if voice == "air":
        colour, tilt = "white", rng.uniform(10000.0, 18000.0)
        phz, pgain, pq = rng.uniform(2200.0, 4800.0), rng.uniform(0.08, 0.45), rng.uniform(1.3, 2.8)
        low, air, band, tmul, fill = rng.uniform(0.02, 0.20), rng.uniform(0.15, 0.40), rng.uniform(0.55, 1.00), rng.uniform(0.45, 0.75), rng.uniform(0.05, 0.28)
    elif voice == "hiss":
        colour, tilt = "pink", rng.uniform(6000.0, 14000.0)
        phz, pgain, pq = rng.uniform(700.0, 3200.0), rng.uniform(0.06, 0.38), rng.uniform(1.1, 2.4)
        low, air, band, tmul, fill = rng.uniform(0.04, 0.25), rng.uniform(0.08, 0.30), rng.uniform(0.60, 1.00), rng.uniform(0.60, 1.00), rng.uniform(0.10, 0.36)
    elif voice == "dark":
        colour, tilt = "brown", rng.uniform(1800.0, 6000.0)
        phz, pgain, pq = rng.uniform(300.0, 1400.0), rng.uniform(0.06, 0.34), rng.uniform(1.1, 2.4)
        low, air, band, tmul, fill = rng.uniform(0.15, 0.50), rng.uniform(0.02, 0.12), rng.uniform(0.50, 0.90), rng.uniform(0.70, 1.10), rng.uniform(0.15, 0.42)
    elif voice == "rumble":
        colour, tilt = "brown", rng.uniform(1600.0, 4500.0)
        phz, pgain, pq = rng.uniform(150.0, 800.0), rng.uniform(0.0, 0.24), rng.uniform(1.0, 1.9)
        low, air, band, tmul, fill = rng.uniform(0.30, 0.65), rng.uniform(0.01, 0.08), rng.uniform(0.40, 0.80), rng.uniform(0.80, 1.10), rng.uniform(0.08, 0.30)
    elif voice == "tonal":
        colour, tilt = "pink", rng.uniform(4000.0, 14000.0)
        phz, pgain, pq = rng.uniform(500.0, 3000.0), rng.uniform(0.0, 0.24), rng.uniform(1.1, 2.6)
        low, air, band, tmul, fill = rng.uniform(0.06, 0.32), rng.uniform(0.02, 0.18), rng.uniform(0.30, 0.65), rng.uniform(1.20, 1.90), rng.uniform(0.05, 0.26)
    else:  # metal: narrow high formant, cutting
        colour, tilt = "white", rng.uniform(8000.0, 18000.0)
        phz, pgain, pq = rng.uniform(2500.0, 4800.0), rng.uniform(0.15, 0.48), rng.uniform(2.2, 4.0)
        low, air, band, tmul, fill = rng.uniform(0.03, 0.22), rng.uniform(0.10, 0.34), rng.uniform(0.45, 0.85), rng.uniform(0.55, 0.85), rng.uniform(0.08, 0.30)
    return dict(voice=voice, colour=colour, tilt_lp=float(tilt),
                presence_hz=float(phz), presence_gain=float(pgain), presence_q=float(pq),
                low_gain=float(low), air_gain=float(air), band_gain=float(band),
                tone_mul=float(tmul), fill_gain=float(fill))


def sample_params(rng: np.random.Generator) -> dict:
    """Design the spread per archetype; the voice, structure and extras are
    rolled independently so 60 candidates never collapse into six clones."""
    arch = str(rng.choice(ARCHETYPES, p=[0.34, 0.24, 0.24, 0.18]))
    if arch == "build":
        dur, fc0, fc1 = rng.uniform(1.8, 4.0), rng.uniform(220.0, 500.0), rng.uniform(7000.0, 14000.0)
        k, q = rng.uniform(2.4, 4.0), rng.uniform(1.5, 2.5)
        tone, tone_k, tail = rng.uniform(0.18, 0.75), rng.uniform(0.5, 0.9), rng.uniform(4.0, 14.0)
        tick_p = 0.55
    elif arch == "sweep":
        dur, fc0, fc1 = rng.uniform(1.0, 2.6), rng.uniform(350.0, 900.0), rng.uniform(6000.0, 15000.0)
        k, q = rng.uniform(1.4, 2.5), rng.uniform(1.2, 2.0)
        tone, tone_k, tail = rng.uniform(0.05, 0.38), rng.uniform(0.5, 1.0), rng.uniform(40.0, 130.0)
        tick_p = 0.12
    elif arch == "tension":
        dur, fc0, fc1 = rng.uniform(2.0, 4.0), rng.uniform(180.0, 380.0), rng.uniform(3500.0, 9000.0)
        k, q = rng.uniform(2.0, 3.2), rng.uniform(1.7, 2.8)
        tone, tone_k, tail = rng.uniform(0.25, 0.80), rng.uniform(0.4, 0.8), rng.uniform(90.0, 260.0)
        tick_p = 0.65
    else:  # uplift
        dur, fc0, fc1 = rng.uniform(1.5, 3.2), rng.uniform(280.0, 700.0), rng.uniform(8000.0, 15000.0)
        k, q = rng.uniform(2.0, 3.2), rng.uniform(1.3, 2.2)
        tone, tone_k, tail = rng.uniform(0.20, 0.75), rng.uniform(0.4, 0.8), rng.uniform(180.0, 420.0)
        tick_p = 0.25

    struct = str(rng.choice(STRUCTS, p=[0.38, 0.15, 0.18, 0.14, 0.15]))
    v = _voice(rng)
    note = str(rng.choice(ROOT_NOTES[arch]))

    if struct == "siren":                      # pitch glide is the whole point
        glide = float(rng.uniform(6.0, 16.0))
        vib = float(rng.uniform(18.0, 60.0))
        tone = max(tone, rng.uniform(0.60, 0.95))
    else:
        glide = float(rng.choice([4.0, 6.0, 8.0] if arch != "sweep" else [4.0, 6.0]))
        vib = float(rng.uniform(0.0, 14.0))

    # --- within-voice axes: tremolo, a second sweep, band grit, sub, chord ---
    sub_boost = float(rng.uniform(0.20, 0.42)) if (v["voice"] == "rumble" or rng.random() < 0.30) else 0.0
    return dict(
        archetype=arch,
        struct=struct,
        dur=float(dur),
        fc0=float(fc0),
        fc1=float(fc1),
        k=float(k),
        q=float(q),
        tone_gain=float(min(0.95, tone * v["tone_mul"])),
        tone_k=float(tone_k),
        tail_ms=float(tail),
        glide_ratio=glide,
        lock_frac=float(rng.uniform(0.12, 0.32)),
        vib_cents=vib,
        vib_rate=float(rng.uniform(4.5, 7.0)),
        bed_gain=float(rng.uniform(0.02, 0.35)),
        fill_mul=float(rng.uniform(0.30, 1.50)),
        air_pow=float(rng.uniform(3.5, 6.5)),
        env_shape=str(rng.choice(["pow", "exp", "staged"], p=[0.52, 0.26, 0.22])),
        flutter=bool(rng.random() < 0.45),
        flutter_hz=float(rng.uniform(2.5, 13.0)),
        flutter_depth=float(rng.uniform(0.15, 0.55)),
        dual=bool(rng.random() < 0.42),
        dual_ratio=float(rng.uniform(1.5, 4.0)),
        dual_gain=float(rng.uniform(0.18, 0.45)),
        grit=bool(rng.random() < 0.35),
        grit_drive=float(rng.uniform(1.8, 4.0)),
        chord=int(rng.integers(0, len(CHORDS))),
        sub_boost=sub_boost,
        # highpass: transparent on some renders, a thin air-only riser on
        # others. A large MFCC lever and a real production distinction.
        hpf=float(25.0 * (3000.0 / 25.0) ** (rng.uniform(0.0, 1.0) ** 1.8)),
        pump=bool(rng.random() < 0.22),
        pump_hz=float(rng.uniform(2.0, 9.0)),
        pump_duty=float(rng.uniform(0.35, 0.80)),
        hi_ratio=float(rng.choice([2.0, 3.0, 4.0, 5.0, 6.0])),
        hi_gain=float(rng.uniform(0.08, 0.40)),
        tick_rate_hi=float(rng.uniform(10.0, 24.0)),
        ticks=bool(rng.random() < tick_p),
        dirt=float(rng.uniform(0.0, 1.0)),
        root_note=note,
        root_oct=int(note[-1]),
        **v,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return x if p <= 1e-12 else x / p * target


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f^2 noise by spectral shaping.

    NOT core.dsp.brown: that helper does `out -= detrend(out)`; detrend already
    removes the linear trend, so the subtraction puts it back and every seed
    returns the *same* linear ramp (see CORE ISSUES in riser.md).
    """
    w = rng.standard_normal(int(n))
    spec = np.fft.rfft(w)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    spec /= f
    out = np.fft.irfft(spec, int(n))
    return out / (np.max(np.abs(out)) + 1e-12)


def _flat_rms(x: np.ndarray, fs: int, win_ms: float = 70.0) -> np.ndarray:
    """Divide out the short-time RMS.

    A swept bandpass over pink/brown noise is 20+ dB hotter at the bottom of
    the sweep than at the top; peak-normalising the layer then hides the whole
    upper sweep under the tone. Flattening the local RMS keeps the band
    audible all the way up, which is the entire point of a riser.
    """
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
    """Linear-to-zero release on the last tail_ms. 4-14 ms reads as a hard stop."""
    m = int(fs * tail_ms / 1000.0)
    if 1 < m < len(env):
        env = env.copy()
        env[-m:] *= np.linspace(1.0, 0.0, m) ** 1.4
    return env


def _rise(n: int, fs: int, k: float, tail_ms: float, shape: str = "pow") -> np.ndarray:
    """Three rise shapes with the same endpoints, different shoulders:
    pow = t^k, exp = late exponential surge, staged = a slow stage plus a
    second accelerated stage (the classic two-step trailer build)."""
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


def _pitch_curve(n: int, fs: int, f_start: float, f_end: float,
                 lock_frac: float, vib_cents: float, vib_rate: float) -> np.ndarray:
    """Exponential glide to f_end, then hold. Vibrato deepens toward the lock."""
    n_lock = int(n * lock_frac)
    n_pre = max(2, n - n_lock)
    f = np.concatenate([dsp.sweep(n_pre, f_start, f_end, "exp"),
                        np.full(n - n_pre, f_end)])
    if vib_cents > 0.05:
        t = np.arange(n) / fs
        depth = (vib_cents / 1200.0) * np.clip(t / max(t[-1], 1e-6), 0.0, 1.0) ** 2
        f = f * (1.0 + depth * np.sin(2.0 * np.pi * vib_rate * t))
    return f


def _onsets_for(rate: np.ndarray, fs: int) -> np.ndarray:
    phase = np.cumsum(rate) / fs
    return np.flatnonzero(np.diff(np.floor(phase))) + 1


def _tick_train(n: int, fs: int, rng: np.random.Generator,
                rate_hi: float, accent: float) -> np.ndarray:
    """Accelerating click train: rate rises 2.5 Hz -> rate_hi over the file."""
    onsets = _onsets_for(dsp.sweep(n, 2.5, rate_hi, "lin"), fs)
    out = np.zeros(n)
    k = max(8, int(fs * 0.0025))
    burst = dsp.hp(dsp.white(k, rng), 3200.0, fs, order=2)
    burst *= dsp.env_exp(k, fs, attack=0.00015, tau=0.0007)
    for i in onsets:
        m = min(k, n - i)
        if m <= 2:
            continue
        amp = accent * (i / n) ** 1.6 * rng.uniform(0.55, 1.0)
        out[i:i + m] += burst[:m] * amp
    return _norm(out)


def _roll(n: int, fs: int, rng: np.random.Generator, rate_hi: float) -> np.ndarray:
    """Accelerating hit train: 1.2 Hz to rate_hi. A drum-roll build."""
    onsets = _onsets_for(dsp.sweep(n, 1.2, rate_hi, "lin"), fs)
    out = np.zeros(n)
    k = int(fs * 0.055)
    body = dsp.bp(dsp.white(k, rng), 900.0, fs, order=2, q=1.2)
    body *= dsp.env_exp(k, fs, attack=0.0004, tau=0.013)
    snap = dsp.hp(dsp.white(k, rng), 4200.0, fs, order=2)
    snap *= dsp.env_exp(k, fs, attack=0.0002, tau=0.0035)
    burst = _norm(body + 0.55 * snap)
    for i in onsets:
        m = min(k, n - i)
        if m <= 2:
            continue
        amp = (i / n) ** 1.2 * rng.uniform(0.70, 1.0)
        out[i:i + m] += burst[:m] * amp
    return _norm(out)


def _shepard(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Octave-spaced sines with a Gaussian log-frequency window. Each sine fades
    out at the top and its neighbour enters from below, so the pitch keeps
    rising forever (Shepard 1964)."""
    centre = rng.uniform(600.0, 1200.0)
    octaves = rng.uniform(1.0, 1.7)
    sigma = rng.uniform(0.85, 1.35)
    u = np.linspace(0.0, octaves, n)
    lc = np.log2(centre)
    out = np.zeros(n)
    for k in range(-4, 4):
        logf = lc + k + u
        g = np.exp(-0.5 * ((logf - lc) / sigma) ** 2)
        if g.max() < 1e-3:
            continue
        f = np.clip(2.0 ** logf, 20.0, fs * 0.45)
        out += dsp.sine(f, fs) * g
    return _norm(out)


def _grain_rise(n: int, fs: int, rng: np.random.Generator, root: float) -> np.ndarray:
    """Accelerating granular chime cloud: 6-14 grains/s ramping to 40-80/s,
    each grain a Hann-windowed sine whose pitch rises with time."""
    onsets = _onsets_for(dsp.sweep(n, rng.uniform(6.0, 14.0), rng.uniform(40.0, 80.0), "lin"), fs)
    out = np.zeros(n)
    for i in onsets:
        gl = int(fs * rng.uniform(0.008, 0.045))
        m = min(gl, n - i)
        if m <= 4:
            continue
        f0 = root * 2.0 ** rng.uniform(0.4, 3.2) * (1.0 + 1.2 * (i / n))
        f0 = min(f0, fs * 0.4)
        w = np.hanning(m)
        out[i:i + m] += np.sin(2.0 * np.pi * f0 * np.arange(m) / fs) * w * rng.uniform(0.5, 1.0)
    return _norm(out)


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    arch, struct = p["archetype"], p["struct"]
    n = int(fs * p["dur"])
    tail = float(p["tail_ms"])
    root = dsp.note_hz(p["root_note"][:-1], p["root_oct"])
    esh = p["env_shape"]
    t = np.arange(n) / fs

    colour = {"white": dsp.white, "pink": dsp.pink, "brown": _brown}[p["colour"]]
    noise_mul = STRUCT_NOISE_MUL[struct]

    # ---- bed: broadband floor. Guarantees a continuous LTAS. ----
    bed = colour(n, rng)
    bed = dsp.lp(dsp.hp(bed, 30.0, fs, order=2), 16000.0, fs, order=2)
    bed = _norm(bed) * _rise(n, fs, p["k"] * 0.75, tail, esh) * p["bed_gain"] * noise_mul

    # ---- moving band(s), short-time-RMS flattened ----
    env_band = _rise(n, fs, p["k"], tail, esh)
    flutter_mod = np.ones(n)
    if p["flutter"]:
        d = p["flutter_depth"]
        flutter_mod = 1.0 - 0.5 * d + 0.5 * d * np.sin(2.0 * np.pi * p["flutter_hz"] * t)

    fc = dsp.sweep(n, p["fc0"], p["fc1"], "exp")
    band = _norm(_flat_rms(dsp.moving_bp(colour(n, rng), fc, fs, q=p["q"], bands=20), fs))
    if p["grit"]:
        band = _norm(fx.sat_adaa(band * 1.2, p["grit_drive"]))
    band = band * flutter_mod * env_band * p["band_gain"] * noise_mul

    band2 = np.zeros(n)
    if p["dual"]:
        fc2 = np.clip(fc * p["dual_ratio"], 40.0, fs * 0.45)
        band2 = _norm(_flat_rms(dsp.moving_bp(colour(n, rng), fc2, fs,
                                              q=max(1.0, p["q"] - 0.4), bands=20), fs))
        band2 = band2 * flutter_mod * env_band * p["dual_gain"] * noise_mul

    # ---- fixed mid fill: carries body, closes the low-mid gap ----
    fill = dsp.bp(colour(n, rng), 800.0, fs, order=2, q=4.0)
    fill = _norm(fill) * _rise(n, fs, p["k"] * 0.9, tail, esh) * p["fill_gain"] * p["fill_mul"] * noise_mul

    # ---- presence formant: a resonant fixed band, the "voice" of the riser ----
    pres = np.zeros(n)
    if p["presence_gain"] > 0.01:
        pres = dsp.bp(dsp.white(n, rng), p["presence_hz"], fs, order=2, q=p["presence_q"])
        pres = _norm(pres) * _rise(n, fs, p["k"] * 0.8, tail, esh) * p["presence_gain"] * noise_mul

    # ---- low: root/4 sine glide (kept above 30 Hz so the hp does not eat it) ----
    f_low = max(32.0, root / 4.0)
    low_f = dsp.sweep(n, f_low * 0.72, f_low, "exp")
    low = dsp.sine(low_f, fs) * _rise(n, fs, p["k"] * 1.1, tail, esh) * p["low_gain"] * noise_mul

    sub = np.zeros(n)
    if p["sub_boost"] > 0.0:
        sub = dsp.sine(np.full(n, rng.uniform(38.0, 56.0)), fs)
        sub = sub * _rise(n, fs, p["k"] * 1.4, tail, esh) * p["sub_boost"]

    # ---- tone: additive chord that glides and locks ----
    f_curve = _pitch_curve(n, fs, root / p["glide_ratio"], root, p["lock_frac"],
                           p["vib_cents"], p["vib_rate"])
    tone = np.zeros(n)
    for ratio, g in CHORDS[p["chord"]]:
        tone += dsp.sine(np.clip(f_curve * ratio, 10.0, fs * 0.45), fs) * g
    if arch in ("tension", "uplift"):
        # +7 cents twin: mono-compatible beating, no panning
        tone += dsp.sine(f_curve * (2.0 ** (7.0 / 1200.0)), fs) * 0.35
    # one random upper harmonic: the tone stack is not the same chord each time
    tone += dsp.sine(np.clip(f_curve * p["hi_ratio"], 10.0, fs * 0.45), fs) * p["hi_gain"]
    tone = _norm(tone) * _rise(n, fs, p["tone_k"], tail, esh) * p["tone_gain"] * STRUCT_TONE_MUL[struct]

    # ---- air: opens only at the top ----
    air = dsp.hp(dsp.white(n, rng), 5200.0, fs, order=3)
    air = _norm(air) * _rise(n, fs, p["air_pow"], max(4.0, tail * 0.5)) * p["air_gain"] * noise_mul

    # ---- structure layer ----
    special = np.zeros(n)
    if struct == "shepard":
        special = _shepard(n, fs, rng) * _rise(n, fs, p["k"] * 0.8, tail, esh) * 0.9
    elif struct == "roll":
        special = _roll(n, fs, rng, p["tick_rate_hi"]) * _rise(n, fs, 1.1, tail, esh) * 0.9
    elif struct == "grain":
        special = _grain_rise(n, fs, rng, root) * _rise(n, fs, p["k"], tail, esh) * 0.8

    layers = [
        Layer(bed, gain=1.0, label="bed"),
        Layer(band, gain=1.0, label="band"),
        Layer(band2, gain=1.0, label="band2"),
        Layer(fill, gain=1.0, label="fill"),
        Layer(low, gain=1.0, label="low"),
        Layer(sub, gain=1.0, label="sub"),
        Layer(tone, gain=1.0, label="tone"),
        Layer(pres, gain=1.0, label="presence"),
        Layer(air, gain=1.0, label="air"),
        Layer(special, gain=1.0, label="structure"),
    ]
    if p["ticks"] and struct != "roll":
        ticks = _tick_train(n, fs, rng, p["tick_rate_hi"], 0.22)
        ticks *= _rise(n, fs, 1.0, tail)
        layers.append(Layer(_norm(ticks) * 0.22, gain=1.0, label="ticks"))

    buf = mix(layers, fs=fs, stereo=False)

    # ---- spectral tilt: one of the biggest timbre levers ----
    if p["tilt_lp"] < 17500.0:
        buf = dsp.lp(buf, p["tilt_lp"], fs, order=2)

    # ---- pumping gate: sidechain-style riser, a distinct production sound ----
    if p["pump"]:
        buf = fx.stutter(buf, fs=fs, rate_hz=p["pump_hz"], duty=p["pump_duty"], ramp_ms=3.0)

    # ---- character ----
    if p["dirt"] > 0.65:
        buf = fx.sat_adaa(buf, 1.4 + 1.2 * p["dirt"])
    elif p["dirt"] > 0.35:
        buf = fx.sat_adaa(buf, 1.0 + 0.6 * p["dirt"])

    # ---- finish: kill sub-sonic residue, de-click ----
    buf = dsp.dc_block(buf, fs, fc=12.0)
    buf = dsp.hp(buf, max(30.0, p["hpf"]), fs, order=4)
    buf = dsp.fade(buf, fs, ms=4.0)
    # A low tone that does not end on an integer number of cycles leaves a
    # residual sample mean (the audit's dc_offset). Zero it, then re-fade so
    # the first/last samples stay at zero.
    buf = buf - float(np.mean(buf))
    buf = dsp.fade(buf, fs, ms=4.0)
    # Stay under master()'s soft_clip ceiling (0.995). tanh compresses the
    # positive and negative half-cycles of an asymmetric peak unequally, which
    # walks a DC offset into the file; a linear -6 dBFS normalise afterwards
    # does not. Peak target is arbitrary: master() re-normalises.
    buf = _norm(buf, 0.85)

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, archetype=arch, struct=struct, loop=False,
                            descriptor=f"{arch}_{struct}_{p['root_note']}_{p['voice']}",
                            root_hz=round(root, 2),
                            fc0=round(p["fc0"], 1), fc1=round(p["fc1"], 1),
                            k=round(p["k"], 2), voice=p["voice"], colour=p["colour"],
                            tilt_lp=round(p["tilt_lp"], 0),
                            presence_hz=round(p["presence_hz"], 0),
                            env=p["env_shape"], chord=p["chord"],
                            flutter=(round(p["flutter_hz"], 1) if p["flutter"] else 0.0),
                            dual=(round(p["dual_ratio"], 2) if p["dual"] else 0.0),
                            grit=p["grit"], hpf=round(p["hpf"], 0),
                            pump=(round(p["pump_hz"], 1) if p["pump"] else 0.0),
                            ticks=p["ticks"]))
