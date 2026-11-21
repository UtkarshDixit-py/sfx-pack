"""Family: bass_synth — pitched synth bass (acid / reese / lead / fm), one per note.

PITCHING: `sample_params` assigns the note from the call index, so 12 consecutive
draws cover C..B exactly once (60 draws = five full passes). `generate` resolves
the note with `dsp.note_hz(note, octave)` and oscillates at that frequency; there
is no pitch-shifting. Other macro-params come from the seed's rng.

Archetypes (all mono, 0.30-1.20 s, octave 2 = 65-124 Hz):
  acid    band-limited saw OR square -> resonant lowpass, envelope-swept cutoff
  reese   two detuned saws (+ soft sub), static lowpass, beating
  lead    duty-square OR saw with vibrato -> static lowpass
  fm      carrier at the root with a decaying modulation index -> lowpass

DIVERSITY. Measured on core's block dedup (min over {MFCC mean, MFCC std,
log-mel, 24-pt RMS envelope, timbre scalars}), the amplitude-envelope block is
the real separator — the spectral blocks stay similar for any harmonic bass. So a
per-draw ENVELOPE SHAPE is stratified with `(c + c//12) % 4`, which gives all four
shapes to the 12 notes of a short run and to the five draws of every note in a
60-draw run:
  pluck   fast attack, short exponential decay (sits in silence)
  ring    fast attack, long exponential decay
  hold    fast attack, flat sustain, gated release
  trem    fast attack, sustained with a 4-10 Hz amplitude tremolo
Decay and spectral tilt additionally follow a low-discrepancy sequence keyed to
the call index, so the draws of one note are always spread. Waveform (saw/square,
duty), cutoff, resonance Q, detune, FM ratio/index and an optional edge/harmonic
layer vary per seed on top of that.

Envelope-swept lowpass: core has no time-varying lowpass (`moving_bp` is a
bandpass and would drop the fundamental a bass needs). Logged as a CORE ISSUE and
worked around with `_swept_lp`, an RBJ biquad whose cutoff updates block-wise
with filter state carried.

Delivered at 0.9 peak so render.master's soft_clip never engages (that
nonlinearity shifts the mean of a low-frequency one-shot and the audit reads DC).
Final hp at 32 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sg

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "bass_synth"
DESCRIPTORS = ["bass", "acid", "reese", "fm"]

OCTAVE = 2                       # C2..B2 = 65.4..123.5 Hz
_WORD = {"lead": "bass", "acid": "acid", "reese": "reese", "fm": "fm"}
_SHAPES = ("pluck", "ring", "hold", "trem")
_NOTE_CURSOR = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    if m <= 1e-12:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) / m * peak


def _release(w: np.ndarray, fs: int, ms: float = 20.0) -> np.ndarray:
    m = int(fs * ms / 1000.0)
    if m > 1 and w.size > m:
        w = w.copy()
        w[-m:] *= np.linspace(1.0, 0.0, m)
    return w


def _tick() -> int:
    """Call index: c % 12 covers the notes; c also drives the low-discrepancy
    spread of the strongest diversity axes."""
    global _NOTE_CURSOR
    c = _NOTE_CURSOR
    _NOTE_CURSOR += 1
    return c


def _frac(x: float) -> float:
    return float(x) % 1.0


def _rbj_lp(fc: float, q: float, fs: int):
    """RBJ cookbook 2-pole lowpass with resonance (Q). Normalised by a0."""
    fc = float(np.clip(fc, 20.0, fs * 0.45))
    w0 = 2.0 * np.pi * fc / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * max(q, 0.5))
    b0 = (1.0 - cw) / 2.0
    b = np.array([b0, 1.0 - cw, b0]) / (1.0 + alpha)
    a = np.array([1.0 + alpha, -2.0 * cw, 1.0 - alpha]) / (1.0 + alpha)
    return b, a


def _swept_lp(x: np.ndarray, fc_curve: np.ndarray, fs: int,
              q: float = 2.0, block: int = 128) -> np.ndarray:
    """Time-varying resonant lowpass. Cutoff updated per block, state carried."""
    n = len(x)
    y = np.zeros(n)
    zi = np.zeros(2)
    pos = 0
    while pos < n:
        end = min(pos + block, n)
        b, a = _rbj_lp(float(np.mean(fc_curve[pos:end])), q, fs)
        seg, zi = sg.lfilter(b, a, x[pos:end], zi=zi)
        y[pos:end] = seg
        pos = end
    return y


def _amp_env(n: int, fs: int, shape: str, attack: float, tau: float,
             curve: float, trem_hz: float, trem_depth: float) -> np.ndarray:
    """Qualitatively different amplitude envelopes (the dedup discriminator)."""
    dur = n / fs
    t = np.arange(n) / fs
    a = max(2e-3, attack)
    if shape == "hold":
        rel = max(0.02, dur * 0.12)
        e = np.clip(t / a, 0.0, 1.0) * np.clip((dur - t) / rel, 0.0, 1.0)
        return e ** curve
    if shape == "trem":
        rel = min(0.06, dur * 0.20)
        sus = np.clip(t / a, 0.0, 1.0) * np.clip((dur - t) / rel, 0.0, 1.0)
        trem = 1.0 - trem_depth * (0.5 + 0.5 * np.sin(2.0 * np.pi * trem_hz * t))
        return sus * trem
    return dsp.env_exp(n, fs, attack=a, tau=tau, curve=curve)     # pluck / ring


def _click(n: int, fs: int, rng: np.random.Generator, amount: float) -> np.ndarray:
    if amount <= 0:
        return np.zeros(n)
    k = max(6, int(fs * 0.005))
    burst = dsp.hp(dsp.white(k, rng), float(rng.uniform(1200.0, 5000.0)), fs, order=3)
    c = np.zeros(k)
    c[:k] = burst * np.exp(-np.arange(k) / (k * 0.30))
    out = np.zeros(n)
    out[:k] = c
    return _norm(out, amount)


# --------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    c = _tick()
    note = dsp.NOTE_NAMES[c % 12]
    arch = str(rng.choice(["acid", "reese", "lead", "fm"], p=[0.32, 0.25, 0.25, 0.18]))
    dur = float(np.clip(rng.uniform(0.30, 1.20), 0.30, 1.20))

    # envelope shape: (c + c//12) % 4 gives all four shapes across the 12 notes
    # of a short run AND across the five draws of each note (c, c+12, ...) of a
    # 60-draw run
    shape = _SHAPES[(c + (c // 12)) % 4]
    if shape == "pluck":
        tau = float(2.0 ** (np.log2(0.05) + _frac(c * 0.4142135624)
                            * (np.log2(0.22) - np.log2(0.05))))
    elif shape == "ring":
        tau = float(2.0 ** (np.log2(0.50) + _frac(c * 0.4142135624)
                            * (np.log2(1.20) - np.log2(0.50))))
    else:
        tau = 0.30

    p = dict(
        note=note, key=note, octave=OCTAVE, arch=arch, dur=dur, shape=shape,
        attack=float(rng.uniform(0.002, 0.015)),
        tau=tau,
        curve=float(rng.uniform(0.5, 2.5)),
        trem_hz=float(rng.uniform(4.0, 10.0)),
        trem_depth=float(rng.uniform(0.35, 0.85)),
        drive=1.5, click=0.0, duty=0.5, wave="saw",
        h2=float(rng.uniform(0.0, 0.35)), h3=float(rng.uniform(0.0, 0.20)),
    )
    if arch == "acid":
        p.update(wave=str(rng.choice(["saw", "square"])),
                 duty=float(rng.uniform(0.15, 0.70)),
                 cut_start=float(rng.uniform(4.0, 30.0)),
                 cut_end=float(rng.uniform(1.2, 5.0)),
                 cut_tau=float(rng.uniform(0.03, 0.40)),
                 q=float(rng.uniform(1.5, 9.0)),
                 drive=float(rng.uniform(1.2, 3.2)),
                 click=float(rng.uniform(0.05, 0.24)))
    elif arch == "reese":
        p.update(detune=float(rng.uniform(3.0, 40.0)),
                 cut_mult=float(rng.uniform(2.0, 16.0)),
                 sub=float(rng.uniform(0.10, 0.50)),
                 drive=float(rng.uniform(1.0, 2.6)))
    elif arch == "lead":
        p.update(wave=str(rng.choice(["square", "saw"])),
                 duty=float(rng.uniform(0.15, 0.70)),
                 cut_mult=float(rng.uniform(2.0, 14.0)),
                 vib=float(rng.uniform(0.0, 0.010)),
                 vib_hz=float(rng.uniform(3.0, 8.0)),
                 drive=float(rng.uniform(1.0, 2.4)),
                 click=float(rng.uniform(0.0, 0.10)))
    else:  # fm
        p.update(fm_ratio=float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])),
                 idx0=float(rng.uniform(1.0, 9.0)),
                 idx_tau=float(rng.uniform(0.02, 0.30)),
                 cut_mult=float(rng.uniform(2.0, 16.0)),
                 pitch_mult=float(rng.uniform(1.2, 2.8)),
                 pitch_t=float(rng.uniform(0.010, 0.035)),
                 drive=float(rng.uniform(1.0, 2.4)),
                 click=float(rng.uniform(0.04, 0.16)))
    p["edge"] = float(rng.uniform(0.0, 0.40))
    p["edge_hz"] = float(rng.uniform(600.0, 4000.0))
    # spectral tilt on a low-discrepancy sequence: dark -> bright per draw
    p["tilt_hz"] = float(2.0 ** (np.log2(400.0) + _frac(c * 0.6180339887)
                                 * (np.log2(15000.0) - np.log2(400.0))))
    return p


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    root = dsp.note_hz(p["note"], int(p["octave"]))
    n = int(fs * p["dur"])
    t = np.arange(n) / fs
    arch = p["arch"]
    fconst = np.full(n, root)
    click = _click(n, fs, rng, p["click"])
    env = _amp_env(n, fs, p["shape"], p["attack"], p["tau"], p["curve"],
                   p["trem_hz"], p["trem_depth"])
    env = _release(env, fs, 18.0)

    if arch == "acid":
        fc = root * p["cut_end"] + (root * p["cut_start"] - root * p["cut_end"]) \
            * np.exp(-t / max(1e-3, p["cut_tau"]))
        osc = dsp.square(fconst, fs, duty=p["duty"]) if p["wave"] == "square" else dsp.saw(fconst, fs)
        y = _swept_lp(osc, fc, fs, q=p["q"])
        core = _norm(y * env, 1.0)
        core = _norm(fx.sat_adaa(core, p["drive"]), 1.0)

    elif arch == "reese":
        d = 2.0 ** (p["detune"] / 1200.0)
        s1 = dsp.saw(np.full(n, root * d), fs)
        s2 = dsp.saw(np.full(n, root / d), fs, phase0=0.5)
        y = dsp.lp(0.5 * (s1 + s2), root * p["cut_mult"], fs, order=2)
        core = _norm(y * env, 1.0)
        core = _norm(fx.sat_adaa(core, p["drive"]), 1.0)
        core = core + _norm(dsp.sine(fconst, fs) * env, p["sub"])

    elif arch == "lead":
        f = root * (1.0 + p["vib"] * np.sin(2.0 * np.pi * p["vib_hz"] * t))
        osc = dsp.square(f, fs, duty=p["duty"]) if p["wave"] == "square" else dsp.saw(f, fs)
        y = dsp.lp(osc, root * p["cut_mult"], fs, order=2)
        core = _norm(y * env, 1.0)
        core = _norm(fx.sat_adaa(core, p["drive"]), 1.0)

    else:  # fm
        pitch = 1.0 + (p["pitch_mult"] - 1.0) * np.exp(-t / max(1e-3, p["pitch_t"]))
        idx_env = p["idx0"] * np.exp(-t / max(1e-3, p["idx_tau"]))
        y = dsp.fm(root * pitch, p["fm_ratio"], idx_env, fs)
        y = dsp.lp(y, root * p["cut_mult"], fs, order=2)
        core = _norm(y * env, 1.0)
        core = _norm(fx.sat_adaa(core, p["drive"]), 1.0)

    core = _norm(dsp.lp(core, p["tilt_hz"], fs, order=2), 1.0)
    if p["h2"] > 0.01:
        core = core + p["h2"] * dsp.bp(core, 2.0 * root, fs, q=2.5)
    if p["h3"] > 0.01:
        core = core + p["h3"] * dsp.bp(core, 3.0 * root, fs, q=2.5)
    core = _norm(core, 1.0)
    if p["edge"] > 0.01:
        core = _norm(core + p["edge"] * dsp.hp(core, p["edge_hz"], fs, order=2), 1.0)

    layers = [Layer(core, gain=1.0, label="core")]
    if click.max() > 0:
        layers.append(Layer(click, gain=1.0, label="click"))
    buf = mix(layers, fs=fs, stereo=False)
    buf = dsp.hp(buf, 32.0, fs, order=2)
    # Deliver at 0.9 peak so render.master's soft_clip never engages (it shifts
    # the mean of a low-frequency one-shot and the audit reads that as DC).
    buf = dsp.dc_block(buf, fs, fc=6.0)
    buf = _norm(buf, 0.9)

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, note=p["note"], key=p["key"],
                            octave=int(p["octave"]), archetype=arch, shape=p["shape"],
                            loop=False, descriptor=f"{_WORD[arch]}_{p['note']}",
                            drive=round(float(p["drive"]), 3),
                            duty=round(float(p["duty"]), 3)))
