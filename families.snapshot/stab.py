"""Family: stab — pitched melodic stabs, plucks, bells and chords, one per note.

PITCHING: `sample_params` assigns the note from a round-robin cursor, so any 12
consecutive seeds cover C..B exactly once (60 seeds = five full passes).
`generate` resolves the note with `dsp.note_hz(note, octave)` and every voice is
synthesised at that frequency; nothing is pitch-shifted. Chords are built by
transposing within the render (root + third + fifth) and are named by root.

Archetypes (mono, 0.2-0.8 s, octave 4 = 262-494 Hz):
  pluck   dsp.karplus (extended Karplus-Strong), noise excitation
  stab    marimba-like stack: sine + 2f + 3f, fast decay (bright short hit)
  bell    dsp.fm with an inharmonic ratio, decaying modulation index
  chord   root + (minor/major) third + perfect fifth, bright stacked tones

A short highpassed click is added under every archetype for attack definition
(except where the excitation already is one). Every layer is normalised to an
explicit peak target. Final hp at 45 Hz; the mandatory fade is applied once by
core render.master.

CORE ISSUES: none. (dsp.karplus rounds its delay to an integer number of samples,
so pitch is quantised to fs/N - a few cents at octave 4. That is intrinsic to the
model, not a fault; documented rather than patched.)
"""
from __future__ import annotations

import numpy as np

from core import dsp
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "stab"
DESCRIPTORS = ["stab", "pluck", "chord", "bell"]

OCTAVE = 4                       # C4..B4 = 261.6..493.9 Hz, melodic stab register
_WORD = {"pluck": "pluck", "stab": "stab", "bell": "bell", "chord": "chord"}
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


def _next_note() -> str:
    global _NOTE_CURSOR
    note = dsp.NOTE_NAMES[_NOTE_CURSOR % 12]
    _NOTE_CURSOR += 1
    return note


def _click(n: int, fs: int, rng: np.random.Generator, amount: float) -> np.ndarray:
    if amount <= 0:
        return np.zeros(n)
    k = max(6, int(fs * 0.004))
    burst = dsp.hp(dsp.white(k, rng), 2500.0, fs, order=3)
    c = np.zeros(n)
    c[:k] = burst * np.exp(-np.arange(k) / (k * 0.30))
    return _norm(c, amount)


# --------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    note = _next_note()
    arch = str(rng.choice(["pluck", "stab", "chord", "bell"], p=[0.35, 0.30, 0.20, 0.15]))
    dur = float(np.clip(rng.uniform(0.20, 0.80), 0.20, 0.80))
    # register is a diversity axis: the same note recurs at octave 3, 4 or 5
    octave = int(rng.choice([3, 4, 4, 5]))

    p = dict(note=note, key=note, octave=octave, arch=arch, dur=dur,
             click=float(rng.uniform(0.06, 0.14)))
    if arch == "pluck":
        p.update(ks_decay=float(rng.uniform(0.988, 0.9997)),
                 ks_bright=float(rng.uniform(0.10, 0.90)),
                 click=float(rng.uniform(0.03, 0.12)))
    elif arch == "stab":
        p.update(tau=float(rng.uniform(0.04, 0.24)),
                 bright2=float(rng.uniform(0.18, 0.55)),
                 bright3=float(rng.uniform(0.0, 0.28)),
                 n_partials=int(rng.choice([2, 3])))
    elif arch == "bell":
        p.update(ratio=float(rng.choice([1.41, 2.0, 2.4, 2.76, 3.5, 4.0, 5.0])),
                 idx0=float(rng.uniform(1.0, 9.0)),
                 idx_tau=float(rng.uniform(0.015, 0.15)),
                 tau=float(rng.uniform(0.04, 0.28)),
                 click=float(rng.uniform(0.02, 0.10)))
    else:  # chord
        p.update(quality=str(rng.choice(["maj", "min"])),
                 tau=float(rng.uniform(0.06, 0.26)),
                 detune=float(rng.uniform(0.0, 10.0)),
                 strum=float(rng.uniform(0.0, 0.008)),
                 gain2=float(rng.uniform(0.15, 0.45)),
                 gain3=float(rng.uniform(0.0, 0.18)),
                 click=float(rng.uniform(0.02, 0.09)))
    # tone axis: a lowpass plus an optional highpassed edge give two same-note
    # takes distinct spectral envelopes (and therefore separate dedup clusters)
    p["tone_hz"] = float(rng.uniform(1500.0, 17000.0))
    p["edge"] = float(rng.uniform(0.0, 0.35))
    p["edge_hz"] = float(rng.uniform(700.0, 4000.0))
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

    if arch == "pluck":
        y = dsp.karplus(root, n, fs, decay=p["ks_decay"], bright=p["ks_bright"], rng=rng)
        y = dsp.dc_block(y, fs, fc=12.0)
        y = _norm(y, 1.0)
        y = _norm(y * _release(np.ones(n), fs, 25.0), 1.0)
        core = y

    elif arch == "stab":
        y = (dsp.sine(np.full(n, root), fs)
             + p["bright2"] * dsp.sine(np.full(n, 2.003 * root), fs)
             + p["bright3"] * dsp.sine(np.full(n, 3.010 * root), fs))
        env = _release(dsp.env_exp(n, fs, attack=0.0015, tau=p["tau"]), fs, 20.0)
        core = _norm(y * env, 1.0)

    elif arch == "bell":
        idx_env = p["idx0"] * np.exp(-t / max(1e-3, p["idx_tau"]))
        y = dsp.fm(np.full(n, root), p["ratio"], idx_env, fs)
        env = _release(dsp.env_exp(n, fs, attack=0.001, tau=p["tau"]), fs, 15.0)
        core = _norm(y * env, 1.0)

    else:  # chord: root + third + fifth, still named by the root
        third = 4 if p["quality"] == "maj" else 3
        notes = [root, root * 2.0 ** (third / 12.0), root * 2.0 ** (7.0 / 12.0)]
        out = np.zeros(n)
        for j, fr in enumerate(notes):
            fr = fr * 2.0 ** ((p["detune"] * j) / 1200.0)
            tone = (dsp.sine(np.full(n, fr), fs)
                    + 0.28 * dsp.sine(np.full(n, 2.0 * fr), fs)
                    + 0.08 * dsp.sine(np.full(n, 3.0 * fr), fs))
            env = _release(dsp.env_exp(n, fs, attack=0.0015,
                                       tau=p["tau"] * (1.0 - 0.07 * j)), fs, 20.0)
            seg = tone * env * (1.0 - 0.08 * j)
            off = int(j * p["strum"] * fs)
            if off <= 0:
                out += seg
            elif off < n:
                out[off:] += seg[: n - off]
        core = _norm(out, 1.0)

    core = _norm(dsp.lp(core, p["tone_hz"], fs, order=2), 1.0)
    if p["edge"] > 0.01:
        core = _norm(core + p["edge"] * dsp.hp(core, p["edge_hz"], fs, order=2), 1.0)

    buf = mix([Layer(core, gain=1.0, label="core"),
               Layer(_click(n, fs, rng, p["click"]), gain=1.0, label="click")],
              fs=fs, stereo=False)
    buf = dsp.hp(buf, 45.0, fs, order=2)
    # Deliver at 0.9 peak so render.master's soft_clip never engages (it shifts
    # the mean of an asymmetric low tone and the audit reads that as DC).
    buf = dsp.dc_block(buf, fs, fc=6.0)
    buf = _norm(buf, 0.9)

    meta = dict(family=FAMILY, note=p["note"], key=p["key"], octave=int(p["octave"]),
                archetype=arch, loop=False, descriptor=f"{_WORD[arch]}_{p['note']}")
    if arch == "chord":
        meta["quality"] = p["quality"]
    return Render(buf=buf, fs=fs, stereo=False, meta=meta)
