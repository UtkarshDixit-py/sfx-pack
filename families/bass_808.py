"""Family: bass_808 — pitched 808 sub bass, one render per chromatic note.

PITCHING: never pitch-shift one take. `sample_params` assigns the note from a
round-robin cursor, so any run of 12 consecutive seeds covers C..B exactly once
(and 60 seeds cover the set five times). `generate` resolves that note to Hz with
`dsp.note_hz(note, octave)` and synthesises the fundamental at that frequency.
All other macro-parameters come from the seed's rng.

DIVERSITY: the draws for a single note are spread across an explicit design grid
so two 808s at the same note differ in LENGTH AND ATTACK, not just loudness:
  length band   short 0.25-0.55 s / mid 0.65-1.10 s / long 1.20-1.80 s,
                scaled down for higher notes (low notes ring longer)
  attack        pitch settle 15-35 / 40-70 / 80-120 ms, depth 1.5-2.5x
  gesture       steady / glide-in from a musical interval / distorted
  tone          clean / punchy / growl / bright (drive 1.5-6.0, edge layer)
  click         hardness 800-6000 Hz highpass, 3-9 ms
  sub           optional sub-octave layer (adds low-band weight on ~40% of takes)
Because core's dedup is a min-over-blocks (envelope, MFCC mean/std, log-mel,
timbre scalars) comparison, varying length, attack, brightness and low end per
seed is what makes the set read as many sounds instead of 12.

Signal: pitch-enveloped sine core -> fx.sat_adaa drive -> optional highpassed
edge for phone-speaker presence; optional sub-octave; a short click transient.
Every layer has an explicit peak target. Delivered at 0.9 peak so render.master's
soft_clip never engages (that nonlinearity shifts the mean of a low-frequency
one-shot and the audit reports DC).

CORE ISSUES:
  (1) No time-varying resonant lowpass in core (only `moving_bp`, a bandpass);
      bass_synth works around it with a local RBJ biquad. Not used here.
  (2) Plain mean+std MFCC dedup (the old qa.dedup) cannot resolve a 33-62 Hz
      fundamental: the mel floor is 50 Hz and the STFT bin width is 47 Hz, so
      different sub-bass notes scored 0.99 similar. The current block-wise dedup
      fixes most of this, but the low-register envelope/timbre blocks still cost
      some uniqueness. See families/bass_808.md for measured numbers.
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "bass_808"
DESCRIPTORS = ["808", "glide", "distorted"]

OCTAVE = 1                       # C1..B1 = 32.7..61.7 Hz, proper 808 sub weight
_VARIANT_WORD = {"plain": "808", "glide": "808glide", "distorted": "808dist"}

# Round-robin note cursor: guarantees 12-note coverage in a 12-seed run.
# generate() is a pure function of params, so re-rendering stored params is exact.
_NOTE_CURSOR = 0


def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    if m <= 1e-12:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) / m * peak


def _release(w: np.ndarray, fs: int, ms: float = 15.0) -> np.ndarray:
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


def sample_params(rng: np.random.Generator) -> dict:
    note = _next_note()
    i = dsp.NOTE_NAMES.index(note)         # 0 = C (lowest), 11 = B

    # --- length band, scaled by note (low notes ring longer) ---------------
    band = str(rng.choice(["short", "mid", "long"], p=[0.32, 0.34, 0.34]))
    if band == "short":
        dur = rng.uniform(0.25, 0.55)
    elif band == "mid":
        dur = rng.uniform(0.65, 1.10)
    else:
        dur = rng.uniform(1.20, 1.80)
    dur = float(np.clip(dur * (1.0 - 0.05 * i), 0.25, 1.80))
    # decay is drawn in absolute seconds, decoupled from length, so the normalised
    # envelope shape varies from a fast pluck sitting in silence to a long ring
    tau = float(2.0 ** rng.uniform(np.log2(0.07), np.log2(1.60)))

    # --- attack character: pitch-envelope settle time + depth --------------
    attack_mode = str(rng.choice(["fast", "snap", "slow"], p=[0.40, 0.32, 0.28]))
    if attack_mode == "fast":
        settle = float(rng.uniform(0.015, 0.035))
    elif attack_mode == "snap":
        settle = float(rng.uniform(0.040, 0.070))
    else:
        settle = float(rng.uniform(0.080, 0.120))
    start_mult = float(rng.uniform(1.5, 2.5))

    # --- gesture: steady / glide-in / distorted ----------------------------
    variant = str(rng.choice(["plain", "plain", "glide", "distorted"]))
    if variant == "glide":
        glide_semi = float(rng.choice([-12.0, -7.0, -5.0, 5.0, 7.0, 12.0]))
        glide_t = float(rng.uniform(0.05, 0.18))
    else:
        glide_semi, glide_t = 0.0, 0.001

    # --- tone character: drive + optional highpassed edge for presence -----
    character = str(rng.choice(["clean", "punchy", "growl", "bright"],
                               p=[0.25, 0.28, 0.25, 0.22]))
    if character == "clean":
        drive, edge, click = float(rng.uniform(1.5, 2.4)), 0.0, float(rng.uniform(0.12, 0.20))
    elif character == "punchy":
        drive, edge, click = float(rng.uniform(2.4, 3.6)), 0.0, float(rng.uniform(0.28, 0.40))
    elif character == "growl":
        drive, edge, click = float(rng.uniform(3.8, 6.0)), 0.0, float(rng.uniform(0.10, 0.18))
    else:  # bright
        drive, edge, click = float(rng.uniform(1.8, 3.0)), float(rng.uniform(0.20, 0.45)), \
            float(rng.uniform(0.16, 0.26))
    if variant == "distorted":
        drive = float(max(drive, rng.uniform(4.5, 6.0)))

    sub_on = bool(rng.random() < 0.42)

    return dict(
        note=note, key=note, octave=OCTAVE, variant=variant, character=character,
        length_band=band, attack_mode=attack_mode,
        dur=dur, tau=tau, settle=settle, start_mult=start_mult,
        glide_semi=glide_semi, glide_t=float(glide_t), drive=float(drive),
        edge=float(edge), edge_hz=float(rng.uniform(600.0, 2600.0)),
        click=float(click), click_hp=float(rng.uniform(800.0, 6000.0)),
        click_ms=float(rng.uniform(0.003, 0.009)),
        sub=float(rng.uniform(0.20, 0.50)) if sub_on else 0.0,
        h2=float(rng.uniform(0.0, 0.40)), h3=float(rng.uniform(0.0, 0.25)),
        amp_attack=float(rng.uniform(0.001, 0.004)),
        curve=float(rng.uniform(0.4, 2.6)),
    )


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    root = dsp.note_hz(p["note"], int(p["octave"]))
    n = int(fs * p["dur"])
    t = np.arange(n) / fs

    # --- pitch envelope: settle from 1.5-2.5x to root, or a musical glide ----
    if p["glide_semi"] != 0.0:
        theta = max(1e-4, p["glide_t"] / 3.0)
        ratio = 2.0 ** (p["glide_semi"] / 12.0 * np.exp(-t / theta))
    else:
        ratio = 1.0 + (p["start_mult"] - 1.0) * np.exp(-t / max(1e-4, p["settle"] / 3.0))
    f = root * ratio
    core = dsp.sine(f, fs)
    if p["h2"] > 0.01:
        core = core + p["h2"] * dsp.sine(2.0 * f, fs)
    if p["h3"] > 0.01:
        core = core + p["h3"] * dsp.sine(3.0 * f, fs)

    env = dsp.env_exp(n, fs, attack=p["amp_attack"], tau=p["tau"], curve=p["curve"])
    env = _release(env, fs, ms=15.0)
    core = _norm(core * env, 1.0)
    core = _norm(fx.sat_adaa(core, p["drive"]), 1.0)
    if p["edge"] > 0.01:
        core = _norm(core + p["edge"] * dsp.hp(core, p["edge_hz"], fs, order=2), 1.0)

    # --- optional sub-octave weight ---------------------------------------
    sub = np.zeros(n)
    if p["sub"] > 0.0:
        sub = _norm(dsp.sine(np.full(n, root * 0.5), fs) * env, p["sub"])

    # --- onset click: hardness and length vary per seed -------------------
    k = max(6, int(fs * p["click_ms"]))
    click = np.zeros(n)
    burst = dsp.hp(dsp.white(k, rng), p["click_hp"], fs, order=3)
    click[:k] = burst * np.exp(-np.arange(k) / (k * 0.30))
    click = _norm(click, p["click"])

    buf = mix([Layer(core, gain=1.0, label="core"),
               Layer(sub, gain=1.0, label="sub"),
               Layer(click, gain=1.0, label="click")], fs=fs, stereo=False)
    buf = dsp.hp(buf, 25.0, fs, order=2)
    # Deliver at 0.9 peak so render.master's soft_clip(0.995) never engages: on a
    # low-frequency one-shot that nonlinearity shifts the mean and the audit
    # reports DC offset. A final dc_block removes any residual window mean.
    buf = dsp.dc_block(buf, fs, fc=6.0)
    buf = _norm(buf, 0.9)

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, note=p["note"], key=p["key"],
                            octave=int(p["octave"]), variant=p["variant"],
                            character=p["character"], length_band=p["length_band"],
                            attack_mode=p["attack_mode"], loop=False,
                            descriptor=f"{_VARIANT_WORD[p['variant']]}_{p['note']}",
                            drive=round(float(p["drive"]), 3),
                            tau=round(float(p["tau"]), 3)))
