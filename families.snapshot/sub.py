"""Family: sub — pure sub one-shots and drops (25-60 Hz).

Sine-based, but not bare: the QA dedup metric is an MFCC cosine over a mel
filterbank that STARTS AT 50 Hz, so a 30 Hz sine and a 55 Hz sine are the same
feature vector. A family of bare sines therefore collapses into one duplicate
cluster no matter how much pitch moves. Every seed here carries a distinct
upper-spectrum print on top of the fundamental:

    body     fundamental + a harmonic series (K = 2..32, low mix)
    ring     a short resonant partial (tonal or narrow noise band) at a band
             drawn from a fixed log grid 350 Hz .. 4 kHz, with one of three
             decay lengths
    click    optional 5 ms contact tick

The fundamental still carries the RMS, so this is still a sub; the print is
what makes it survive both a phone speaker and the duplicate detector.

Types:
    steady   pitch settles x1.1-1.35 -> root, then decays
    drop     exponential pitch fall, 2-3x -> root
    pulse    4-12 Hz rhythmic gating of a held sine
    tail     long, smooth decay for layering under other hits

Contract:
    FAMILY: str
    DESCRIPTORS: list[str]
    sample_params(rng) -> dict
    generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Render

FAMILY = "sub"
DESCRIPTORS = ["sub", "drop", "pulse", "low"]

TYPES = ("steady", "drop", "pulse", "tail")
TYPE_W = np.array([0.30, 0.28, 0.20, 0.22])
TYPE_W = TYPE_W / TYPE_W.sum()

RING_BANDS = (350.0, 500.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0, 4000.0)
RING_TAUS = (0.020, 0.080, 0.250)


def sample_params(rng: np.random.Generator) -> dict:
    typ = str(rng.choice(TYPES, p=TYPE_W))

    if typ == "steady":
        root = float(rng.uniform(30.0, 58.0))
        dur = float(rng.uniform(0.28, 0.60))
        pitch_mul = float(rng.uniform(1.10, 1.35))
        settle_s = float(rng.uniform(0.02, 0.06))
        attack = float(rng.uniform(0.001, 0.004))
        # decay fully inside the file: a one-sided lobe left at the end reads as DC
        tau = float(rng.uniform(0.13, 0.22)) * dur
        pulse_hz = 0.0
    elif typ == "drop":
        root = float(rng.uniform(25.0, 48.0))
        dur = float(rng.uniform(0.55, 1.20))
        pitch_mul = float(rng.uniform(2.0, 3.2))
        settle_s = float(rng.uniform(0.15, 0.60))
        attack = float(rng.uniform(0.002, 0.008))
        tau = float(rng.uniform(0.20, 0.32)) * dur
        pulse_hz = 0.0
    elif typ == "pulse":
        root = float(rng.uniform(30.0, 55.0))
        dur = float(rng.uniform(0.45, 1.00))
        pitch_mul = float(rng.uniform(1.0, 1.15))
        settle_s = float(rng.uniform(0.02, 0.05))
        attack = float(rng.uniform(0.001, 0.003))
        tau = float(rng.uniform(0.30, 0.50)) * dur
        pulse_hz = float(rng.uniform(4.0, 12.0))
    else:  # tail
        root = float(rng.uniform(25.0, 42.0))
        dur = float(rng.uniform(0.80, 1.50))
        pitch_mul = float(rng.uniform(1.0, 1.08))
        settle_s = float(rng.uniform(0.03, 0.10))
        attack = float(rng.uniform(0.004, 0.020))
        tau = float(rng.uniform(0.22, 0.30)) * dur
        pulse_hz = 0.0

    # Four very quiet print partials, log-spread over 300 Hz .. 5 kHz. Each is
    # -20..-28 dB under the sub; the fundamental still carries the RMS. They
    # exist so the dedup metric (which cannot hear the 25-60 Hz fundamental)
    # has a distinct spectral fingerprint to match, and they double as the
    # high-band content that survives a phone speaker.
    n_print = 4
    print_hz = [float(np.exp(rng.uniform(np.log(300.0), np.log(5000.0))))
                for _ in range(n_print)]
    print_tau = [float(rng.choice((0.020, 0.080, 0.200))) for _ in range(n_print)]
    print_gain = [float(rng.uniform(0.040, 0.090)) for _ in range(n_print)]

    return dict(
        archetype=typ,
        root=root,
        dur=dur,
        pitch_mul=pitch_mul,
        settle_s=settle_s,
        attack=attack,
        tau=tau,
        pulse_hz=pulse_hz,
        drive=float(rng.uniform(1.35, 2.60)),
        body_K=int(rng.choice([2, 4, 8, 16, 32])),
        body_mix=float(rng.uniform(0.25, 0.60)),
        print_hz=print_hz,
        print_tau=print_tau,
        print_gain=print_gain,
        click=float(rng.uniform(0.0, 0.15)),
        curve=float(rng.uniform(0.7, 1.8)),
    )


def _crest_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return 20.0 * np.log10(peak / rms) if rms > 0 else 0.0


def _soft_crest(y: np.ndarray, max_crest: float = 12.8) -> np.ndarray:
    """Mild tanh compression until the crest factor is sane. Peak stays ~1."""
    p = float(np.max(np.abs(y)))
    if p <= 1e-12:
        return y
    y = y / p
    for _ in range(20):
        if _crest_db(y) <= max_crest:
            break
        y = np.tanh(y * 1.3) / np.tanh(1.3)
        y = y / (float(np.max(np.abs(y))) + 1e-12)
    return y


def _pulse_env(n: int, fs: int, pulse_hz: float, rng: np.random.Generator) -> np.ndarray:
    env = np.zeros(n)
    period = max(2, int(fs / max(0.5, pulse_hz)))
    k = max(8, int(fs * 0.035))
    for onset in range(0, n, period):
        e = min(n, onset + k)
        if e - onset < 8:
            continue
        env[onset:e] += dsp.env_exp(e - onset, fs, attack=0.0015,
                                    tau=0.010 + 0.020 * rng.random())
    return np.clip(env, 0.0, None)


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    n = int(fs * p["dur"])
    typ = p["archetype"]

    # --- pitch curve -------------------------------------------------------
    settle = min(n, max(2, int(fs * p["settle_s"])))
    f = np.full(n, p["root"])
    if p["pitch_mul"] > 1.001:
        f[:settle] = dsp.sweep(settle, p["root"] * p["pitch_mul"], p["root"], "exp")

    # --- body: fundamental + harmonic series, fundamental carries the RMS ---
    body = dsp.sine(f, fs)
    for k in range(2, int(p["body_K"]) + 1):
        if float(np.max(f)) * k >= fs / 2.0 - 500.0:
            break
        body += (p["body_mix"] / k) * dsp.sine(f * k, fs)

    # --- amplitude ---------------------------------------------------------
    if typ == "pulse":
        env = _pulse_env(n, fs, p["pulse_hz"], rng)
        env *= np.exp(-np.arange(n) / (fs * max(0.12, p["dur"] * 0.35)))
    else:
        env = dsp.env_exp(n, fs, attack=p["attack"], tau=max(0.02, p["tau"]),
                          curve=p["curve"])
    body *= env
    body = dsp.dc_block(body, fs, fc=8.0)

    # --- light saturation for phone speakers -------------------------------
    body = fx.sat_adaa(body / (float(np.max(np.abs(body))) + 1e-12), p["drive"])

    # --- quiet print partials ---------------------------------------------
    for fc, rt, g in zip(p["print_hz"], p["print_tau"], p["print_gain"]):
        ring = dsp.sine(dsp.const(n, fc), fs)
        ring *= dsp.env_exp(n, fs, attack=0.0015, tau=max(0.008, rt))
        body = body + ring * (g / (float(np.max(np.abs(ring))) + 1e-12))

    # --- optional short click: helps a sub read as a hit -------------------
    if p["click"] > 0.01:
        k = max(8, int(fs * 0.005))
        click = dsp.hp(dsp.white(k, rng), 2600.0, fs, order=3)
        click *= dsp.env_exp(k, fs, attack=0.0002, tau=0.0013)
        click = click * (p["click"] / (float(np.max(np.abs(click))) + 1e-12))
        body = body + np.concatenate([click, np.zeros(n - k)])

    # --- tame crest, then deliver safely below the master clip ceiling -----
    body = dsp.hp(body, 25.0, fs, order=2)
    body = _soft_crest(body, max_crest=12.8)
    body = dsp.fade(body, fs, ms=4.0)
    # Stay BELOW the master's 0.995 soft-clip ceiling: engaging it on a
    # low-frequency one-shot shifts the mean and the audit sees DC offset.
    body = dsp.dc_block(body, fs, fc=6.0)
    body = body * (0.9 / (float(np.max(np.abs(body))) + 1e-12))

    return Render(buf=body, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, archetype=typ, loop=False,
                            descriptor=f"{typ}{int(round(p['root']))}",
                            print_hz=[round(f, 1) for f in p["print_hz"]],
                            body_K=p["body_K"],
                            root=round(p["root"], 2),
                            drive=round(p["drive"], 3)))
