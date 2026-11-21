"""Family: laser. REFERENCE IMPLEMENTATION — copy this structure for new families.

A sellable laser is never one oscillator. Every render here is 3-6 layers:
  transient  short noise click, defines the attack
  body       pitch-swept band-limited square/saw (polyBLEP, no aliasing)
  sub        sine an octave down, gives weight on small speakers
  fm         inharmonic FM pair, the sci-fi "shine"
  air        highpassed noise that decays faster than the body
  space      short synthesized IR, mono-compatible

Contract every family must implement:
    FAMILY: str
    DESCRIPTORS: list[str]            short UCS-safe descriptors
    sample_params(rng) -> dict        the macro-parameter grid for one candidate
    generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "laser"
DESCRIPTORS = ["zap", "beam", "blaster", "charge", "warp"]

ARCHETYPES = ("zap", "beam", "blaster", "charge", "warp")


def sample_params(rng: np.random.Generator) -> dict:
    """Design the spread, do not sample blindly: every archetype gets its own range."""
    arch = str(rng.choice(ARCHETYPES, p=[0.32, 0.24, 0.22, 0.12, 0.10]))
    if arch == "zap":
        dur, f0, drop = rng.uniform(0.16, 0.45), rng.uniform(1100, 3200), rng.uniform(0.12, 0.30)
    elif arch == "beam":
        dur, f0, drop = rng.uniform(0.30, 0.90), rng.uniform(700, 1800), rng.uniform(0.80, 0.98)
    elif arch == "blaster":
        dur, f0, drop = rng.uniform(0.22, 0.55), rng.uniform(220, 640), rng.uniform(0.20, 0.45)
    elif arch == "charge":
        dur, f0, drop = rng.uniform(0.55, 1.10), rng.uniform(300, 900), rng.uniform(2.2, 4.0)
    else:  # warp
        dur, f0, drop = rng.uniform(0.40, 1.20), rng.uniform(400, 1400), rng.uniform(0.06, 0.16)
    return dict(
        archetype=arch,
        dur=float(dur),
        f0=float(f0),
        drop=float(drop),
        duty=float(rng.uniform(0.18, 0.62)),
        fm_ratio=float(rng.choice([1.41, 2.37, 3.51, 4.73, 7.11])),
        fm_index=float(rng.uniform(1.0, 5.5)),
        sub_gain=float(rng.uniform(0.10, 0.45)),
        noise=float(rng.uniform(0.05, 0.30)),
        dirt=float(rng.uniform(0.0, 1.0)),
        space=float(rng.uniform(0.05, 0.35)),
        space_t60=float(rng.uniform(0.25, 1.1)),
    )


def _transient(n: int, fs: int, rng: np.random.Generator, amount: float) -> np.ndarray:
    if amount <= 0:
        return np.zeros(n)
    k = max(8, int(fs * 0.006))
    noise = dsp.hp(dsp.white(k, rng), 3000.0, fs, order=3)
    out = np.zeros(n)
    out[:k] = noise * dsp.env_exp(k, fs, attack=0.0003, tau=0.0012) * amount
    return out


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    n = int(fs * p["dur"])
    arch = p["archetype"]

    f_curve = dsp.sweep(n, p["f0"], p["f0"] * p["drop"], "exp")
    if arch == "charge":
        # rising sweep that lands on a short hit
        f_curve = dsp.sweep(n, p["f0"], p["f0"] * p["drop"], "exp")

    # --- body: band-limited square, pitch swept ---
    body = dsp.square(f_curve, fs, duty=p["duty"])
    body *= dsp.env_exp(n, fs, attack=0.0008, tau=p["dur"] * (0.30 if arch != "beam" else 0.75))
    body = dsp.lp(body, min(16000.0, p["f0"] * 8.0 + 2000.0), fs, order=3)
    # a duty-cycled square carries real DC; remove it at the source, per layer
    body = dsp.dc_block(body, fs, fc=10.0)

    # --- sub: an octave down, sine, short ---
    sub = dsp.sine(f_curve * 0.5, fs) * p["sub_gain"]
    sub *= dsp.env_exp(n, fs, attack=0.001, tau=p["dur"] * 0.35)

    # --- fm shine: inharmonic, decays twice as fast as the body ---
    idx_env = p["fm_index"] * np.exp(-np.arange(n) / (fs * p["dur"] * 0.12))
    fm = dsp.fm(f_curve, p["fm_ratio"], idx_env, fs) * 0.35
    fm *= dsp.env_exp(n, fs, attack=0.001, tau=p["dur"] * 0.16)

    # --- transient + air ---
    tr = _transient(n, fs, rng, p["noise"])
    air = dsp.hp(dsp.white(n, rng), max(4000.0, p["f0"] * 2.0), fs, order=3)
    air *= dsp.env_exp(n, fs, attack=0.001, tau=p["dur"] * 0.08) * p["noise"] * 1.6

    layers = [
        Layer(body, gain=0.9, label="body"),
        Layer(sub, gain=1.0, label="sub"),
        Layer(fm, gain=0.8, label="fm"),
        Layer(tr, gain=1.0, label="transient"),
        Layer(air, gain=0.7, label="air"),
    ]
    buf = mix(layers, fs=fs, stereo=False)

    # --- dirt: the same core, two characters ---
    if p["dirt"] > 0.55:
        buf = fx.tape(buf, fs=fs, drive=1.4 + 2.2 * p["dirt"],
                      hiss_db=-70.0 + 8.0 * p["dirt"], rng=rng)
    elif p["dirt"] > 0.25:
        buf = fx.sat_adaa(buf, 1.2 + 1.5 * p["dirt"])

    # --- space ---
    if p["space"] > 0.02:
        buf = fx.reverb(buf, fs=fs, t60=p["space_t60"], size=0.3, mix=p["space"], rng=rng)

    buf = dsp.hp(buf, 30.0, fs, order=2)
    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, archetype=arch, loop=False,
                            descriptor=f"{arch}{int(p['f0'])}",
                            fm_ratio=p["fm_ratio"], dirt=round(p["dirt"], 3)))
