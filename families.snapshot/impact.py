"""Family: impact — hard-surface collisions.

Modal synthesis, not noise bursts. The audible identity of a collision is the
ring of the object's normal modes; the contact burst only decides how hard and
how bright it is hit (O'Brien et al. 2002).

Layers (every one explicitly peak-normalised before the mix):
    click    0-4 ms contact transient, highpassed noise
    body     two-pole modal bank driven by dsp.contact_burst()
    sub      60-90 Hz thud, gives weight on small speakers
    air      short highpassed noise tail, softer than the body

Size scales the fundamental DOWN and the decay LONGER. Hardness sets the
contact burst bandwidth (soft thud -> sharp crack).

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
from core.render import Layer, Render, mix

FAMILY = "impact"
DESCRIPTORS = ["metal", "wood", "glass", "stone", "plastic", "crack"]

# material -> (f0 low, f0 high @ size 1, tau0 low, tau0 high, descriptor)
MAT_PROFILES = {
    "metal":    (300.0, 1400.0, 0.18, 0.50, "metal"),
    "glass":    (500.0, 1800.0, 0.12, 0.40, "glass"),
    "wood":     (110.0,  460.0, 0.06, 0.18, "wood"),
    "concrete": ( 65.0,  220.0, 0.05, 0.16, "stone"),
    "ceramic":  (280.0, 1100.0, 0.08, 0.26, "stone"),
    "plastic":  (150.0,  520.0, 0.025, 0.09, "plastic"),
}
MATERIALS = list(MAT_PROFILES)
MAT_WEIGHTS = np.array([0.20, 0.12, 0.22, 0.20, 0.12, 0.14])
MAT_WEIGHTS = MAT_WEIGHTS / MAT_WEIGHTS.sum()


def sample_params(rng: np.random.Generator) -> dict:
    material = str(rng.choice(MATERIALS, p=MAT_WEIGHTS))
    f_lo, f_hi, t_lo, t_hi = MAT_PROFILES[material][:4]

    f0 = float(rng.uniform(f_lo, f_hi))
    size = float(rng.uniform(0.55, 1.9))
    tau0 = float(rng.uniform(t_lo, t_hi))

    # bimodal hit: soft thuds and sharp cracks are different products
    hardness = float(rng.uniform(0.18, 0.48) if rng.random() < 0.42
                     else rng.uniform(0.62, 1.0))

    ring = tau0 * size
    dur = float(np.clip(ring * 3.6 + 0.07, 0.25, 1.2))

    # bigger object rings lower and longer; sub follows the size down
    sub_f = float(rng.uniform(60.0, 90.0) - 26.0 * (size - 0.55) / 1.35)
    sub_f = float(np.clip(sub_f, 50.0, 92.0))

    return dict(
        material=material,
        f0=f0,
        size=size,
        tau0=tau0,
        hardness=hardness,
        dur=dur,
        sub_f=sub_f,
        sub_gain=float(rng.uniform(0.22, 0.55)),
        air_gain=float(rng.uniform(0.10, 0.34)),
        air_hz=float(rng.uniform(3200.0, 7200.0)),
        bright=float(rng.uniform(3500.0, 15000.0)),
        jitter=float(rng.uniform(0.0, 0.012)),
        dirt=float(rng.uniform(0.0, 1.0)),
        space=float(rng.uniform(0.02, 0.22)),
        space_t60=float(rng.uniform(0.12, 0.55)),
    )


def _norm(y: np.ndarray, target: float) -> np.ndarray:
    p = float(np.max(np.abs(y)))
    return y * (target / p) if p > 1e-12 else y


def _click(n: int, fs: int, rng: np.random.Generator, hardness: float) -> np.ndarray:
    k = max(8, int(fs * 0.004))
    cut = 2500.0 + 9000.0 * hardness
    noise = dsp.hp(dsp.white(k, rng), cut, fs, order=3)
    out = np.zeros(n)
    out[:k] = noise * dsp.env_exp(k, fs, attack=0.00015, tau=0.00075 + 0.0006 * (1.0 - hardness))
    return out


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    n = int(fs * p["dur"])
    material = p["material"]

    # --- modal body: contact acceleration drives a 2-pole bank ---------------
    modes = dsp.modal_bank(p["f0"], material, tau0=p["tau0"], size=p["size"],
                           rng=rng, jitter=p["jitter"])
    exc = dsp.contact_burst(n, fs, hardness=p["hardness"], rng=rng)
    # contact_burst lets a soft hit smear for ~20 ms, which pushes the modal
    # peak past the 15 ms transient rule. A short extra gate keeps the contact
    # bright and impulse-like without changing its bandwidth (set by hardness).
    exc = exc * dsp.env_exp(n, fs, attack=0.0002,
                            tau=0.003 + 0.008 * (1.0 - p["hardness"]))
    body = dsp.modal(exc, modes, fs)
    body = dsp.lp(body, p["bright"], fs, order=3)
    body = _norm(body, 1.0)

    # --- click transient -----------------------------------------------------
    click = _norm(_click(n, fs, rng, p["hardness"]), 0.45 + 0.55 * p["hardness"])

    # --- sub thud: size-scaled, tiny pitch settle for punch ------------------
    settle = int(fs * 0.03)
    sub_f = np.full(n, p["sub_f"])
    sub_f[:settle] = dsp.sweep(settle, p["sub_f"] * 1.22, p["sub_f"], "exp")
    sub = dsp.sine(sub_f, fs) * dsp.env_exp(n, fs, attack=0.0012, tau=p["dur"] * 0.30)
    sub = dsp.dc_block(sub, fs, fc=12.0)
    sub = _norm(sub, p["sub_gain"])

    # --- air tail: highpassed noise, decays faster than the body -------------
    ring = p["tau0"] * p["size"]
    air = dsp.hp(dsp.white(n, rng), p["air_hz"], fs, order=3)
    air *= dsp.env_exp(n, fs, attack=0.0008, tau=max(0.012, ring * 0.42))
    air = _norm(air, p["air_gain"])

    layers = [
        Layer(click, gain=1.0, label="click"),
        Layer(body, gain=1.0, label="body"),
        Layer(sub, gain=1.0, label="sub"),
        Layer(air, gain=1.0, label="air"),
    ]
    buf = mix(layers, fs=fs, stereo=False)

    # --- character -----------------------------------------------------------
    if p["dirt"] > 0.6:
        buf = fx.sat_adaa(buf, 1.6 + 2.0 * p["dirt"])
    elif p["dirt"] > 0.3:
        buf = fx.sat_adaa(buf, 1.0 + 1.1 * p["dirt"])

    if p["space"] > 0.03:
        buf = fx.reverb(buf, fs=fs, t60=p["space_t60"], size=0.22,
                        mix=p["space"], rng=rng)

    # concrete/stone fundamentals can reach ~40 Hz; keep the hp below that
    hp_hz = 28.0 if material in ("concrete", "wood") else 38.0
    buf = dsp.hp(buf, hp_hz, fs, order=2)
    buf = dsp.fade(buf, fs, ms=3.0)
    # stay under the master soft-clip ceiling (nonlinear clip can add DC)
    buf = buf * (0.9 / (float(np.max(np.abs(buf))) + 1e-12))

    word = MAT_PROFILES[material][4]
    if p["hardness"] > 0.86 and material in ("glass", "ceramic", "concrete"):
        word = "crack"
    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, archetype=material, loop=False,
                            descriptor=f"{word}{int(round(p['size'] * 10)):02d}",
                            size=round(p["size"], 3),
                            hardness=round(p["hardness"], 3),
                            f0=round(p["f0"], 1)))
