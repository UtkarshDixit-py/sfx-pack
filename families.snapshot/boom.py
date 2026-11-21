"""Family: boom — cinematic sub-boom / explosion.

Three layers (Farnell), all synthesised, each explicitly peak-normalised:
    blast     saturated lowpassed noise burst, 5-20 ms attack
    body      sine sweep 120 -> 30 Hz over 150-400 ms (the weight)
    debris    sparse filtered noise grains + a low smoke bed for 0.3-1.5 s

Real sub energy lives below 60 Hz (body lands at 26-42 Hz); the debris bed is
deliberately broadband at a low peak so it fills the LTAS without masking the
hit. Bright debris UTF sits UNDER the boom, never over it.

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

FAMILY = "boom"
DESCRIPTORS = ["boom", "explosion", "sub", "impact", "cinematic"]

ARCHETYPES = ("boom", "explosion", "sub", "cinematic")
ARCH_W = np.array([0.34, 0.28, 0.20, 0.18])
ARCH_W = ARCH_W / ARCH_W.sum()


def sample_params(rng: np.random.Generator) -> dict:
    arch = str(rng.choice(ARCHETYPES, p=ARCH_W))

    if arch == "sub":                 # deeper, tighter, less debris
        dur = float(rng.uniform(0.9, 1.8))
        f_start, f_end = float(rng.uniform(90.0, 120.0)), float(rng.uniform(26.0, 34.0))
        debris_gain = float(rng.uniform(0.08, 0.18))
        space = float(rng.uniform(0.04, 0.14))
    elif arch == "explosion":         # longer, dirtier, more debris
        dur = float(rng.uniform(1.3, 2.5))
        f_start, f_end = float(rng.uniform(110.0, 155.0)), float(rng.uniform(28.0, 42.0))
        debris_gain = float(rng.uniform(0.24, 0.42))
        space = float(rng.uniform(0.10, 0.28))
    elif arch == "cinematic":         # slow attack, big room
        dur = float(rng.uniform(1.6, 2.5))
        f_start, f_end = float(rng.uniform(80.0, 120.0)), float(rng.uniform(25.0, 34.0))
        debris_gain = float(rng.uniform(0.10, 0.24))
        space = float(rng.uniform(0.14, 0.32))
    else:                             # boom: the default punchy one
        dur = float(rng.uniform(0.9, 1.9))
        f_start, f_end = float(rng.uniform(95.0, 140.0)), float(rng.uniform(28.0, 40.0))
        debris_gain = float(rng.uniform(0.16, 0.34))
        space = float(rng.uniform(0.06, 0.20))

    blast_attack = float(rng.uniform(0.004, 0.016) if arch != "cinematic"
                         else rng.uniform(0.012, 0.020))

    # a spectral character per seed; without this every boom is the same blob
    character = str(rng.choice(["deep", "cracky", "dirty", "cinematic", "wide"],
                               p=[0.22, 0.24, 0.18, 0.18, 0.18]))
    if character == "deep":
        blast_cut, smoke_hp, smoke_lp = rng.uniform(260, 700), rng.uniform(90, 250), rng.uniform(1200, 3500)
        air_gain, mix_lp = rng.uniform(0.0, 0.06), rng.choice([0.0, 700.0, 1100.0])
    elif character == "cracky":
        blast_cut, smoke_hp, smoke_lp = rng.uniform(2200, 6000), rng.uniform(500, 1400), rng.uniform(11000, 16000)
        air_gain, mix_lp = rng.uniform(0.10, 0.30), 0.0
    elif character == "dirty":
        blast_cut, smoke_hp, smoke_lp = rng.uniform(800, 2400), rng.uniform(200, 700), rng.uniform(5000, 10000)
        air_gain, mix_lp = rng.uniform(0.03, 0.14), 0.0
    elif character == "wide":
        blast_cut, smoke_hp, smoke_lp = rng.uniform(900, 2600), rng.uniform(150, 400), rng.uniform(8000, 14000)
        air_gain, mix_lp = rng.uniform(0.08, 0.22), 0.0
    else:                              # cinematic
        blast_cut, smoke_hp, smoke_lp = rng.uniform(500, 1600), rng.uniform(100, 300), rng.uniform(4000, 9000)
        air_gain, mix_lp = rng.uniform(0.02, 0.10), 0.0

    return dict(
        archetype=arch,
        character=character,
        dur=dur,
        f_start=f_start,
        f_end=f_end,
        sweep_s=float(rng.uniform(0.15, 0.40)),
        blast_cut=float(blast_cut),
        blast_attack=blast_attack,
        blast_tau=float(rng.uniform(0.08, 0.28)),
        body_gain=float(rng.uniform(0.75, 1.15)),
        body_tau=float(rng.uniform(0.35, 0.65) * dur),
        body_h2=float(rng.uniform(0.05, 0.45)),
        body_h3=float(rng.uniform(0.0, 0.28)),
        drive=float(rng.uniform(2.2, 5.0)),
        debris_density=float(rng.uniform(18.0, 70.0)),
        debris_gain=debris_gain,
        smoke_gain=float(rng.uniform(0.08, 0.30)),
        smoke_hp=float(smoke_hp),
        smoke_lp=float(smoke_lp),
        smoke_color=str(rng.choice(["brown", "pink", "white"], p=[0.3, 0.45, 0.25])),
        air_gain=float(air_gain),
        air_hz=float(rng.uniform(2200.0, 7000.0)),
        mix_lp=float(mix_lp),
        debris_offset=float(rng.uniform(0.03, 0.10)),
        space=space,
        space_t60=float(rng.uniform(0.5, 1.7)),
        dirt=float(rng.uniform(0.0, 1.0)),
    )


def _norm(y: np.ndarray, target: float) -> np.ndarray:
    p = float(np.max(np.abs(y)))
    return y * (target / p) if p > 1e-12 else y


def _blast(n: int, fs: int, p: dict, rng: np.random.Generator) -> np.ndarray:
    noise = dsp.white(n, rng)
    low = dsp.lp(noise, p["blast_cut"], fs, order=3)
    low *= dsp.env_exp(n, fs, attack=p["blast_attack"], tau=p["blast_tau"])
    # saturated blast body: drive adds the harmonics that make it read as loud
    low = fx.sat_adaa(_norm(low, 1.0), p["drive"])

    # sub-3 ms crack on top of the blast
    k = int(fs * 0.012)
    crack = dsp.hp(dsp.white(k, rng), 2500.0, fs, order=3)
    crack *= dsp.env_exp(k, fs, attack=0.0002, tau=0.0016)
    out = np.zeros(n)
    out[:k] = _norm(crack, 0.45 + 0.35 * p["dirt"])
    out += low
    return out


def _body(n: int, fs: int, p: dict) -> np.ndarray:
    n_sw = int(fs * p["sweep_s"])
    fc = np.full(n, p["f_end"])
    fc[:n_sw] = dsp.sweep(n_sw, p["f_start"], p["f_end"], "exp")
    tone = dsp.sine(fc, fs)
    tone += p["body_h2"] * dsp.sine(fc * 2.0, fs)      # octave: reads on phone speakers
    if p["body_h3"] > 0.01:
        tone += p["body_h3"] * dsp.sine(fc * 3.0, fs)
    tone *= dsp.env_exp(n, fs, attack=0.004, tau=p["body_tau"])
    tone = dsp.dc_block(tone, fs, fc=10.0)
    tone = fx.sat_adaa(_norm(tone, 1.0), 1.6)  # gentle, keep the sweep clean
    return _norm(tone, p["body_gain"])


def _debris(n: int, fs: int, p: dict, rng: np.random.Generator) -> np.ndarray:
    """Sparse crackle grains over a low broadband smoke bed. Both low peak."""
    m = max(64, int(fs * (p["dur"] - p["debris_offset"])))

    src = dsp.white(int(fs * 0.12), rng)
    src = dsp.bp(src, float(rng.uniform(1200.0, 4800.0)), fs, order=2, q=1.0)
    grains = dsp.granulate(src, m, p["debris_density"], fs=fs, rng=rng,
                           grain_ms=(5.0, 34.0), pitch=(0.6, 1.8),
                           position=(0.0, 1.0))
    t = np.arange(m) / fs
    grains *= np.exp(-t / max(0.10, p["dur"] * 0.28))
    grains = _norm(grains, p["debris_gain"])

    # smoke bed: coloured noise, smooth LTAS so the mix has no holes
    colour = {"brown": dsp.brown, "pink": dsp.pink, "white": dsp.white}[p["smoke_color"]]
    smoke = dsp.lp(colour(m, rng), p["smoke_lp"], fs, order=2)
    smoke = dsp.hp(smoke, p["smoke_hp"], fs, order=2)
    smoke *= dsp.env_exp(m, fs, attack=0.006, tau=max(0.15, p["dur"] * 0.42))
    smoke = _norm(smoke, p["smoke_gain"])

    return grains + smoke


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    n = int(fs * p["dur"])

    blast = _norm(_blast(n, fs, p, rng), 1.0)
    body = _body(n, fs, p)
    debris = _debris(n, fs, p, rng)

    layers = [
        Layer(blast, gain=1.0, label="blast"),
        Layer(body, gain=1.0, label="body"),
        Layer(debris, gain=1.0, offset_s=p["debris_offset"], label="debris"),
    ]
    buf = mix(layers, fs=fs, stereo=False)

    # air band: the open top end that makes a boom sit in a mix
    if p["air_gain"] > 0.015:
        L = len(buf)
        air = dsp.hp(dsp.white(L, rng), p["air_hz"], fs, order=3)
        air *= dsp.env_exp(L, fs, attack=0.003, tau=max(0.05, p["dur"] * 0.18))
        buf = buf + _norm(air, p["air_gain"])

    if p["dirt"] > 0.5:
        buf = fx.sat_adaa(buf, 1.3 + 1.4 * p["dirt"])
    if p["space"] > 0.03:
        buf = fx.reverb(buf, fs=fs, t60=p["space_t60"], size=0.55,
                        mix=p["space"], rng=rng)

    if p["mix_lp"] > 0.0:
        buf = dsp.lp(buf, p["mix_lp"], fs, order=2)

    buf = dsp.hp(buf, 24.0, fs, order=2)
    buf = dsp.fade(buf, fs, ms=4.0)
    # stay under the master soft-clip ceiling (nonlinear clip can add DC)
    buf = buf * (0.9 / (float(np.max(np.abs(buf))) + 1e-12))

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, archetype=p["archetype"], loop=False,
                            descriptor=f"{p['archetype']}_{p['character']}",
                            character=p["character"],
                            f_start=round(p["f_start"], 1),
                            f_end=round(p["f_end"], 1),
                            sweep_s=round(p["sweep_s"], 3)))
