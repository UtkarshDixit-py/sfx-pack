"""Family: retro. 8-bit / chiptune game sounds.

Everything is synthesised from band-limited square and saw oscillators (plus a
5-term Fourier triangle, because core.dsp has no triangle primitive).  The
aliasing comes from one deliberate final stage:

    fx.bitcrush(x, bits=8, rate_div=2, mode="harsh")

which is the only family in the pack where sample-hold alias is the point.
Note sequencing is done note-by-note with the oscillator phase reset, which is
how a sound chip actually retriggers a channel.

Archetype -> marketing descriptor mapping (the descriptor vocabulary is
deliberately narrower than the archetype set):
    coin, jump, powerup, hit, gameover, select, blip
    laser                -> blip
    door                 -> select
    1up                  -> powerup
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "retro"
DESCRIPTORS = ["coin", "jump", "powerup", "hit", "gameover", "select", "blip"]
ARCHETYPES = ("coin", "jump", "powerup", "hit", "gameover",
              "select", "laser", "blip", "door", "1up")
DESCR_OF = {"laser": "blip", "door": "select", "1up": "powerup"}


def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    a = float(np.max(np.abs(x))) if np.size(x) else 0.0
    if a <= 1e-12:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) * (peak / a)


def _osc(f, fs: int, wave: str, duty: float) -> np.ndarray:
    if wave == "sq":
        return dsp.square(f, fs, duty=duty)
    if wave == "saw":
        return dsp.saw(f, fs)
    # band-limited triangle: odd harmonics 1, -1/9, +1/25 (no core triangle)
    return (dsp.sine(f, fs) - dsp.sine(3.0 * np.asarray(f), fs) / 9.0
            + dsp.sine(5.0 * np.asarray(f), fs) / 25.0)


def _note(f: float, dur: float, fs: int, duty: float = 0.5, wave: str = "sq",
          vib: float = 0.0, vol: float = 1.0, decay: float = 0.55,
          attack: float = 0.0015) -> np.ndarray:
    k = max(8, int(round(fs * dur)))
    t = np.arange(k) / fs
    f_arr = f * (1.0 + vib * np.sin(2.0 * np.pi * 7.0 * t))
    y = _osc(f_arr, fs, wave, duty)
    env = dsp.env_exp(k, fs, attack=attack, tau=max(dur * decay, 0.004))
    m = min(k, max(2, int(fs * 0.003)))
    env[-m:] *= np.linspace(1.0, 0.0, m)
    return _norm(y * env, vol)


def _seq(total_s: float, fs: int, specs: list[tuple]) -> np.ndarray:
    """specs: [(freq, frac_of_total, wave, duty, vol, vib), ...]"""
    n = int(fs * total_s)
    out = np.zeros(n)
    pos = 0
    for i, (f, frac, wave, duty, vol, vib) in enumerate(specs):
        k = (n - pos) if i == len(specs) - 1 else int(round(n * frac))
        k = max(0, min(k, n - pos))
        if k >= 8:
            out[pos:pos + k] = _note(f, k / fs, fs, duty, wave, vib, vol)
        pos += k
        if pos >= n:
            break
    return out


def _sweep_tone(n: int, fs: int, f0: float, f1: float, wave: str,
                duty: float, decay: float, vib: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fs
    f = dsp.sweep(n, f0, f1, "exp") * (1.0 + vib * np.sin(2.0 * np.pi * 7.0 * t))
    y = _osc(f, fs, wave, duty)
    env = dsp.env_exp(n, fs, attack=0.001, tau=max(n / fs * decay, 0.004))
    m = min(n, max(2, int(fs * 0.003)))
    env[-m:] *= np.linspace(1.0, 0.0, m)
    return _norm(y * env)


# --------------------------------------------------------------------------
# parameter spread
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    arch = str(rng.choice(ARCHETYPES,
                          p=[0.13, 0.12, 0.11, 0.11, 0.10, 0.10,
                             0.10, 0.09, 0.07, 0.07]))
    p = dict(archetype=arch,
             shift=float(rng.uniform(-2.0, 2.0)),          # semitone transpose
             bits=int(rng.choice([6, 7, 8, 8, 8], p=[0.08, 0.17, 0.5, 0.15, 0.10])),
             rate_div=int(rng.choice([1, 2, 2, 2, 3], p=[0.15, 0.45, 0.2, 0.1, 0.1])),
             duty=float(rng.choice([0.125, 0.25, 0.5, 0.5, 0.75])),
             vib=float(rng.choice([0.0, 0.0, 0.004, 0.008, 0.014])))

    if arch == "coin":
        p.update(dur=float(rng.uniform(0.17, 0.32)),
                 first=str(rng.choice(["B5", "A5", "G5"])),
                 second=str(rng.choice(["E6", "F6", "D6"])),
                 split=float(rng.uniform(0.24, 0.38)))
    elif arch == "jump":
        p.update(dur=float(rng.uniform(0.14, 0.30)),
                 f0=float(rng.uniform(150.0, 260.0)),
                 f1=float(rng.uniform(550.0, 950.0)),
                 wave=str(rng.choice(["sq", "sq", "tri"])),
                 sub=float(rng.uniform(0.0, 0.4)))
    elif arch in ("powerup", "1up"):
        p.update(dur=float(rng.uniform(0.34, 0.78)),
                 base_midi=int(rng.integers(69, 79)),
                 steps=int(rng.integers(5, 9)),
                 duty=float(rng.choice([0.125, 0.25, 0.25, 0.5])),
                 fifth=float(rng.uniform(0.0, 0.35)))
    elif arch == "hit":
        p.update(dur=float(rng.uniform(0.12, 0.30)),
                 f0=float(rng.uniform(420.0, 950.0)),
                 f1=float(rng.uniform(80.0, 190.0)),
                 noise=str(rng.choice(["white", "white", "velvet"])))
    elif arch == "gameover":
        p.update(dur=float(rng.uniform(0.48, 0.80)),
                 base_midi=int(rng.integers(57, 65)),
                 minor=bool(rng.random() < 0.8),
                 timbre=str(rng.choice(["sq", "sq", "tri"])))
    elif arch == "select":
        p.update(dur=float(rng.uniform(0.05, 0.11)),
                 f0=float(rng.uniform(700.0, 1100.0)),
                 interval=int(rng.choice([5, 7, 12])),
                 second_frac=float(rng.uniform(0.45, 0.6)))
    elif arch == "laser":
        p.update(dur=float(rng.uniform(0.10, 0.26)),
                 f0=float(rng.uniform(1200.0, 2600.0)),
                 f1=float(rng.uniform(140.0, 340.0)),
                 wave=str(rng.choice(["sq", "sq", "saw"])),
                 decay=float(rng.uniform(0.45, 0.9)))
    elif arch == "blip":
        p.update(dur=float(rng.uniform(0.05, 0.10)),
                 f=float(rng.uniform(1100.0, 2600.0)),
                 duty=float(rng.choice([0.125, 0.25, 0.5])))
    else:  # door
        p.update(dur=float(rng.uniform(0.30, 0.62)),
                 f0=float(rng.uniform(65.0, 135.0)),
                 am=float(rng.uniform(18.0, 42.0)),
                 grit=float(rng.uniform(0.10, 0.35)),
                 sweep=float(rng.uniform(0.75, 1.0)))
    return p


# --------------------------------------------------------------------------
# archetypes
# --------------------------------------------------------------------------
def _coin(p, rng, fs, n):
    f1 = dsp.note_hz(p["first"][:-1], int(p["first"][-1])) * 2 ** (p["shift"] / 12)
    f2 = dsp.note_hz(p["second"][:-1], int(p["second"][-1])) * 2 ** (p["shift"] / 12)
    specs = [(f1, p["split"], "sq", 0.5, 1.0, 0.0),
             (f2, 1.0 - p["split"], "sq", 0.5, 1.0, 0.0)]
    return [Layer(_seq(p["dur"], fs, specs), 1.0, label="lead")]


def _jump(p, rng, fs, n):
    lead = _sweep_tone(n, fs, p["f0"], p["f1"], p["wave"], p["duty"], 0.55)
    layers = [Layer(_norm(lead, 0.95), 1.0, label="lead")]
    if p["sub"] > 0.02:
        t = np.arange(n) / fs
        f = dsp.sweep(n, p["f0"] * 0.5, p["f1"] * 0.5, "exp")
        sub = dsp.sine(f, fs) * dsp.env_exp(n, fs, attack=0.002,
                                            tau=max(n / fs * 0.5, 0.01))
        layers.append(Layer(_norm(sub, p["sub"]), 1.0, label="sub"))
    return layers


def _arp(p, rng, fs, n):
    base = dsp.midi_to_hz(p["base_midi"] + p["shift"])
    steps = int(p["steps"])
    # major-pentatonic run with an octave lift at the top (the classic "1up"
    # contour), not a blind chromatic scale
    pattern = [0, 4, 7, 11, 12, 16, 19, 24]
    freqs = [base * 2 ** (pattern[i % len(pattern)] / 12) for i in range(steps)]
    specs = []
    for i, f in enumerate(freqs):
        frac = 0.75 / steps if i < steps - 1 else 0.35
        specs.append((f, frac, "sq", p["duty"], 1.0, p["vib"]))
    lead = _seq(p["dur"], fs, specs)
    layers = [Layer(_norm(lead, 0.95), 1.0, label="lead")]
    if p["fifth"] > 0.02:
        specs2 = [(f * 1.5, frac, "sq", 0.5, 1.0, 0.0) for (f, frac, *_r) in specs]
        fifth = _seq(p["dur"], fs, specs2)
        layers.append(Layer(_norm(fifth, p["fifth"] * 0.5), 1.0, label="fifth"))
    return layers


def _hit(p, rng, fs, n):
    t = np.arange(n) / fs
    tone = _sweep_tone(n, fs, p["f0"], p["f1"], "sq", 0.5, 0.35)
    if p["noise"] == "velvet":
        nz = dsp.velvet(n, period=max(4, int(fs / 2000.0)), rng=rng)
        nz = dsp.bp(nz, 2600.0, fs, order=2, q=1.6)
    else:
        nz = dsp.hp(dsp.white(n, rng), 900.0, fs, order=2)
    nz *= dsp.env_exp(n, fs, attack=0.0005, tau=0.018)
    layers = [Layer(_norm(tone, 0.85), 1.0, label="tone"),
              Layer(_norm(nz, 0.55), 1.0, label="noise")]
    # a small sub tail so the hit lands on tiny speakers too
    k = min(n, int(fs * 0.10))
    sub = dsp.sine(dsp.sweep(k, 120.0, 55.0, "exp"), fs)
    sub *= dsp.env_exp(k, fs, attack=0.001, tau=0.035)
    layers.append(Layer(_norm(sub, 0.35), 1.0, label="sub"))
    return layers


def _gameover(p, rng, fs, n):
    base = dsp.midi_to_hz(p["base_midi"] + p["shift"])
    # descending minor (or major) arpeggio, long last note
    intervals = [12, 7, 3, 0, -5, -12] if p["minor"] else [12, 7, 4, 0, -5, -12]
    k = 6
    specs = []
    for i, s in enumerate(intervals):
        f = base * 2 ** (s / 12)
        frac = 0.13 if i < k - 1 else 0.35
        specs.append((f, frac, p["timbre"], 0.5, 1.0, p["vib"] * (i / k)))
    return [Layer(_norm(_seq(p["dur"], fs, specs), 0.95), 1.0, label="lead")]


def _select(p, rng, fs, n):
    f0 = p["f0"] * 2 ** (p["shift"] / 12)
    f1 = f0 * 2 ** (p["interval"] / 12)
    specs = [(f0, p["second_frac"], "sq", p["duty"], 1.0, 0.0),
             (f1, 1.0 - p["second_frac"], "sq", p["duty"], 1.0, 0.0)]
    return [Layer(_seq(p["dur"], fs, specs), 1.0, label="lead")]


def _laser(p, rng, fs, n):
    lead = _sweep_tone(n, fs, p["f0"] * 2 ** (p["shift"] / 12),
                       p["f1"], p["wave"], p["duty"], p["decay"])
    k = min(n, int(fs * 0.04))
    click = dsp.hp(dsp.white(k, rng), 4000.0, fs, order=2)
    click *= dsp.env_exp(k, fs, attack=0.0003, tau=0.006)
    return [Layer(_norm(lead, 0.95), 1.0, label="lead"),
            Layer(_norm(click, 0.35), 1.0, label="click")]


def _blip(p, rng, fs, n):
    f = p["f"] * 2 ** (p["shift"] / 12)
    return [Layer(_norm(_note(f, n / fs, fs, p["duty"], "sq", 0.0, 1.0,
                              decay=0.7), 0.95), 1.0, label="lead")]


def _door(p, rng, fs, n):
    t = np.arange(n) / fs
    f = dsp.sweep(n, p["f0"], p["f0"] * p["sweep"], "exp")
    # square for the edge, saw for the even harmonics a 50 % square cannot emit
    buzz = dsp.square(f, fs, duty=0.5) + 0.45 * dsp.saw(f, fs)
    am = 1.0 + 0.6 * np.sin(2.0 * np.pi * p["am"] * t)
    buzz *= dsp.env_exp(n, fs, attack=0.005, tau=max(n / fs * 0.6, 0.02)) * am
    layers = [Layer(_norm(buzz, 0.85), 1.0, label="buzz")]
    grit = dsp.bp(dsp.white(n, rng), 900.0, fs, order=2, q=1.2)
    grit *= dsp.env_exp(n, fs, attack=0.004, tau=max(n / fs * 0.35, 0.01))
    layers.append(Layer(_norm(grit, p["grit"]), 1.0, label="grit"))
    # faint wideband air: the bitcrushed harmonics of a low square are sparse and
    # leave real spectral valleys; a noise floor fills them without changing the
    # character (and a door motor does hiss)
    air = dsp.hp(dsp.white(n, rng), 90.0, fs, order=2)
    air *= dsp.env_exp(n, fs, attack=0.01, tau=max(n / fs * 0.6, 0.02))
    layers.append(Layer(_norm(air, 0.16), 1.0, label="air"))
    for j, frac in enumerate((0.0, 0.5)):
        k = min(n, int(fs * 0.03))
        c = dsp.hp(dsp.white(k, rng), 2500.0, fs, order=2)
        c *= dsp.env_exp(k, fs, attack=0.0004, tau=0.006)
        layers.append(Layer(_norm(c, 0.3), 1.0, offset_s=frac * p["dur"],
                            label=f"click{j}"))
    return layers


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    arch = p["archetype"]
    n = int(fs * p["dur"])

    if arch == "coin":
        layers = _coin(p, rng, fs, n)
    elif arch == "jump":
        layers = _jump(p, rng, fs, n)
    elif arch in ("powerup", "1up"):
        layers = _arp(p, rng, fs, n)
    elif arch == "hit":
        layers = _hit(p, rng, fs, n)
    elif arch == "gameover":
        layers = _gameover(p, rng, fs, n)
    elif arch == "select":
        layers = _select(p, rng, fs, n)
    elif arch == "laser":
        layers = _laser(p, rng, fs, n)
    elif arch == "blip":
        layers = _blip(p, rng, fs, n)
    else:
        layers = _door(p, rng, fs, n)

    buf = mix(layers, fs=fs, stereo=False)
    if buf.size > n:
        buf = buf[:n]
    elif buf.size < n:
        buf = np.pad(buf, (0, n - buf.size))

    # the one family where the aliasing is the product
    buf = fx.bitcrush(buf, bits=int(p["bits"]), rate_div=int(p["rate_div"]),
                      mode="harsh", fs=fs)

    buf = dsp.dc_block(buf, fs, fc=15.0)
    buf = dsp.hp(buf, 60.0, fs, order=2)
    buf = dsp.fade(buf, fs, ms=3.0)
    buf = _norm(buf, 0.95)
    buf = buf - float(np.mean(buf))     # quantisation can leave a residual offset
    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, archetype=arch, loop=False,
                            descriptor=DESCR_OF.get(arch, arch),
                            bits=int(p["bits"]), rate_div=int(p["rate_div"])))
