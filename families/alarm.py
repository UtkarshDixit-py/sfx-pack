"""Family: alarm — sirens, klaxons, electronic pulses, warning tones.

Two design families, per the brief:
  (a) electronic alarm pulses: a tone gated at 1-6 Hz (pulse, two_tone, alert, warning)
  (b) siren sweeps:            continuous tone with a periodic pitch sweep (siren, klaxon)

Urgency is a parameter, not a vibe: it sets the fundamental (400-1500 Hz), the pulse
rate (1-6 Hz), the attack shape (1 ms .. 80 ms) and the harmonic spread (almost pure
sine .. strong odd/even partials). Tiers:
  low     400-620 Hz, 1.0-2.2 Hz, 30-80 ms attack, warm (little harmonic spread)
  medium  600-950 Hz, 2.0-3.8 Hz, 8-30 ms attack, bright
  high    950-1500 Hz, 3.5-6.0 Hz, 1-8 ms attack, harsh

Every cue is mono, 0.5-2.5 s, one-shot. The gate has raised-cosine edges so it never
clicks; the whole file is highpassed, DC-corrected, scaled under the soft-clip knee
and faded. A little air + room is added so the cues sit in a scene instead of in a
vacuum.

Contract:
    FAMILY, DESCRIPTORS, sample_params(rng), generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Render

FAMILY = "alarm"
DESCRIPTORS = ["alarm", "siren", "klaxon", "warning", "alert"]

ARCHETYPES = ("pulse", "two_tone", "siren", "klaxon", "warning", "alert")
_P_ARCH = (0.22, 0.18, 0.24, 0.12, 0.12, 0.12)

_DESCRIPTOR = {
    "pulse": "alarm",
    "two_tone": "alarm",
    "siren": "siren",
    "klaxon": "klaxon",
    "warning": "warning",
    "alert": "alert",
}

_URGENCY = ("low", "medium", "high")

# fundamental / pulse-rate / attack / harmonic-spread ranges per urgency tier
_TIER = {
    "low":    dict(f0=(400.0, 620.0),  rate=(1.0, 2.2), attack=(0.030, 0.080), spread=(0.10, 0.35)),
    "medium": dict(f0=(600.0, 950.0),  rate=(2.0, 3.8), attack=(0.008, 0.030), spread=(0.30, 0.60)),
    "high":   dict(f0=(950.0, 1500.0), rate=(3.5, 6.0), attack=(0.001, 0.008), spread=(0.50, 0.90)),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-12:
        return np.zeros_like(x)
    return x * (peak / m)


def _gate(n: int, fs: int, rate: float, duty: float, attack: float,
          release: float, phase: float = 0.0) -> np.ndarray:
    """Trapezoidal on/off gate with raised-cosine edges (never clicks)."""
    period = max(8, int(round(fs / max(rate, 0.1))))
    on = max(3, int(round(period * float(np.clip(duty, 0.05, 0.95)))))
    a = max(1, int(attack * fs))
    r = max(1, int(release * fs))
    a = min(a, on // 2)
    r = min(r, on // 2)
    env = np.zeros(n)
    for start in range(int(phase * period), n, period):
        seg = min(on, n - start)
        if seg < 2:
            continue
        e = np.ones(seg)
        # clamp ramps to the truncated segment so a buffer-ending pulse never
        # tries to apply a full-length attack/release ramp to a short slice
        aa = min(a, seg // 2)
        rr = min(r, seg // 2)
        if aa > 1:
            e[:aa] = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, aa))
        if rr > 1:
            e[seg - rr:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, rr))
        env[start:start + seg] = np.maximum(env[start:start + seg], e)
    return env


def _stack(farr: np.ndarray, fs: int, spread: float, odd_bias: float = 0.5) -> np.ndarray:
    """Sine harmonic stack: alias-free, and spread controls how 'harsh' it reads."""
    nyq = fs * 0.45
    w = dsp.sine(farr, fs)
    for k, g in ((2.0, spread * (1.0 - odd_bias)), (3.0, spread * (0.55 + 0.35 * odd_bias)),
                 (5.0, spread * 0.35 * odd_bias)):
        if g <= 0:
            continue
        fk = farr * k
        # partials that would cross Nyquist contribute nothing valid; skip them
        if float(np.max(fk)) >= nyq:
            continue
        w = w + g * dsp.sine(fk, fs)
    return w


def _finish(buf: np.ndarray, fs: int) -> np.ndarray:
    buf = dsp.hp(buf, 60.0, fs, order=2)
    buf = dsp.dc_block(buf, fs, fc=20.0)      # gated/scaled squares carry DC
    buf = _norm(buf, 0.94)
    buf = buf - float(np.mean(buf))
    return dsp.fade(buf, fs, ms=5.0)


def _air(n: int, fs: int, rng: np.random.Generator, amount: float,
         gate: np.ndarray | None = None) -> np.ndarray:
    a = dsp.bp(dsp.white(n, rng), 3200.0, fs, order=2, q=1.1)
    a = _norm(a, amount)
    if gate is not None:
        a = a * gate
    return a


# --------------------------------------------------------------------------
# parameter design
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    arch = str(rng.choice(ARCHETYPES, p=_P_ARCH))
    urg = str(rng.choice(_URGENCY, p=(0.30, 0.40, 0.30)))
    tier = _TIER[urg]

    f0 = float(rng.uniform(*tier["f0"]))
    rate = float(rng.uniform(*tier["rate"]))
    attack = float(rng.uniform(*tier["attack"]))
    spread = float(rng.uniform(*tier["spread"]))

    p: dict = dict(
        archetype=arch, urgency=urg, f0=f0, rate=rate, attack=attack, spread=spread,
        dur=float(rng.uniform(0.8, 2.4)),
        duty=float(rng.uniform(0.35, 0.65)),
        ratio=float(rng.uniform(1.20, 1.35)),
        release=float(rng.uniform(0.004, 0.030)),
        buzz=float(rng.uniform(8.0, 26.0)),
        am=float(rng.uniform(0.15, 0.45)),
        sweep=float(rng.uniform(0.4, 0.9)),
        air=float(rng.uniform(0.02, 0.07)),
        room=float(rng.uniform(0.02, 0.10)),
        room_t60=float(rng.uniform(0.30, 0.80)),
        odd=float(rng.uniform(0.3, 0.8)),
    )

    if arch == "siren":
        p.update(f0=float(np.clip(f0 * rng.uniform(0.95, 1.20), 380.0, 1600.0)),
                 ratio=float(rng.uniform(1.20, 1.75)),
                 sweep=float({"low": rng.uniform(0.35, 1.0),
                              "medium": rng.uniform(0.7, 1.6),
                              "high": rng.uniform(1.1, 2.6)}[urg]),
                 spread=float(spread * rng.uniform(0.70, 1.0)),
                 dur=float(rng.uniform(1.0, 2.5)))
    elif arch == "klaxon":
        p.update(f0=float(rng.uniform(380.0, 700.0)),
                 rate=float(rng.uniform(1.2, 3.0)),
                 duty=float(rng.uniform(0.55, 0.88)),
                 spread=float(np.clip(spread * 1.2 + 0.2, 0.0, 1.2)),
                 buz=float(rng.uniform(10.0, 30.0)),
                 am=float(rng.uniform(0.20, 0.50)),
                 dur=float(rng.uniform(0.8, 2.0)))
    elif arch == "warning":
        p.update(rate=float(rng.uniform(0.8, 1.8)),
                 duty=float(rng.uniform(0.50, 0.72)),
                 spread=float(rng.uniform(0.05, 0.30)),
                 attack=float(rng.uniform(0.020, 0.070)),
                 dur=float(rng.uniform(1.0, 2.5)))
    elif arch == "alert":
        p.update(rate=float(rng.uniform(3.0, 6.0)),
                 duty=float(rng.uniform(0.28, 0.45)),
                 f0=float(rng.uniform(900.0, 1500.0)),
                 attack=float(rng.uniform(0.001, 0.006)),
                 spread=float(rng.uniform(0.40, 0.85)),
                 dur=float(rng.uniform(0.6, 1.4)))
    elif arch == "two_tone":
        p.update(ratio=float(rng.uniform(1.15, 1.45)),
                 duty=float(rng.uniform(0.35, 0.70)),
                 dur=float(rng.uniform(1.0, 2.5)))
    return p


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def _gen_pulse(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    farr = np.full(n, p["f0"])
    tone = _stack(farr, fs, p["spread"], p["odd"])
    gate = _gate(n, fs, p["rate"], p["duty"], p["attack"], p["release"])
    return tone * gate + _air(n, fs, rng, p["air"], gate)


def _gen_two_tone(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    period = max(8, int(round(fs / max(p["rate"], 0.1))))
    idx = np.arange(n) // period
    farr = np.where(idx % 2 == 0, p["f0"], p["f0"] * p["ratio"]).astype(np.float64)
    tone = _stack(farr, fs, p["spread"], p["odd"])
    # switch pitch only while the gate is closed: keeps each burst clean
    gate = _gate(n, fs, p["rate"], p["duty"], p["attack"], p["release"],
                 phase=0.0)
    return tone * gate + _air(n, fs, rng, p["air"], gate)


def _gen_siren(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    t = np.arange(n) / fs
    tri = 0.5 - 0.5 * np.cos(2.0 * np.pi * p["sweep"] * t + float(rng.uniform(0.0, 2.0 * np.pi)))
    farr = p["f0"] * (1.0 + (p["ratio"] - 1.0) * tri)
    tone = _stack(farr, fs, p["spread"], p["odd"])
    env = dsp.env_adsr(n, fs, a=max(p["attack"], 0.004), d=0.06, s=0.92,
                       r=min(0.12, p["dur"] * 0.08))
    return tone * env + _air(n, fs, rng, p["air"] * 0.7) * env


def _gen_klaxon(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    t = np.arange(n) / fs
    farr = np.full(n, p["f0"])
    tone = _stack(farr, fs, p["spread"], odd_bias=0.9)
    # mechanical growl: deep AM at 10-30 Hz, gated into on/off bursts
    am = 1.0 - p["am"] * (0.5 + 0.5 * np.sin(2.0 * np.pi * p["buz"] * t + float(rng.uniform(0, 6.28))))
    tone = dsp.lp(tone * am, 8000.0, fs, order=2)
    gate = _gate(n, fs, p["rate"], p["duty"], p["attack"], p["release"])
    return tone * gate + _air(n, fs, rng, p["air"], gate)


def _gen_warning(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    # slow warning beacons often drift up as they repeat: phase-continuous rise
    t = np.arange(n) / fs
    rise = 1.0 + 0.06 * np.clip(t / max(p["dur"], 1e-3) - 0.4, 0.0, 1.0)
    farr = p["f0"] * rise
    tone = _stack(farr, fs, p["spread"], odd_bias=0.3)     # warm: mostly even partials
    gate = _gate(n, fs, p["rate"], p["duty"], p["attack"], p["release"])
    return tone * gate + _air(n, fs, rng, p["air"] * 0.6, gate)


def _gen_alert(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    farr = np.full(n, p["f0"])
    tone = _stack(farr, fs, p["spread"], odd_bias=0.85)
    gate = _gate(n, fs, p["rate"], p["duty"], p["attack"], p["release"])
    # a short silence before the end: the classic "beep-beep-beep ... repeat"
    hold = int(n * float(rng.uniform(0.10, 0.22)))
    ramp = min(int(fs * 0.005), hold)
    if ramp > 1 and n > hold + ramp:
        gate[n - hold - ramp:n - hold] *= np.linspace(1.0, 0.0, ramp)
    gate[-hold:] = 0.0
    return tone * gate + _air(n, fs, rng, p["air"], gate)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    arch = p["archetype"]
    n = max(int(fs * 0.5), int(fs * p["dur"]))

    if arch == "pulse":
        buf = _gen_pulse(p, rng, fs, n)
    elif arch == "two_tone":
        buf = _gen_two_tone(p, rng, fs, n)
    elif arch == "siren":
        buf = _gen_siren(p, rng, fs, n)
    elif arch == "klaxon":
        buf = _gen_klaxon(p, rng, fs, n)
    elif arch == "warning":
        buf = _gen_warning(p, rng, fs, n)
    elif arch == "alert":
        buf = _gen_alert(p, rng, fs, n)
    else:
        raise ValueError(f"unknown alarm archetype {arch!r}")

    if p["room"] > 0.02:
        buf = fx.reverb(buf, fs=fs, t60=p["room_t60"], size=0.25, mix=p["room"], rng=rng)
    buf = _finish(buf, fs)

    return Render(
        buf=buf, fs=fs, stereo=False,
        meta=dict(family=FAMILY, archetype=arch, urgency=p["urgency"], loop=False,
                  mono=True, descriptor=f"{_DESCRIPTOR[arch]}{int(round(p['f0']))}",
                  rate=round(p["rate"], 2), spread=round(p["spread"], 3),
                  dur=round(len(buf) / fs, 3)),
    )
