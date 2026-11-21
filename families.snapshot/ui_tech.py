"""Family: ui_tech — phones, notifications, buttons, toggles, scans, dial tones.

Brief: 30-300 ms, bright (phone/laptop speakers have no bass), ten named types:
  tap      single soft click-tone (1.6-3 kHz), 30-60 ms
  toggle   two micro-blips, rising = on / falling = off, 60-120 ms
  confirm  rising major-triad arpeggio, 120-200 ms
  error    two descending square-wave buzzes, 150-280 ms
  notify   inharmonic bell (1 partial set), 180-300 ms
  message  soft two-note "ding-dong", 100-180 ms
  unlock   rising sweep 800 -> ~2500 Hz, 130-220 ms
  scan     train of 6-9 short rising blips, 200-300 ms
  dial     DTMF dual tone (one column + one row frequency), 100-200 ms
  charge   rising sweep with a high shimmer partial, 180-300 ms

Design rules:
  * nothing below ~180 Hz survives the final highpass: UI cues live on tiny speakers
  * every note gets a 1-4 ms attack, so every onset is a transient
  * the first event is never quieter than the later ones (transient first)
  * mono, one-shot, peak-scaled under the master soft-clip knee, DC-corrected, faded

Contract:
    FAMILY, DESCRIPTORS, sample_params(rng), generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp
from core.dsp import FS
from core.render import Render

FAMILY = "ui_tech"
DESCRIPTORS = ["tap", "toggle", "notify", "alert", "scan", "charge", "unlock"]

TYPES = ("tap", "toggle", "confirm", "error", "notify", "message",
         "unlock", "scan", "dial", "charge")
_P_TYPE = (0.13, 0.11, 0.10, 0.10, 0.12, 0.09, 0.09, 0.09, 0.08, 0.09)

_DESCRIPTOR = {
    "tap": "tap",
    "toggle": "toggle",
    "confirm": "notify",
    "error": "alert",
    "notify": "notify",
    "message": "notify",
    "unlock": "unlock",
    "scan": "scan",
    "dial": "notify",     # a signalling tone; closest word in the brief's descriptor set
    "charge": "charge",
}

_DUR = {
    "tap": (0.030, 0.060),
    "toggle": (0.060, 0.120),
    "confirm": (0.120, 0.200),
    "error": (0.150, 0.280),
    "notify": (0.180, 0.300),
    "message": (0.100, 0.180),
    "unlock": (0.130, 0.220),
    "scan": (0.200, 0.300),
    "dial": (0.100, 0.200),
    "charge": (0.180, 0.300),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-12:
        return np.zeros_like(x)
    return x * (peak / m)


def _click(n: int, fs: int, rng: np.random.Generator, peak: float = 0.5,
           hp_hz: float = 4000.0, ms: float = 1.6) -> np.ndarray:
    k = int(np.clip(fs * ms / 1000.0, 6, n))
    out = np.zeros(n)
    nz = dsp.hp(dsp.white(k, rng), hp_hz, fs, order=2)
    out[:k] = _norm(nz * dsp.env_exp(k, fs, attack=0.0001, tau=0.0005), 1.0)
    return out * float(peak)


def _note(buf: np.ndarray, fs: int, off_s: float, freq: float, dur_s: float,
          amp: float, attack: float = 0.0015, tau: float | None = None,
          kind: str = "sine", duty: float = 0.5, f_end: float | None = None,
          harm: float = 0.0, phase0: float = 0.0) -> None:
    """Add one normalised note into buf at off_s. Keeps every onset a transient."""
    off = int(round(off_s * fs))
    if off >= len(buf):
        return
    n = min(len(buf) - off, int(round(dur_s * fs)))
    if n <= 2:
        return
    if f_end is None or abs(f_end - freq) < 1.0:
        farr = np.full(n, float(freq))
    else:
        farr = dsp.sweep(n, float(freq), float(f_end), "exp")
    env = dsp.env_exp(n, fs, attack=attack, tau=float(tau if tau is not None else dur_s / 3.0))
    if kind == "square":
        w = dsp.square(farr, fs, duty=duty)
    elif kind == "saw":
        w = dsp.saw(farr, fs)
    else:
        w = dsp.sine(farr, fs, phase0=phase0)
    if harm > 0:
        w = w + float(harm) * dsp.sine(farr * 2.0, fs, phase0=phase0)
    buf[off:off + n] += _norm(w * env, amp)


# --------------------------------------------------------------------------
# parameter design
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    typ = str(rng.choice(TYPES, p=_P_TYPE))
    dur = float(rng.uniform(*_DUR[typ]))
    p: dict = dict(
        type=typ, dur=dur,
        hp=float(rng.uniform(180.0, 260.0)),
        colour=float(rng.uniform(0.0, 1.0)),
    )

    if typ == "tap":
        p.update(f0=float(rng.uniform(1600.0, 3000.0)), drop=float(rng.uniform(0.72, 0.95)),
                 click=float(rng.uniform(0.30, 0.60)))
    elif typ == "toggle":
        p.update(f0=float(rng.uniform(900.0, 2200.0)), ratio=float(rng.uniform(1.18, 1.36)),
                 rising=bool(rng.random() < 0.5), gap=float(rng.uniform(0.030, 0.050)))
    elif typ == "confirm":
        root = float(rng.choice([740.0, 880.0, 987.8, 1046.5, 1174.7, 1318.5]))
        p.update(root=root, notes=int(rng.integers(2, 4)),
                 step=float(rng.choice([1.2599, 1.3348, 1.4983])))   # maj3 / 4th / 5th
    elif typ == "error":
        p.update(f0=float(rng.uniform(420.0, 800.0)), drop=float(rng.uniform(0.62, 0.90)),
                 duty=float(rng.uniform(0.28, 0.58)), gap=float(rng.uniform(0.014, 0.042)))
    elif typ == "notify":
        p.update(f0=float(rng.uniform(800.0, 1800.0)), ratio_set=int(rng.integers(0, 3)),
                 second=bool(rng.random() < 0.55))
    elif typ == "message":
        p.update(f0=float(rng.uniform(850.0, 1200.0)), ratio=float(rng.uniform(0.66, 0.80)))
    elif typ == "unlock":
        p.update(f0=float(rng.uniform(700.0, 1000.0)), ratio=float(rng.uniform(2.2, 3.2)),
                 harm=float(rng.uniform(0.10, 0.26)))
    elif typ == "scan":
        p.update(blips=int(rng.integers(6, 10)), f0=float(rng.uniform(1100.0, 1500.0)),
                 ratio=float(rng.uniform(1.7, 2.2)), gap=float(rng.uniform(0.024, 0.034)),
                 rising=bool(rng.random() < 0.75))
    elif typ == "dial":
        p.update(low=float(rng.choice([697.0, 770.0, 852.0, 941.0])),
                 high=float(rng.choice([1209.0, 1336.0, 1477.0, 1633.0])),
                 click=float(rng.uniform(0.20, 0.40)))
    else:                                              # charge
        p.update(f0=float(rng.uniform(400.0, 700.0)), ratio=float(rng.uniform(2.8, 4.2)),
                 shimmer=float(rng.uniform(0.14, 0.34)))
    return p


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def _gen_tap(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    _note(buf, fs, 0.0, p["f0"], p["dur"] * 0.85, 0.85, attack=0.0008,
          tau=p["dur"] * 0.30, f_end=p["f0"] * p["drop"], harm=0.15)
    buf += _click(n, fs, rng, peak=p["click"], hp_hz=4500.0, ms=1.6)
    return buf


def _gen_toggle(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    second = p["f0"] * p["ratio"] if p["rising"] else p["f0"] / p["ratio"]
    _note(buf, fs, 0.0, p["f0"], p["dur"] * 0.45, 0.85, attack=0.0010,
          tau=p["dur"] * 0.22, harm=0.18)
    _note(buf, fs, p["gap"], second, p["dur"] * 0.45, 0.70, attack=0.0010,
          tau=p["dur"] * 0.22, harm=0.18)
    buf += _click(n, fs, rng, peak=0.35, hp_hz=4500.0, ms=1.4)
    return buf


def _gen_confirm(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    step = p["dur"] / max(1, p["notes"])
    for i in range(p["notes"]):
        _note(buf, fs, i * step * 0.85, p["root"] * (p["step"] ** i),
              p["dur"] - i * step * 0.85 + 0.005, 0.85 - 0.10 * i,
              attack=0.0018, tau=p["dur"] * 0.35, harm=0.20)
    return buf


def _gen_error(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    seg = (p["dur"] - p["gap"]) * 0.5
    _note(buf, fs, 0.0, p["f0"], seg, 0.80, attack=0.0012,
          tau=seg * 0.7, kind="square", duty=p["duty"], f_end=p["f0"] * p["drop"])
    _note(buf, fs, seg + p["gap"], p["f0"] * 0.90, seg, 0.68, attack=0.0012,
          tau=seg * 0.7, kind="square", duty=p["duty"], f_end=p["f0"] * p["drop"] * 0.90)
    buf += _click(n, fs, rng, peak=0.30, hp_hz=3500.0, ms=1.4)
    return buf


def _bell(buf: np.ndarray, fs: int, rng: np.random.Generator, off_s: float,
          f0: float, dur_s: float, amp: float, ratios: tuple[float, ...]) -> None:
    off = int(round(off_s * fs))
    if off >= len(buf):
        return
    n = min(len(buf) - off, int(round(dur_s * fs)))
    if n <= 2:
        return
    seg = np.zeros(n)
    for i, r in enumerate(ratios):
        f = f0 * r
        if f >= fs * 0.44:
            continue
        env = dsp.env_exp(n, fs, attack=0.0010, tau=dur_s * (0.60 - 0.09 * i))
        seg += dsp.sine(np.full(n, f), fs, phase0=float(rng.random())) * (1.0 / (1.0 + 0.8 * i)) * env
    buf[off:off + n] += _norm(seg, amp)


_BELL_SETS = (
    (1.0, 2.41, 3.71, 5.21),
    (1.0, 2.76, 5.40, 8.93),
    (1.0, 2.05, 3.71, 5.89),
)


def _gen_notify(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    ratios = _BELL_SETS[p["ratio_set"]]
    _bell(buf, fs, rng, 0.0, p["f0"], p["dur"] * 0.95, 0.90, ratios)
    if p["second"]:
        _bell(buf, fs, rng, 0.055, p["f0"] * 1.4983, p["dur"] * 0.65, 0.45, ratios)
    buf += _click(n, fs, rng, peak=0.30, hp_hz=4200.0, ms=1.2)
    return buf


def _gen_message(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    _note(buf, fs, 0.0, p["f0"], p["dur"] * 0.55, 0.85, attack=0.0015,
          tau=p["dur"] * 0.28, harm=0.25)
    _note(buf, fs, p["dur"] * 0.45, p["f0"] * p["ratio"], p["dur"] * 0.55, 0.70,
          attack=0.0015, tau=p["dur"] * 0.30, harm=0.20)
    return buf


def _gen_unlock(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    _note(buf, fs, 0.0, p["f0"], p["dur"] * 0.98, 0.85, attack=0.0020,
          tau=p["dur"] * 1.4, f_end=p["f0"] * p["ratio"], harm=p["harm"])
    buf += _click(n, fs, rng, peak=0.25, hp_hz=4500.0, ms=1.3)
    return buf


def _gen_scan(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    k = p["blips"]
    dur = min(p["gap"] * 0.42, p["dur"] / max(1, k))
    for i in range(k):
        t = i / max(1, k - 1)
        f = p["f0"] * (p["ratio"] ** t) if p["rising"] else p["f0"] * (p["ratio"] ** (1.0 - t))
        _note(buf, fs, i * p["gap"], f, dur, 0.85 - 0.015 * i, attack=0.0009, tau=dur * 0.5)
    buf += _click(n, fs, rng, peak=0.20, hp_hz=5000.0, ms=1.0)
    return buf


def _gen_dial(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    m = max(3, int(p["dur"] * fs * 0.92))
    ramp = int(fs * 0.004)
    env = np.ones(m)
    ramp = min(ramp, m // 2)
    if ramp > 0:
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    low_f, high_f = float(p["low"]), float(p["high"])
    seg = (dsp.sine(np.full(m, low_f), fs)
           + dsp.sine(np.full(m, high_f), fs)
           + 0.16 * dsp.sine(np.full(m, low_f * 2.0), fs)
           + 0.12 * dsp.sine(np.full(m, high_f * 2.0), fs)) * env
    buf[:m] += _norm(seg, 0.85)
    buf += _click(n, fs, rng, peak=p["click"], hp_hz=3000.0, ms=1.4)
    return buf


def _gen_charge(p: dict, rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    buf = np.zeros(n)
    _note(buf, fs, 0.0, p["f0"], p["dur"] * 0.98, 0.80, attack=0.0022,
          tau=p["dur"] * 1.5, f_end=p["f0"] * p["ratio"], harm=0.20)
    _note(buf, fs, p["dur"] * 0.25, p["f0"] * 2.2, p["dur"] * 0.72, p["shimmer"],
          attack=0.0025, tau=p["dur"] * 0.9, f_end=p["f0"] * p["ratio"] * 2.0)
    buf += _click(n, fs, rng, peak=0.22, hp_hz=4500.0, ms=1.2)
    return buf


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    typ = p["type"]
    n = max(int(fs * 0.03), int(fs * p["dur"]))

    if typ == "tap":
        buf = _gen_tap(p, rng, fs, n)
    elif typ == "toggle":
        buf = _gen_toggle(p, rng, fs, n)
    elif typ == "confirm":
        buf = _gen_confirm(p, rng, fs, n)
    elif typ == "error":
        buf = _gen_error(p, rng, fs, n)
    elif typ == "notify":
        buf = _gen_notify(p, rng, fs, n)
    elif typ == "message":
        buf = _gen_message(p, rng, fs, n)
    elif typ == "unlock":
        buf = _gen_unlock(p, rng, fs, n)
    elif typ == "scan":
        buf = _gen_scan(p, rng, fs, n)
    elif typ == "dial":
        buf = _gen_dial(p, rng, fs, n)
    elif typ == "charge":
        buf = _gen_charge(p, rng, fs, n)
    else:
        raise ValueError(f"unknown ui_tech type {typ!r}")

    # bright chain: everything below ~200 Hz is useless on a phone speaker
    buf = dsp.hp(buf, float(p["hp"]), fs, order=2)
    buf = dsp.dc_block(buf, fs, fc=20.0)
    buf = _norm(buf, 0.94)
    buf = buf - float(np.mean(buf))
    buf = dsp.fade(buf, fs, ms=3.0)

    tag_f = p.get("f0", p.get("low", p.get("root", 1000.0)))
    tag = int(round(float(tag_f)))
    return Render(
        buf=buf, fs=fs, stereo=False,
        meta=dict(family=FAMILY, type=typ, loop=False, mono=True,
                  descriptor=f"{_DESCRIPTOR[typ]}{tag}",
                  colour=round(p["colour"], 3), dur=round(len(buf) / fs, 4)),
    )
