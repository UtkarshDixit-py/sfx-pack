#!/usr/bin/env python3
"""Putt Duel — complete sound kit, one pass.

Event vocabulary (games/minigolf/src/logic/physics.ts + scenes/sound.ts):
    shot · wall · water · portal · break · star · holed · timeout
    hole win/lose/draw · match win/lose/draw · ui

Rules:
  * phone-first: main energy 400 Hz - 6 kHz, sub never load-bearing
  * every sound has a real <5 ms transient
  * explicit per-layer peak targets, never emergent mixes
  * stings are C major pentatonic so the game has a key

    python3 tools/golfkit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import dsp, fx  # noqa: E402
from core.dsp import FS, rng_for  # noqa: E402
from core.render import normalize_peak, save_wav  # noqa: E402

C5, D5, E5, G5, A5, C6, E6, G6 = 523.25, 587.33, 659.25, 783.99, 880.0, 1046.5, 1318.5, 1568.0


def _norm(x: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return np.asarray(x) / m * peak


def _click(n: int, level: float, rng: np.random.Generator, hp_hz: float = 1800.0,
           ms: float = 0.003) -> np.ndarray:
    k = max(24, int(FS * ms))
    e = np.exp(-np.arange(k) / (k * 0.28))
    burst = dsp.hp(dsp.white(k, rng), hp_hz, FS, order=3) * e
    out = np.zeros(n)
    out[:k] = burst * level
    return out


def _modal(n: int, modes, tau_scale: float = 1.0, exc_tau: float = 0.0015) -> np.ndarray:
    exc = dsp.env_exp(n, FS, attack=0.0004, tau=exc_tau)
    return dsp.modal(exc, [(f, t * tau_scale, g) for f, t, g in modes], FS)


def _marimba(freq: float, dur: float = 0.5, decay: float = 0.12) -> np.ndarray:
    n = int(FS * dur)
    t = np.arange(n) / FS
    body = (np.sin(2 * np.pi * freq * t)
            + np.sin(2 * np.pi * freq * 2 * t) * 0.30
            + np.sin(2 * np.pi * freq * 3.01 * t) * 0.10)
    return body * np.exp(-t / decay) * np.minimum(1.0, t / 0.002)


# --------------------------------------------------------------------------
# shot
# --------------------------------------------------------------------------
def putt(power: float, rng: np.random.Generator | None = None) -> np.ndarray:
    p = float(np.clip(power, 0.0, 1.0))
    rng = rng or rng_for(int(1000 * p))
    n = int(FS * (0.28 + 0.42 * p))
    f_ball = 820.0 + 620.0 * p

    ball = _modal(n, [(f_ball, 0.030 + 0.055 * p, 1.0),
                      (f_ball * 1.83, 0.020 + 0.035 * p, 0.55),
                      (f_ball * 2.94, 0.013 + 0.022 * p, 0.32),
                      (f_ball * 4.21, 0.008 + 0.014 * p, 0.16)])
    k = int(FS * 0.008)                      # 3% bend reads as impact, not as a slide
    if k < n:
        ball[:k] *= np.linspace(0.97, 1.0, k)

    knock = _modal(n, [(190.0 + 90.0 * p, 0.035, 1.0),
                       (430.0 + 180.0 * p, 0.022, 0.45),
                       (760.0 + 300.0 * p, 0.014, 0.20)])

    air = dsp.moving_bp(dsp.white(n, rng), dsp.sweep(n, 2200.0, 6500.0), FS, q=1.6)
    air *= dsp.env_exp(n, FS, attack=0.003, tau=max(0.012, n / FS * 0.09))

    sub_k = min(n, int(FS * 0.09))
    sub = dsp.sine(dsp.sweep(sub_k, (72.0 + 18 * p) * 1.5, 72.0 + 18 * p, "exp"), FS)
    sub *= dsp.env_exp(sub_k, FS, attack=0.001, tau=0.028)

    roll = dsp.bp(dsp.white(n, rng), 1400.0, FS, order=2, q=1.3)
    roll += dsp.bp(dsp.white(n, rng), 2600.0, FS, order=2, q=1.6) * 0.35
    t = np.arange(n) / FS
    roll *= np.exp(-t / max(0.03, n / FS * 0.20)) * (
        1.0 + 0.18 * np.sin(2 * np.pi * rng.uniform(11, 19) * t + rng.uniform(0, 6.28)))
    onset = int(FS * 0.018)
    roll[:onset] *= np.linspace(0.0, 1.0, onset)

    y = (_norm(_click(n, 1.0, rng), 1.00)
         + _norm(ball, 0.55 + 0.15 * p)
         + _norm(knock, 0.30)
         + _norm(air, 0.12 + 0.10 * p)
         + np.pad(_norm(sub, 0.30 + 0.15 * p), (0, n - sub_k))
         + _norm(roll, 0.07 + 0.05 * p))
    y = fx.sat_adaa(y, 1.6 + 1.4 * p)
    return dsp.fade(dsp.hp(y, 120.0, FS, order=2), FS, ms=3.0)


# --------------------------------------------------------------------------
# physics events
# --------------------------------------------------------------------------
def wall_tap(speed: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    s = float(np.clip(speed, 0.0, 1.0))
    rng = rng or rng_for(int(77 + 31 * s))
    n = int(FS * (0.09 + 0.10 * s))
    f = (900.0 + 400.0 * s) * rng.uniform(0.96, 1.04)
    y = (_norm(_click(n, 1.0, rng, hp_hz=2400, ms=0.002), 0.70)
         + _norm(_modal(n, [(f, 0.016 + 0.012 * s, 1.0),
                            (f * 1.87, 0.010 + 0.008 * s, 0.5),
                            (f * 3.1, 0.006, 0.25)]), 0.55)
         + _norm(_modal(n, [(320.0 + 120.0 * s, 0.020, 1.0)]), 0.18))
    return dsp.fade(dsp.hp(y, 180.0, FS, order=2), FS, ms=2.0)


def sand_land(rng: np.random.Generator | None = None) -> np.ndarray:
    """Dull dead thud then grains scattering — the inverse of a hard surface."""
    rng = rng or rng_for(11)
    n = int(FS * 0.55)
    thud = dsp.lp(dsp.white(n, rng), 700.0, FS, order=3)
    thud *= dsp.env_exp(n, FS, attack=0.002, tau=0.055)
    grains = dsp.granulate(dsp.lp(dsp.white(n, rng), 3500.0, FS, order=2), n,
                           density_hz=120, fs=FS, rng=rng, grain_ms=(5, 18),
                           pitch=(0.7, 1.4))
    y = _norm(thud, 0.75) + _norm(grains, 0.22)
    return dsp.fade(dsp.hp(y, 90.0, FS, order=2), FS, ms=3.0)


def water_splash(rng: np.random.Generator | None = None) -> np.ndarray:
    """Splash + bubble chirps: bubbles are what make water read as water."""
    rng = rng or rng_for(12)
    n = int(FS * 0.85)
    splash = dsp.hp(dsp.white(n, rng), 900.0, FS, order=2)
    splash *= dsp.env_exp(n, FS, attack=0.001, tau=0.075)
    out = _norm(splash, 0.75)

    k = int(FS * 0.18)
    gloop = dsp.sine(dsp.sweep(k, 260.0, 90.0, "exp"), FS)
    gloop *= dsp.env_exp(k, FS, 0.002, 0.05)
    out[:k] += _norm(gloop, 0.45)[:k]

    for _ in range(rng.integers(6, 11)):
        t0 = int(rng.uniform(0.03, 0.45) * FS)
        m = int(FS * rng.uniform(0.015, 0.045))
        if t0 + m >= n:
            continue
        b = dsp.sine(dsp.sweep(m, rng.uniform(280, 520), rng.uniform(900, 1700), "exp"), FS)
        b *= np.exp(-np.arange(m) / (m * 0.35)) * rng.uniform(0.10, 0.26)
        out[t0:t0 + m] += b
    return dsp.fade(dsp.hp(out, 80.0, FS, order=2), FS, ms=4.0)


def portal(rng: np.random.Generator | None = None) -> np.ndarray:
    """Teleport reads as pitch and shimmer, not as volume."""
    rng = rng or rng_for(13)
    n = int(FS * 0.55)
    f = dsp.sweep(n, 380.0, 1500.0, "exp")
    tone = dsp.lp(dsp.saw(f, FS), 4200.0, FS, order=2)
    tone *= dsp.env_exp(n, FS, attack=0.006, tau=0.11)
    shimmer = dsp.sine(f * 2.01, FS) * dsp.env_exp(n, FS, 0.004, 0.07) * 0.25
    whoosh = dsp.moving_bp(dsp.white(n, rng), dsp.sweep(n, 900.0, 7000.0), FS, q=1.5)
    whoosh *= dsp.env_exp(n, FS, 0.01, 0.09)
    y = _norm(tone, 0.60) + _norm(shimmer, 0.18) + _norm(whoosh, 0.22)
    return dsp.fade(fx.reverb(y, FS, t60=0.7, size=0.4, mix=0.18, rng=rng), FS, ms=4.0)


def prop_break(rng: np.random.Generator | None = None) -> np.ndarray:
    """Crack then debris — density and brightness say 'it broke'."""
    rng = rng or rng_for(14)
    n = int(FS * 0.7)
    crack = _norm(_click(n, 1.0, rng, hp_hz=1500, ms=0.004), 0.85)
    body = _norm(_modal(n, [(210.0, 0.06, 1.0), (390.0, 0.04, 0.6), (640.0, 0.028, 0.4)]), 0.60)
    debris = np.zeros(n)
    for _ in range(rng.integers(18, 34)):
        i = int(rng.uniform(0.01, 0.35) * n)
        m = min(n - i, int(FS * rng.uniform(0.004, 0.02)))
        if m > 8:
            g = dsp.hp(dsp.white(m, rng), rng.uniform(1800, 5200), FS, order=2)
            debris[i:i + m] += g * np.exp(-np.arange(m) / (m * 0.3)) * rng.uniform(0.05, 0.22)
    return dsp.fade(fx.sat_adaa(crack + body + _norm(debris, 0.45), 1.5), FS, ms=3.0)


def star(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or rng_for(15)
    n = int(FS * 0.5)
    out = np.zeros(n)
    for i, f in enumerate([C6, E6, G6]):
        seg = _marimba(f, 0.28, decay=0.055)
        i0 = int(i * 0.045 * FS)
        e = min(n, i0 + len(seg))
        out[i0:e] += seg[:e - i0] * (1.0 - 0.15 * i)
    sparkle = dsp.hp(dsp.white(n, rng), 7000.0, FS, order=2)
    sparkle *= dsp.env_exp(n, FS, 0.001, 0.03)
    return dsp.fade(_norm(out, 0.7) + _norm(sparkle, 0.10), FS, ms=3.0)


def holed(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or rng_for(9)
    n = int(FS * 0.9)
    out = np.zeros(n)
    k = int(FS * 0.22)
    plunk = dsp.sine(dsp.sweep(k, 420.0, 150.0, "exp"), FS)
    plunk *= dsp.env_exp(k, FS, 0.002, 0.05)
    out[:k] += _norm(plunk, 0.70)
    for dt in (0.11, 0.17):
        i, m = int(dt * FS), int(FS * 0.05)
        if i + m < n:
            r = _modal(m, [(1250.0, 0.020, 1.0), (2320.0, 0.012, 0.5)])
            out[i:i + m] += _norm(r, 0.30)
    for j, f in enumerate([C5, E5, G5, C6]):
        i = int((0.22 + 0.075 * j) * FS)
        seg = _marimba(f, 0.42, 0.10)
        e = min(n, i + len(seg))
        if e > i:
            out[i:e] += seg[:e - i] * 0.5
    return dsp.fade(_norm(out, 0.75), FS, ms=4.0)


# --------------------------------------------------------------------------
# stings (C major pentatonic -> the game has a key)
# --------------------------------------------------------------------------
def sting(kind: str) -> np.ndarray:
    seq = {
        "hole_win":   [(C5, 0.00), (E5, 0.10), (G5, 0.20), (C6, 0.30)],
        "hole_lose":  [(G5, 0.00), (E5, 0.13), (D5, 0.26)],
        "hole_draw":  [(E5, 0.00), (E5, 0.16)],
        "match_win":  [(C5, 0.0), (E5, 0.09), (G5, 0.18), (C6, 0.27), (E6, 0.36), (G6, 0.45)],
        "match_lose": [(A5, 0.0), (G5, 0.14), (E5, 0.28), (D5, 0.42)],
        "match_draw": [(G5, 0.0), (E5, 0.15), (G5, 0.30)],
    }[kind]
    n = int(FS * (1.1 if kind.startswith("match") else 0.6))
    out = np.zeros(n)
    for i, (f, t0) in enumerate(seq):
        seg = _marimba(f, 0.6, 0.16)
        i0 = int(t0 * FS)
        e = min(n, i0 + len(seg))
        out[i0:e] += seg[:e - i0] * (1.0 - 0.06 * i)
    if kind == "match_win":
        out = fx.reverb(out, FS, t60=1.1, size=0.5, mix=0.25, rng=rng_for(21))
    return dsp.fade(_norm(out, 0.7), FS, ms=5.0)


def ui_tap(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or rng_for(16)
    n = int(FS * 0.05)
    return dsp.fade(_norm(_click(n, 1.0, rng, hp_hz=3000, ms=0.0015), 0.5), FS, ms=1.0)


# --------------------------------------------------------------------------
# kit + reel
# --------------------------------------------------------------------------
def kit() -> dict[str, np.ndarray]:
    return {
        "putt_p025": putt(0.25), "putt_p060": putt(0.60), "putt_p100": putt(1.00),
        "wall_a": wall_tap(0.25, rng_for(101)), "wall_b": wall_tap(0.55, rng_for(102)),
        "wall_c": wall_tap(0.85, rng_for(103)),
        "sand": sand_land(), "water": water_splash(), "portal": portal(),
        "break": prop_break(), "star": star(), "holed": holed(),
        "hole_win": sting("hole_win"), "hole_lose": sting("hole_lose"),
        "hole_draw": sting("hole_draw"), "match_win": sting("match_win"),
        "match_lose": sting("match_lose"), "match_draw": sting("match_draw"),
        "ui_tap": ui_tap(),
    }


def _gap(s: float) -> np.ndarray:
    return np.zeros(int(FS * s))


def gameplay_reel() -> np.ndarray:
    """A short round, sound only. You should be able to follow the ball."""
    r = rng_for(4242)
    parts = [
        putt(0.35, r), _gap(0.10), wall_tap(0.45, r), _gap(0.16),
        wall_tap(0.22, r), _gap(0.30), sand_land(r), _gap(0.35),
        putt(0.85, r), _gap(0.22), portal(r), _gap(0.30),
        prop_break(r), _gap(0.22), star(r), _gap(0.40),
        putt(0.55, r), _gap(0.30), water_splash(r), _gap(0.30),
        putt(0.20, r), _gap(0.45), holed(r), _gap(0.25), sting("hole_win"),
        _gap(0.35), sting("match_win"),
    ]
    return normalize_peak(dsp.fade(np.concatenate(parts), FS, ms=8.0), -3.0)


def main() -> int:
    out = ROOT / "out" / "_reel"
    kitdir = out / "kit"
    kitdir.mkdir(parents=True, exist_ok=True)
    sounds = kit()
    for name, buf in sounds.items():
        save_wav(kitdir / f"{name}.wav", normalize_peak(buf, -4.0), FS, bits=24)
    reel = gameplay_reel()
    save_wav(out / "golf_kit_reel.wav", reel, FS, bits=24)
    print(f"{len(sounds)} sounds -> {kitdir}")
    print(f"reel: {out / 'golf_kit_reel.wav'}  ({len(reel) / FS:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
