#!/usr/bin/env python3
"""Hero demo: a 30-second composed moment built ONLY from this pack's generators.

This is not a montage of samples. It is an arrangement, because first impressions
are made by structure — tension, silence, contrast, weight — not by any single
sound. Everything here will exist as a purchasable item later, so the reel is a
proof of the pack, not a separate asset.

    python3 tools/make_hero.py            -> out/_reel/hero_v1.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal as sg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import dsp, fx  # noqa: E402
from core.dsp import FS, note_hz, rng_for  # noqa: E402
from core.render import normalize_lufs, save_wav  # noqa: E402
from families import laser as laser_fam  # noqa: E402

BPM = 140.0
BEAT = 60.0 / BPM
BAR = BEAT * 4
TOTAL = 31.0


# --------------------------------------------------------------------------
# stereo bus
# --------------------------------------------------------------------------
class Bus:
    def __init__(self, seconds: float = TOTAL, fs: int = FS):
        self.fs = fs
        self.n = int(seconds * fs)
        self.main = np.zeros((self.n, 2))
        self.send = np.zeros((self.n, 2))

    def add(self, x: np.ndarray, t: float, gain: float = 1.0, pan: float = 0.0,
            send: float = 0.0) -> None:
        if x is None or len(x) == 0:
            return
        i = int(round(t * self.fs))
        if i >= self.n:
            return
        seg = x[: self.n - i] * gain
        a = (float(np.clip(pan, -1, 1)) + 1.0) * np.pi / 4.0
        l, r = np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)
        self.main[i:i + len(seg), 0] += seg * l
        self.main[i:i + len(seg), 1] += seg * r
        if send > 0:
            self.send[i:i + len(seg), 0] += seg * l * send
            self.send[i:i + len(seg), 1] += seg * r * send

    def master(self, lufs: float = -14.0) -> np.ndarray:
        wet = fx.conv(self.send, fx.synth_ir(fs=self.fs, t60=2.6, size=0.85,
                                             rng=rng_for(7)), mix=1.0, fs=self.fs)
        wet = fx.hp(wet, 180.0, self.fs, order=2)
        y = self.main + wet * 0.9
        y = fx.soft_clip(y, 0.98)
        y = normalize_lufs(y, self.fs, lufs)
        return y


# --------------------------------------------------------------------------
# elements (each one will become a family)
# --------------------------------------------------------------------------
def sub_drone(dur: float, f: float = 34.0, rng: np.random.Generator | None = None) -> np.ndarray:
    n = int(dur * FS)
    rng = rng or rng_for(1)
    t = np.arange(n) / FS
    body = (np.sin(2 * np.pi * f * t) * 0.65
            + np.sin(2 * np.pi * f * 2.003 * t) * 0.12
            + np.sin(2 * np.pi * f * 0.5 * t) * 0.10)
    swell = np.clip(t / (dur * 0.55), 0, 1) ** 2 * np.minimum(1.0, (dur - t) / (dur * 0.35))
    air = dsp.lp(dsp.white(n, rng), 220.0, FS, order=3) * 0.05
    return (body * swell + air) * 0.9


def riser(dur: float, f0: float = 400.0, f1: float = 9000.0,
          rng: np.random.Generator | None = None) -> np.ndarray:
    n = int(dur * FS)
    rng = rng or rng_for(2)
    noise = dsp.white(n, rng)
    curve = np.geomspace(f0, f1, n)
    y = dsp.moving_bp(noise, curve, FS, q=1.1)
    t = np.arange(n) / FS
    env = (t / dur) ** 2.2
    # tonal layer locks in pitch near the top, which is what makes a riser feel
    # like it is *going somewhere* instead of just getting louder
    tone = dsp.saw(np.geomspace(f0 / 4, f1 / 4, n), FS)
    tone = dsp.lp(tone, 6000.0, FS, order=2) * 0.22
    y = y * 0.9 + tone * env
    return y * env


def impact(size: float = 1.0, hardness: float = 0.85,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Modal body + sub drop + noise blast + debris. Farnell's three layers."""
    rng = rng or rng_for(3)
    dur = 2.2 * size
    n = int(dur * FS)
    exc = dsp.contact_burst(n, FS, hardness, rng)
    modes = dsp.modal_bank(f0=95.0 / size, material="concrete", tau0=0.9 * size, size=size, rng=rng)
    body = dsp.modal(exc, modes, FS)
    body = dsp.lp(body, 9000.0, FS, order=2)

    sub = dsp.sine(dsp.sweep(n, 120.0 / size, 28.0, "exp"), FS)
    sub *= dsp.env_exp(n, FS, attack=0.002, tau=0.55 * size) * 1.1

    blast = dsp.hp(dsp.white(n, rng), 1500.0, FS, order=3)
    blast *= dsp.env_exp(n, FS, attack=0.0008, tau=0.09) * 0.7

    debris = np.zeros(n)
    for _ in range(26):
        p = int(rng.uniform(0.02, 0.85) * n)
        k = min(n - p, int(FS * rng.uniform(0.004, 0.03)))
        if k > 8:
            d = dsp.hp(dsp.white(k, rng), 2500.0, FS, order=2) * rng.uniform(0.02, 0.12)
            debris[p:p + k] += d * np.exp(-np.arange(k) / (k * 0.3))
    return fx.sat_adaa(body * 0.55 + sub * 0.8 + blast + debris, 1.6)


def laser_hit(seed: int) -> np.ndarray:
    p = laser_fam.sample_params(rng_for(seed))
    p.update(archetype="zap", dur=max(0.18, p["dur"] * 0.8), drop=min(p["drop"], 0.32))
    buf = laser_fam.generate(p, rng_for(seed), FS).buf
    return fx.reverb(buf, FS, t60=0.5, size=0.35, mix=0.16, rng=rng_for(seed + 1))


def glitch_take(seed: int) -> np.ndarray:
    p = laser_fam.sample_params(rng_for(seed))
    base = laser_fam.generate(p, rng_for(seed), FS).buf
    rng = rng_for(seed + 5)
    g = dsp.granulate(base, int(FS * 0.55), density_hz=48, fs=FS, rng=rng,
                      grain_ms=(8, 40), pitch=(0.6, 1.9))
    g = fx.stutter(g, FS, rate_hz=1.0 / (BEAT / 4), duty=0.42, ramp_ms=1.0)  # 16ths at 140
    return fx.bitcrush(g, bits=8, rate_div=6, mode="soft", fs=FS)


def texture_bed(dur: float, rng: np.random.Generator | None = None) -> np.ndarray:
    n = int(dur * FS)
    rng = rng or rng_for(4)
    noise = dsp.pink(n, rng)
    y = dsp.moving_bp(noise, np.geomspace(300, 2400, n), FS, q=2.2)
    t = np.arange(n) / FS
    y *= np.minimum(1.0, t / 1.2) * np.minimum(1.0, (dur - t) / 1.5)
    return y * 0.5


# --------------------------------------------------------------------------
# arrangement
# --------------------------------------------------------------------------
def build() -> np.ndarray:
    bus = Bus()
    r = rng_for(2024)

    # 1. tension: near-silence that is not silent
    bus.add(sub_drone(7.4), 0.4, gain=0.55, send=0.02)
    bus.add(texture_bed(6.0, r), 1.0, gain=0.35, pan=-0.35, send=0.25)
    bus.add(texture_bed(6.0, rng_for(41)), 1.6, gain=0.28, pan=0.40, send=0.25)

    # 2. build: riser + accelerating ticks
    bus.add(riser(4.2, 300.0, 11000.0), 2.2, gain=0.75, send=0.35)
    for k, t in enumerate([4.2, 4.95, 5.55, 6.0, 6.35]):
        tick = dsp.hp(dsp.white(int(FS * 0.012), r), 4000.0, FS, order=3)
        tick *= np.exp(-np.arange(len(tick)) / (len(tick) * 0.25))
        bus.add(tick, t, gain=0.22 + 0.06 * k, pan=(-1) ** k * 0.25)
    # reverse swell into the drop
    tail = impact(1.0, 0.9, r)[: int(FS * 0.9)]
    bus.add(tail[::-1] * 0.35, 5.15, gain=0.7, send=0.55)

    # 3. THE DROP (bar 2)
    bus.add(impact(1.0, 0.9, r), 6.9, gain=1.0, send=0.30)
    bus.add(sub_drone(3.2, 29.0, r), 7.0, gain=0.9, send=0.05)

    # 4. lasers in rhythm — musical placement, not random
    for i, (t, pan, g) in enumerate([
        (7.55, -0.45, 0.55), (7.87, 0.45, 0.45), (8.19, -0.15, 0.60),
        (8.51, 0.30, 0.42), (9.26, -0.55, 0.50), (9.58, 0.55, 0.44),
    ]):
        bus.add(laser_hit(5000 + i), t, gain=g, pan=pan, send=0.18)

    # 5. glitch break
    bus.add(glitch_take(6001), 9.9, gain=0.55, pan=-0.3, send=0.12)
    bus.add(glitch_take(6002), 10.35, gain=0.45, pan=0.35, send=0.12)

    # 6. second, bigger hit (bar 4)
    bus.add(impact(1.45, 0.95, r), 11.5, gain=1.0, send=0.35)

    # 7. wide evolving section
    bus.add(texture_bed(7.0, rng_for(77)), 12.2, gain=0.42, pan=0.0, send=0.45)
    bus.add(sub_drone(7.5, 36.5, r), 12.2, gain=0.45, send=0.05)
    for i, (t, pan) in enumerate([(13.0, -0.6), (13.9, 0.6), (14.8, -0.3), (15.7, 0.3)]):
        bus.add(laser_hit(7000 + i), t, gain=0.40, pan=pan, send=0.35)

    # 8. final statement, then get out of the way
    bus.add(riser(3.0, 500.0, 12000.0), 16.6, gain=0.55, send=0.4)
    bus.add(impact(1.7, 1.0, r), 19.5, gain=1.0, send=0.5)
    bus.add(sub_drone(4.0, 27.0, r), 19.6, gain=0.85, send=0.1)
    for i, t in enumerate([20.6, 21.1, 21.5]):
        bus.add(laser_hit(8000 + i), t, gain=0.30 - 0.06 * i, pan=0.0, send=0.6)

    # 9. silence is an instrument
    return bus.master(lufs=-13.5)


def main() -> int:
    y = build()
    out = ROOT / "out" / "_reel" / "hero_v1.wav"
    save_wav(out, y, FS, bits=24)
    peak = 20 * np.log10(np.max(np.abs(y)) + 1e-12)
    print(f"{out}  {len(y) / FS:.1f}s  peak {peak:.1f} dBFS")
    print(f"open \"{out}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
