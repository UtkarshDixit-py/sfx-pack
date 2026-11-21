#!/usr/bin/env python3
"""60-second storefront preview reel, built from the delivered master files.

A preview is not a montage of clips — it is a short arrangement. It runs the
buyer through the pack in the order that answers "would this help me?":
ambience -> tension -> weight -> character -> groove -> low end -> damage -> out.

    python3 tools/preview_reel.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import dsp, fx  # noqa: E402
from core.dsp import FS  # noqa: E402
from core.render import normalize_lufs, save_wav  # noqa: E402

MASTER = ROOT / "master"
BPM = 140.0
BEAT = 60.0 / BPM
TOTAL = 60.0


def pool(cat: str) -> list[Path]:
    return sorted((MASTER / cat).glob("*.wav")) if (MASTER / cat).exists() else []


class Bus:
    def __init__(self, seconds: float = TOTAL):
        self.n = int(seconds * FS)
        self.x = np.zeros((self.n, 2))

    def add(self, path: Path, t: float, gain: float = 1.0, pan: float = 0.0) -> None:
        if path is None or not Path(path).exists():
            return
        y, sr = sf.read(str(path), always_2d=False, dtype="float64")
        if sr != FS:
            return
        if y.ndim == 2:
            y = y.mean(axis=1)
        i = int(round(t * FS))
        if i >= self.n:
            return
        seg = y[: self.n - i] * gain
        a = (float(np.clip(pan, -1, 1)) + 1.0) * np.pi / 4.0
        self.x[i:i + len(seg), 0] += seg * np.cos(a) * np.sqrt(2)
        self.x[i:i + len(seg), 1] += seg * np.sin(a) * np.sqrt(2)

    def finish(self) -> np.ndarray:
        y = fx.soft_clip(self.x, 0.97)
        y = normalize_lufs(y, FS, -14.0)
        return dsp.fade_asym(y, FS, in_ms=30.0, out_ms=900.0)


def pick(paths: list[Path], k: int, total: int) -> list[Path]:
    """Even spread through a category so the reel does not audition 4 neighbours."""
    if not paths:
        return []
    idx = np.linspace(0, len(paths) - 1, total).astype(int)
    return [paths[i] for i in idx]


def build() -> np.ndarray:
    random.seed(7)
    bus = Bus()

    room = pool("AMBRoom") + pool("AMBDsgn") + pool("AMBTech") + pool("AMBAir") + pool("DSGNDron")
    booms = pool("DSGNBoom")
    rises = pool("DSGNRise")
    whsh = pool("DSGNWhsh")
    impacts = (pool("METLImpt") + pool("WOODImpt") + pool("GLASImpt")
               + pool("ROCKImpt") + pool("PLASImpt") + pool("CERMImpt"))
    lasers = pool("LASRBeam") + pool("LASRGun")
    scifi = pool("SCIEnrg") + pool("SCICmpt") + pool("SCIDoor") + pool("SCIMach")
    perc = pool("MUSCPerc")
    lowend = pool("MUSCInst") + pool("MUSCSynth")
    glitch = pool("UIGlitch") + pool("DSGNRythm")
    ui = pool("UIClick") + pool("UIBeep") + pool("UIAlert") + pool("UIData")
    alarm = pool("ALRMSirn") + pool("ALRMElec") + pool("ALRMBuzr")
    arcade = pool("GAMEArcd")

    # 1. ambience — set the room
    bus.add(pick(room, 0, 4)[0] if room else None, 0.2, 0.55, -0.3)
    if len(room) > 8:
        bus.add(pick(room, 1, 6)[3], 1.0, 0.40, 0.35)

    # 2. tension -> drop
    bus.add(rises[len(rises) // 2] if rises else None, 2.6, 0.85)
    bus.add(booms[0] if booms else None, 6.4, 1.0)
    bus.add(whsh[len(whsh) // 3] if whsh else None, 5.6, 0.5, 0.2)

    # 3. weight and material — the "is this real?" test
    for i, p in enumerate(pick(impacts, 0, 6)):
        bus.add(p, 7.4 + 0.72 * i, 0.85 if i % 2 == 0 else 0.55, (-0.6, 0.6)[i % 2])
    bus.add(booms[len(booms) // 2] if booms else None, 12.1, 0.95)

    # 4. character — lasers and sci-fi
    for i, p in enumerate(pick(lasers, 0, 4)):
        bus.add(p, 13.6 + 0.42 * i, 0.55, (-0.5, 0.5, -0.2, 0.25)[i % 4])
    for i, p in enumerate(pick(scifi, 0, 3)):
        bus.add(p, 15.4 + 0.55 * i, 0.45, (0.4, -0.4, 0.0)[i % 3])

    # 5. groove — the pack has to survive being played rhythmically
    if perc:
        kicks = [p for p in perc if "kick" in p.name.lower()] or perc[::4]
        snrs = [p for p in perc if "snare" in p.name.lower()] or perc[1::4]
        hats = [p for p in perc if "hat" in p.name.lower()] or perc[2::4]
        claps = [p for p in perc if "clap" in p.name.lower()] or snrs
        t = 17.2
        for bar in range(4):
            for step in range(16):
                st = t + (bar * 4 + step / 4) * BEAT
                if step in (0, 3, 6, 10, 14):
                    bus.add(kicks[(bar + step) % len(kicks)], st, 0.95)
                if step in (4, 12):
                    bus.add(snrs[(bar + step) % len(snrs)], st, 0.70, 0.05)
                if step == 12:
                    bus.add(claps[(bar + step) % len(claps)], st, 0.35, -0.1)
                bus.add(hats[(bar * 16 + step) % len(hats)], st, 0.22 if step % 2 else 0.34, 0.25)

    # 6. low end — 808s and synth bass, five different notes
    for i, p in enumerate(pick(lowend, 0, 5)):
        bus.add(p, 24.1 + 0.86 * i, 0.80)
    bus.add(rises[0] if rises else None, 28.4, 0.45)

    # 7. damage — glitch and stutter
    for i, p in enumerate(pick(glitch, 0, 6)):
        bus.add(p, 30.2 + 0.34 * i, 0.48, (-0.4, 0.4)[i % 2])
    bus.add(booms[-1] if booms else None, 32.4, 0.85)

    # 8. interface, alarms, arcade — the fast, readable stuff
    for i, p in enumerate(pick(ui, 0, 5)):
        bus.add(p, 34.2 + 0.20 * i, 0.42, (-0.3, 0.3)[i % 2])
    for i, p in enumerate(pick(alarm, 0, 2)):
        bus.add(p, 35.6 + 0.9 * i, 0.40)
    for i, p in enumerate(pick(arcade, 0, 6)):
        bus.add(p, 37.4 + 0.26 * i, 0.45, (-0.5, 0.5, -0.25, 0.25, -0.1, 0.1)[i % 6])

    # 9. second act: bigger groove plus bass, the "this is usable" section
    t = 40.0
    if perc:
        for bar in range(4):
            for step in range(16):
                st = t + (bar * 4 + step / 4) * BEAT
                if step in (0, 6, 10):
                    bus.add(kicks[(bar + step) % len(kicks)], st, 0.9)
                if step in (4, 12):
                    bus.add(snrs[(bar + step) % len(snrs)], st, 0.6)
                bus.add(hats[(bar * 16 + step) % len(hats)], st, 0.25, 0.3)
        for i, p in enumerate(pick(lowend, 2, 4)):
            bus.add(p, t + 1.2 + 1.7 * i, 0.75)

    # 10. get out cleanly
    bus.add(whsh[0] if whsh else None, 48.6, 0.55)
    for i, p in enumerate(pick(impacts, 3, 3)):
        bus.add(p, 49.6 + 0.5 * i, 0.55, (-0.4, 0.4, 0)[i % 3])
    bus.add(booms[len(booms) // 3] if booms else None, 51.0, 1.0)
    if room:
        bus.add(room[-1], 50.0, 0.35, 0.0)

    return bus.finish()


def main() -> int:
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    y = build()
    wav = out / "preview.wav"
    save_wav(wav, y, FS, bits=24)
    print(f"preview: {wav}  {len(y) / FS:.1f}s  {wav.stat().st_size / 1024 ** 2:.1f} MB")
    try:
        import subprocess
        mp3 = out / "preview.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)], check=True)
        print(f"mp3:     {mp3}  {mp3.stat().st_size / 1024 ** 2:.1f} MB")
    except Exception as exc:
        print(f"mp3 skipped ({type(exc).__name__}) — wav only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
