"""Family: glitch — digital damage.

A glitch is a source we would never sell, destroyed on purpose. Every render:
  1. synthesise a short motif (square/saw stab, a 16th-note arp, or a noise burst)
     at a known peak so downstream stages have a reference level
  2. destroy it: buffer-repeat (stutter repeats), dropout gates, bitcrush,
     granular scatter
  3. protect the attack: the first 15 ms are never gated away and the grain cloud
     is normalised to 0.30 against the main body at 1.0
Two characters:
  'harsh'  unfiltered sample-hold bitcrush (aliasing is part of the artifact),
           more dropouts
  'soft'   pre-LPF'd bitcrush + a post low-pass (warm, tape-degraded)
The grain layer always uses the OPPOSITE bitcrush mode, so every render hears
both modes; the character decides which one carries the body.
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "glitch"
DESCRIPTORS = ["glitch", "crush", "corrupt", "datamosh", "stutter"]

SOURCES = ("stab", "arp", "burst")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _set_peak(x: np.ndarray, target: float) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p <= 0.0:
        return np.zeros_like(x)
    return x / p * float(target)


def _accent_head(buf: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Guarantee that the strike, not a mid-file repeat, is the loudest moment:
    measure the first 20 ms against the loudest 20 ms block of the rest and
    taper the tail down by a random 3-9 dB if it would out-shout the head."""
    h = int(fs * 0.020)
    if buf.size < 2 * h:
        return buf
    head_rms = float(np.sqrt(np.mean(buf[:h] ** 2))) + 1e-9
    seg = buf[h:]
    blk = max(1, int(fs * 0.020))
    m = seg.size // blk
    if m < 1:
        return buf
    rest = max(float(np.sqrt(np.mean(seg[i * blk:(i + 1) * blk] ** 2)))
               for i in range(m)) + 1e-9
    g = min(1.0, 10.0 ** (float(rng.uniform(3.0, 9.0)) / 20.0) * head_rms / rest)
    if g >= 0.999:
        return buf
    t1 = min(buf.size, h + int(fs * 0.040))
    env = np.full(buf.size, g)
    env[:h] = 1.0
    env[h:t1] = np.linspace(1.0, g, t1 - h)
    return buf * env


def _restore_attack(x: np.ndarray, fs: int, ms: float = 4.0,
                    final_ms: float = 1.0, cap: float = 40.0) -> np.ndarray:
    """CORE ISSUE workaround. render.master() fades EVERY one-shot in over 4 ms
    (dsp.fade(..., ms=4.0)), which is most of a 30 ms tick and lets mid-file
    events out-shout the strike. Called on our own faded buffer, this divides
    out that exact ramp (capped) so master's 4 ms fade lands as a 1 ms fade:
    the delivered attack is sharp and still click-free."""
    m = int(fs * ms / 1000.0)
    if m <= 2 or x.size < m:
        return x
    k = max(1, int(fs * final_ms / 1000.0))
    r = np.maximum(np.linspace(0.0, 1.0, m), 1e-6)
    s = np.minimum(1.0, np.arange(m) / float(k))
    g = np.minimum(cap, s / (r * r))
    y = x.copy()
    y[:m] *= g
    return y


def _click(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    """2 ms broadband click that guarantees a transient at t=0."""
    k = max(8, int(fs * 0.0018))
    out = np.zeros(n)
    burst = dsp.hp(dsp.white(k, rng), 3200.0, fs, order=3)
    burst = burst * dsp.env_exp(k, fs, attack=0.00012, tau=0.0006)
    out[:k] = _set_peak(burst, 1.0)
    return out


def _synth_source(kind: str, n: int, fs: int, f0: float, bpm: float,
                  dur: float, rng: np.random.Generator) -> np.ndarray:
    """The undamaged material. Normalised to a known peak by the caller."""
    t = np.arange(n) / fs
    if kind == "stab":
        f = f0 * (1.0 + 0.02 * np.exp(-t / 0.03))
        a = dsp.saw(f * 0.996, fs)
        b = dsp.square(f * 1.004, fs, duty=0.42)
        src = 0.55 * a + 0.45 * b
        src = dsp.lp(src, min(16000.0, f0 * 14.0 + 1500.0), fs, order=2)
        src *= dsp.env_exp(n, fs, attack=0.002, tau=max(0.06, dur * 0.45))
        src = src + 0.06 * dsp.white(n, rng) * np.exp(-t / 0.02)
    elif kind == "arp":
        src = np.zeros(n)
        step = 60.0 / (bpm * 4.0)                      # 16th notes
        L = max(8, int(fs * step))
        scale = np.array([0, 3, 5, 7, 10, 12])
        pos = 0
        while pos < n:
            semi = int(rng.choice(scale))
            nf = f0 * 2.0 ** (semi / 12.0)
            ln = min(L, n - pos)
            seg = dsp.square(dsp.const(ln, nf), fs, duty=0.45)
            seg *= dsp.env_exp(ln, fs, attack=0.0015, tau=(L / fs) * 0.80)
            src[pos:pos + ln] += seg
            pos += L
        src = dsp.lp(src, min(16000.0, f0 * 12.0 + 2000.0), fs, order=2)
    else:  # burst: gated noise blasts with a swept tone underneath
        src = np.zeros(n)
        pos = 0
        while pos < n:
            ln = int(fs * rng.uniform(0.006, 0.020))
            if pos + ln > n:
                break
            nb = dsp.white(ln, rng) * dsp.env_exp(ln, fs, attack=0.0004,
                                                 tau=(ln / fs) * 0.35)
            nb = dsp.bp(nb, float(rng.uniform(600.0, 6000.0)), fs, order=2, q=1.2)
            src[pos:pos + ln] += nb
            pos += ln + int(fs * rng.uniform(0.008, 0.030))
        sw = dsp.sweep(n, f0 * 2.0, f0 * 0.5, "exp")
        tone = dsp.saw(sw, fs) * dsp.env_exp(n, fs, attack=0.002, tau=dur * 0.45)
        src = src + 0.35 * dsp.lp(tone, min(12000.0, f0 * 8.0 + 800.0), fs, order=2)
    return _set_peak(src, 0.85)


def _buffer_repeat(x: np.ndarray, fs: int, bpm: float, subdiv: int,
                   reps: int, rng: np.random.Generator) -> np.ndarray:
    """Tile one tempo-locked slice over the back half. 1.5 ms fades per copy so
    the repeats chop instead of clicking."""
    n = len(x)
    L = int(fs * 60.0 / (bpm * subdiv))               # one subdivision
    L = max(8, min(L, n // 2))
    start = int(rng.uniform(0.05, 0.55) * max(1, n - L))
    begin = int(rng.uniform(0.35, 0.65) * n)
    seg = x[start:start + L].copy()
    out = x.copy()
    pos = begin
    for r in range(int(reps)):
        if pos >= n:
            break
        take = min(L, n - pos)
        piece = dsp.fade(seg[:take].copy(), fs, ms=1.5)
        out[pos:pos + take] = piece * (1.0 - 0.12 * r)
        pos += L
    return out


def _dropout_mask(n: int, fs: int, count: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Zero short random windows with 1.5 ms edges. Never touches the first 40 ms
    so the attack survives."""
    mask = np.ones(n)
    guard = int(fs * 0.040)
    for _ in range(int(count)):
        ln = int(fs * rng.uniform(0.004, 0.030))
        if n - guard - ln <= 0:
            break
        s = guard + int(rng.uniform(0.0, 1.0) * (n - guard - ln))
        mask[s:s + ln] = 0.0
    m = max(1, int(fs * 0.0015))
    kern = np.hanning(m)
    kern /= max(kern.sum(), 1e-9)
    return np.convolve(mask, kern, mode="same")


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    char = "harsh" if rng.random() < 0.58 else "soft"
    src = str(rng.choice(SOURCES, p=[0.45, 0.30, 0.25]))
    dur = float(0.18 + 0.82 * (rng.random() ** 0.70))          # 0.18 .. 1.00
    bpm = float(rng.choice([120.0, 128.0, 140.0, 150.0, 160.0, 174.0]))
    subdiv = int(rng.choice([4, 8, 16]))
    f0 = float(np.exp(rng.uniform(np.log(90.0), np.log(900.0))))
    if char == "harsh":
        word = str(rng.choice(["crush", "corrupt"]))
        dropouts = int(rng.integers(0, 5))
    else:
        word = str(rng.choice(["glitch", "datamosh", "stutter"]))
        dropouts = int(rng.integers(0, 3))
    return dict(
        char=char,
        src=src,
        word=word,
        dur=dur,
        bpm=bpm,
        subdiv=subdiv,
        f0=f0,
        bits=int(rng.integers(4, 11)),                          # 4 .. 10
        rate_div=int(rng.choice([2, 3, 4, 6, 8, 12, 16, 24, 32])),
        repeat=bool(rng.random() < 0.75),
        reps=int(rng.integers(2, 7)),
        dropouts=dropouts,
        gesture=str(rng.choice(["none", "reverse", "scratch", "freeze"],
                               p=[0.34, 0.24, 0.24, 0.18])),
        tone=str(rng.choice(["full", "bright", "dark", "narrow"],
                            p=[0.34, 0.22, 0.20, 0.24])),
        tone_f=float(np.exp(rng.uniform(np.log(300.0), np.log(4000.0)))),
        gran_density=float(rng.uniform(18.0, 150.0)),
        gran_pitch_lo=float(rng.uniform(0.45, 0.85)),
        gran_pitch_hi=float(rng.uniform(1.25, 2.20)),
        gran_gain=float(rng.uniform(0.12, 0.42)),
        gran_band=float(np.exp(rng.uniform(np.log(250.0), np.log(6000.0)))),
        bed_gain=float(rng.uniform(0.010, 0.050)),
        kill_at=float(rng.uniform(0.0, 0.65)),
        bed_band=float(np.exp(rng.uniform(np.log(300.0), np.log(5000.0)))),
        click_gain=float(rng.uniform(0.35, 0.70)),
        glue=float(rng.uniform(1.05, 1.45)),
    )


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    n = max(64, int(fs * p["dur"]))

    # 1. source -------------------------------------------------------------
    src = _synth_source(p["src"], n, fs, p["f0"], p["bpm"], p["dur"], rng)

    # 2. destroy it ---------------------------------------------------------
    main = src.copy()
    if p["gesture"] == "reverse":
        s = int(rng.uniform(0.25, 0.50) * n)
        main[s:] = main[s:][::-1]
    elif p["gesture"] == "scratch":
        rc = 1.0 + 0.8 * np.sin(np.linspace(0.0, rng.uniform(2.0, 5.0) * np.pi, n))
        main = fx.scratch(main, np.clip(rc, 0.05, 2.5), fs=fs,
                          friction=float(rng.uniform(0.0, 0.05)), rng=rng)
    elif p["gesture"] == "freeze":
        # buffer freeze: a 5-18 ms slice tiled into a buzz
        fl = max(8, int(fs * rng.uniform(0.005, 0.018)))
        fstart = int(rng.uniform(0.10, 0.70) * max(1, n - fl))
        start = int(rng.uniform(0.25, 0.60) * n)
        reps = int(rng.integers(4, 20))
        seg = main[fstart:fstart + fl].copy()
        pos = start
        for r in range(reps):
            if pos >= n:
                break
            take = min(fl, n - pos)
            main[pos:pos + take] = seg[:take] * (1.0 - 0.03 * r)
            pos += fl
    if p["repeat"]:
        main = _buffer_repeat(main, fs, p["bpm"], p["subdiv"], p["reps"], rng)
    if p["dropouts"]:
        main = main * _dropout_mask(n, fs, p["dropouts"], rng)

    if p["char"] == "harsh":
        main = fx.bitcrush(main, bits=p["bits"], rate_div=p["rate_div"],
                           mode="harsh", fs=fs)
    else:
        main = fx.bitcrush(main, bits=p["bits"], rate_div=p["rate_div"],
                           mode="soft", fs=fs)
        main = dsp.lp(main, float(rng.uniform(8500.0, 13000.0)), fs, order=2)

    # a wide spectral-tilt axis keeps 60 candidates from sharing one timbre
    if p["tone"] == "bright":
        main = dsp.hp(main, float(np.clip(p["tone_f"], 500.0, 4000.0)), fs, order=2)
    elif p["tone"] == "dark":
        main = dsp.lp(main, float(np.clip(p["tone_f"], 900.0, 4000.0)), fs, order=2)
    elif p["tone"] == "narrow":
        main = dsp.bp(main, float(np.clip(p["tone_f"], 300.0, 4000.0)), fs,
                      order=2, q=float(rng.uniform(2.0, 4.5)))

    # the machine sometimes just dies: abrupt collapse to near-silence
    if p["kill_at"] > 0.05:
        k0 = int(p["kill_at"] * n)
        kl = max(4, int(fs * rng.uniform(0.015, 0.060)))
        kl = min(kl, n - k0)
        if kl > 0:
            env = np.full(n, 0.02)
            env[:k0] = 1.0
            env[k0:k0 + kl] = np.linspace(1.0, 0.02, kl)
            main = main * env
    main = _set_peak(main, 1.0)

    # 3. scatter: grain cloud in the opposite crush mode, well under the body --
    grain_mode = "soft" if p["char"] == "harsh" else "harsh"
    gr = dsp.granulate(src, n, p["gran_density"], fs, rng,
                       grain_ms=(8.0, 45.0),
                       pitch=(p["gran_pitch_lo"], p["gran_pitch_hi"]),
                       position=(0.0, 0.80))
    gr = fx.bitcrush(gr, bits=min(12, p["bits"] + 3),
                     rate_div=max(2, p["rate_div"] // 3), mode=grain_mode, fs=fs)
    gr = dsp.bp(gr, p["gran_band"], fs, order=2, q=float(rng.uniform(0.9, 2.2)))
    gr = dsp.hp(gr, 150.0, fs, order=2)
    k = int(fs * 0.020)
    if 0 < k < len(gr):
        gr[:k] *= np.linspace(0.0, 1.0, k)                     # keep the attack clear
    gr = _set_peak(gr, p["gran_gain"])

    # a very low digital noise floor: keeps dropout gaps from turning into
    # identical "silence" regions (which is what makes cheap glitch packs clone-y)
    bed = dsp.bp(dsp.pink(n, rng), p["bed_band"], fs, order=2,
                 q=float(rng.uniform(0.7, 1.8)))
    bed = _set_peak(bed, p["bed_gain"])

    clk = _set_peak(_click(n, fs, rng), p["click_gain"])

    buf = mix([
        Layer(main, gain=1.0, label="main"),
        Layer(gr, gain=1.0, label="grains"),
        Layer(bed, gain=1.0, label="bed"),
        Layer(clk, gain=1.0, label="click"),
    ], fs=fs, stereo=False)

    # the strike is the loudest moment; the corrupted tail sits under it
    tl = float(rng.uniform(0.55, 0.80))
    hd = max(8, int(fs * rng.uniform(0.015, 0.040)))
    head_env = np.full(n, tl)
    head_env[:hd] = np.linspace(1.0, tl, hd)
    buf = buf * head_env

    buf = fx.sat_adaa(buf, p["glue"])                          # glue the seams
    buf = dsp.hp(buf, float(rng.uniform(45.0, 90.0)), fs, order=2)
    buf = _accent_head(buf, fs, rng)
    buf = _set_peak(buf, 0.95)
    buf = dsp.fade(buf, fs, ms=4.0)
    buf = _restore_attack(buf, fs)
    buf = dsp.dc_block(buf, fs, fc=15.0)      # the restored head can carry offset
    buf = _set_peak(buf, 0.95)

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, char=p["char"], source=p["src"],
                            loop=False,
                            descriptor=f"{p['word']}_{int(p['f0'])}",
                            bits=p["bits"], rate_div=p["rate_div"],
                            bpm=p["bpm"], subdiv=p["subdiv"]))
