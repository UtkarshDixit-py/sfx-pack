"""Family: stutter — tempo-locked gated chops (trap / jersey / drill).

The gate rate is never a free number: it is derived from a BPM and a subdivision,
so every chop lands on the grid of the track it will be dropped into
(e.g. 16ths at 140 BPM = 140/60*4 = 9.33 Hz).

Three gestures:
  gate     continuous duty-cycled gate (classic jersey 16th stutter)
  pattern  an 8-step on/off mask at the gate rate (syncopated chop)
  accel    rate ramps geometrically from the nominal subdivision to 2-4x
           (the stutter "roll" at the end of a bar)

A real source always sits under the gate: a detuned-saw stab with sustain, a
band-passed noise hit with a tonal anchor, or a Karplus plucked note. Gaps are
filled by a very low noise floor so chopped silence does not read as dead air.
Every gate edge is smoothed over <=2 ms (shorter at high rates) so nothing clicks.
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "stutter"
DESCRIPTORS = ["stutter", "chop", "gate", "jersey", "drill"]

PATTERNS = (
    [1, 0, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 0],
)


def _set_peak(x: np.ndarray, target: float) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p <= 0.0:
        return np.zeros_like(x)
    return x / p * float(target)


def _accent_head(buf: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    """The first chop is the accent. If a later 20 ms block is louder than the
    first one, taper the tail down by a random 3-9 dB so the strike stays first."""
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
    """CORE ISSUE workaround. render.master() fades EVERY one-shot in over 4 ms.
    Applied to our own faded buffer, this divides out that ramp (capped) so
    master's 4 ms fade lands as a 1 ms fade: the first chop keeps its attack."""
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
    k = max(8, int(fs * 0.0016))
    out = np.zeros(n)
    burst = dsp.hp(dsp.white(k, rng), 3000.0, fs, order=3)
    burst = burst * dsp.env_exp(k, fs, attack=0.00012, tau=0.0006)
    out[:k] = _set_peak(burst, 1.0)
    return out


def _synth_source(kind: str, n: int, fs: int, f0: float, dur: float,
                  rng: np.random.Generator, cut: float,
                  sustain: float = 0.55) -> np.ndarray:
    t = np.arange(n) / fs
    if kind == "stab":
        f = f0 * (1.0 + 0.012 * np.exp(-t / 0.02))
        a = dsp.saw(f * 0.997, fs)
        b = dsp.saw(f * 1.003, fs)
        sq = 0.30 * dsp.square(f * 0.5, fs, duty=0.5)
        src = 0.45 * a + 0.45 * b + sq
        src += 0.06 * dsp.white(n, rng) * np.exp(-t / 0.03)
        src = dsp.lp(src, min(14400.0, max(cut, f0 * 2.5)), fs, order=2)
        src *= dsp.env_adsr(n, fs, a=0.002, d=min(0.060, 0.25 * dur),
                            s=sustain, r=min(0.080, 0.20 * dur))
    elif kind == "noise":
        nb = dsp.bp(dsp.pink(n, rng), f0 * 3.0, fs, order=2, q=1.8)
        nb *= dsp.env_exp(n, fs, attack=0.001, tau=dur * 0.70)
        tone = dsp.sine(dsp.const(n, f0), fs) * float(rng.uniform(0.0, 0.60))
        tone *= dsp.env_exp(n, fs, attack=0.001, tau=dur * 0.50)
        src = nb + tone
    else:  # pluck
        decay = float(rng.uniform(0.9950, 0.9996))
        src = dsp.karplus(f0, n, fs, decay=decay,
                          bright=float(rng.uniform(0.45, 0.85)), rng=rng)
    src = dsp.lp(src, min(14400.0, max(cut, f0 * 2.0)), fs, order=2)
    return _set_peak(src, 1.0)


def _gate(n: int, fs: int, rate_hz: float, duty: float, ramp_ms: float,
          rng: np.random.Generator, pattern: list[int] | None = None,
          rate_curve: np.ndarray | None = None) -> np.ndarray:
    """Tempo-locked gate. Edges convolved with a Hann ramp (patched by hand so
    the ramp can never exceed half the open window at fast subdivisions)."""
    gate = np.zeros(n)
    if rate_curve is not None:
        rc = np.asarray(rate_curve, dtype=np.float64)
        pos, i, periods = 0, 0, []
        while pos < n:
            r = float(rc[min(i, rc.size - 1)])
            period = max(2, int(fs / max(1.0, r)))
            k = max(1, int(period * duty))
            if pattern is None or pattern[i % len(pattern)]:
                gate[pos:min(n, pos + k)] = 1.0
            periods.append(period)
            pos += period
            i += 1
        base_period = float(np.median(periods)) if periods else float(fs / rate_hz)
    else:
        period = max(2, int(fs / max(0.1, rate_hz)))
        k = max(1, int(period * duty))
        for i, s in enumerate(range(0, n, period)):
            if pattern is None or pattern[i % len(pattern)]:
                gate[s:min(n, s + k)] = 1.0
        base_period = float(period)

    open_ms = duty * base_period / fs * 1000.0
    ramp_eff = float(min(max(ramp_ms, 0.6), max(0.6, 0.45 * open_ms)))
    m = max(2, int(fs * ramp_eff / 1000.0))
    kern = np.hanning(m)
    kern /= max(kern.sum(), 1e-9)
    return np.clip(np.convolve(gate, kern, mode="same"), 0.0, 1.0)


def sample_params(rng: np.random.Generator) -> dict:
    bpm = float(rng.choice([100.0, 120.0, 128.0, 140.0, 150.0, 160.0, 174.0]))
    subdiv = int(rng.choice([2, 4, 8], p=[0.20, 0.55, 0.25]))
    mode = str(rng.choice(["gate", "pattern", "accel"], p=[0.40, 0.35, 0.25]))
    src = str(rng.choice(["stab", "noise", "pluck"], p=[0.50, 0.30, 0.20]))
    if mode == "accel":
        word = "stutter"
    elif src == "pluck":
        word = "chop"
    elif mode == "pattern":
        word = "jersey" if bpm >= 145.0 else "drill"
    else:
        word = "gate"
    pat = list(PATTERNS[int(rng.integers(0, len(PATTERNS)))]) if mode == "pattern" else [1]
    return dict(
        bpm=bpm,
        subdiv=subdiv,
        mode=mode,
        src=src,
        word=word,
        pattern=pat,
        accel_ratio=float(rng.uniform(1.8, 5.0)),
        dur=float(rng.uniform(0.32, 1.18)),
        duty=float(rng.uniform(0.15, 0.85)),
        dry_frac=float(rng.uniform(0.0, 0.55)),
        sustain=float(rng.uniform(0.25, 0.90)),
        life=float(rng.uniform(0.35, 6.0)),
        head_accent=float(rng.uniform(0.30, 1.50)),
        f0=float(np.exp(rng.uniform(np.log(80.0), np.log(700.0)))),
        ramp_ms=float(rng.uniform(1.0, 2.0)),
        drive=float(rng.uniform(1.10, 2.20)),
        sub_gain=float(rng.uniform(0.10, 0.70)) if rng.random() < 0.70 else 0.0,
        click_gain=float(rng.uniform(0.20, 0.55)),
        floor_gain=float(rng.uniform(0.005, 0.035)),
        floor_band=float(np.exp(rng.uniform(np.log(250.0), np.log(7000.0)))),
        floor_q=float(rng.uniform(1.2, 3.5)),
        gate_depth=float(rng.uniform(0.55, 1.0)),
        tone=str(rng.choice(["full", "bright", "dark", "narrow"],
                            p=[0.32, 0.24, 0.22, 0.22])),
        tone_f=float(np.exp(rng.uniform(np.log(300.0), np.log(4500.0)))),
        src_cut=float(np.exp(rng.uniform(np.log(600.0), np.log(12000.0)))),
    )


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    n = max(64, int(fs * p["dur"]))
    rate_hz = p["bpm"] / 60.0 * p["subdiv"]

    src = _synth_source(p["src"], n, fs, p["f0"], p["dur"], rng,
                        p["src_cut"], p["sustain"])

    if p["mode"] == "accel":
        curve = np.geomspace(rate_hz, rate_hz * p["accel_ratio"], 96)
        gate = _gate(n, fs, rate_hz, p["duty"], p["ramp_ms"], rng,
                     rate_curve=curve)
    elif p["mode"] == "pattern":
        gate = _gate(n, fs, rate_hz, p["duty"], p["ramp_ms"], rng,
                     pattern=p["pattern"])
    else:
        gate = _gate(n, fs, rate_hz, p["duty"], p["ramp_ms"], rng)

    # optional dry intro: the source plays straight, then the chop takes over
    if p["dry_frac"] > 0.02:
        t0 = int(p["dry_frac"] * n)
        ramp = np.zeros(n)
        ramp[:t0] = 1.0
        fl = max(4, int(fs * 0.006))
        ramp[t0:min(n, t0 + fl)] = np.linspace(1.0, 0.0, min(fl, n - t0))
        gate = np.maximum(gate, ramp)

    # closed gate -> floor; open gate -> full source (depth < 1 keeps a ghost)
    depth = p["gate_depth"]
    chop_env = depth * gate

    chop = _set_peak(src * chop_env, 1.0)

    # drill weight: hard-gated sine an octave down
    t = np.arange(n) / fs
    sub = dsp.sine(dsp.const(n, p["f0"] * 0.5), fs)
    sub *= dsp.env_exp(n, fs, attack=0.002, tau=p["dur"] * 0.60)
    sub = _set_peak(sub * chop_env, p["sub_gain"])

    floor = dsp.bp(dsp.white(n, rng), p["floor_band"], fs, order=2,
                   q=p["floor_q"])
    floor = _set_peak(floor * (1.0 - 0.85 * gate), p["floor_gain"])

    clk = _set_peak(_click(n, fs, rng), p["click_gain"])

    buf = mix([
        Layer(chop, gain=1.0, label="chop"),
        Layer(sub, gain=1.0, label="sub"),
        Layer(floor, gain=1.0, label="floor"),
        Layer(clk, gain=1.0, label="click"),
    ], fs=fs, stereo=False)

    # the first chop is the accent; later chops sit under it
    buf = buf * (1.0 + p["head_accent"] * np.exp(-t / 0.050))
    # global life: 0.35 = the chop dies fast, >3 = essentially flat
    buf = buf * np.exp(-t / (p["dur"] * p["life"]))
    buf = fx.sat_adaa(buf, p["drive"])
    if p["tone"] == "bright":
        buf = dsp.hp(buf, float(np.clip(p["tone_f"], 500.0, 4500.0)), fs, order=2)
    elif p["tone"] == "dark":
        buf = dsp.lp(buf, float(np.clip(p["tone_f"], 700.0, 4500.0)), fs, order=2)
    elif p["tone"] == "narrow":
        buf = dsp.bp(buf, float(np.clip(p["tone_f"], 300.0, 4500.0)), fs,
                     order=2, q=float(rng.uniform(1.8, 4.0)))
    buf = dsp.hp(buf, float(rng.uniform(40.0, 70.0)), fs, order=2)
    buf = _accent_head(buf, fs, rng)
    buf = _set_peak(buf, 0.95)
    buf = dsp.fade(buf, fs, ms=4.0)
    buf = _restore_attack(buf, fs)
    buf = dsp.dc_block(buf, fs, fc=15.0)      # the restored head can carry offset
    buf = _set_peak(buf, 0.95)

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, mode=p["mode"], source=p["src"],
                            loop=False,
                            descriptor=f"{p['word']}_{int(p['bpm'])}",
                            bpm=p["bpm"], rate_hz=round(rate_hz, 3),
                            subdiv=p["subdiv"]))
