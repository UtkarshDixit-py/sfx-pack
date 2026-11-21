"""Post-processing: the clean <-> dirty axis of the pack.

References (briefs/RESEARCH.md):
  ADAA / antialiased waveshaping   Parker, Zavalishin & Le Bivic, DAFx-16
  tape hysteresis, wow/flutter     Chowdhury, DAFx-19
  vinyl / gramophone degradation   Gonzalez 2007
  scratch / backspin               Hansen 2010
  image-source IR + velvet tail    Allen & Berkley 1979; Valimaki et al. 2017
  bitcrush soft/harsh              Relyea 2024
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import dsp
from .dsp import FS


# --------------------------------------------------------------------------
# saturation
# --------------------------------------------------------------------------
def sat_adaa(x: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """First-order antiderivative antialiasing around tanh. No fold-back at 48k."""
    d = float(max(drive, 1e-6))
    # Clip BEFORE the antiderivative and use the same clipped signal in the
    # denominator. If the two disagree, large inputs make the formula return ~0
    # instead of +/-1 and the loudest part of a transient is silently erased.
    v = np.clip(np.asarray(x, dtype=np.float64) * d, -20.0, 20.0)
    F = np.log(np.cosh(v)) / d
    Fm = np.concatenate(([F[0]], F[:-1])) if F.size else F
    vm = np.concatenate(([v[0]], v[:-1])) if v.size else v
    den = v - vm
    out = np.empty_like(v)
    small = np.abs(den) < 1e-9
    out[~small] = (F[~small] - Fm[~small]) / den[~small]
    out[small] = np.tanh(v[small])
    return out


def sat_os(x: np.ndarray, drive: float = 3.0, factor: int = 4) -> np.ndarray:
    """Oversampled tanh. Slower than sat_adaa but dead simple to trust."""
    up = signal.resample_poly(x, factor, 1)
    up = np.tanh(up * float(drive))
    return signal.resample_poly(up, 1, factor)[:len(x)]


def soft_clip(x: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(x))) or 1.0
    if peak <= ceiling:
        return x
    return np.tanh(x / peak * 1.2) / np.tanh(1.2) * ceiling


# --------------------------------------------------------------------------
# digital degradation
# --------------------------------------------------------------------------
def bitcrush(x: np.ndarray, bits: int = 8, rate_div: int = 4,
             mode: str = "soft", fs: int = FS) -> np.ndarray:
    """mode 'harsh' -> raw sample-hold (aliasing), 'soft' -> pre-filtered (warm)."""
    y = np.asarray(x, dtype=np.float64)
    rd = max(1, int(rate_div))
    if mode == "soft" and rd > 1:
        y = dsp.lp(y, fs / (2.0 * rd), fs, order=4)
    if rd > 1:
        idx = (np.arange(len(y)) // rd) * rd
        y = y[np.clip(idx, 0, len(y) - 1)]
    q = 2.0 ** (max(2, int(bits)) - 1)
    return np.round(y * q) / q


def stutter(x: np.ndarray, fs: int = FS, rate_hz: float = 8.0,
            duty: float = 0.5, ramp_ms: float = 2.0) -> np.ndarray:
    """Tempo-locked gate. Smooth edges so it does not click."""
    n = len(x)
    period = max(2, int(fs / max(0.1, rate_hz)))
    gate = np.zeros(n)
    k = int(period * float(np.clip(duty, 0.02, 0.98)))
    for i in range(0, n, period):
        gate[i:i + k] = 1.0
    if ramp_ms > 0:
        m = max(1, int(fs * ramp_ms / 1000.0))
        gate = np.convolve(gate, np.hanning(m) / np.hanning(m).sum(), mode="same")
    return x * np.clip(gate, 0.0, 1.0)


# --------------------------------------------------------------------------
# analog character
# --------------------------------------------------------------------------
def tape(x: np.ndarray, fs: int = FS, drive: float = 1.8, wow: float = 0.0018,
         flutter: float = 0.0005, hiss_db: float = -58.0, bw: float = 12000.0,
         rng: np.random.Generator | None = None, wow_hz: float = 0.7,
         flutter_hz: float = 8.5) -> np.ndarray:
    y = sat_adaa(x, drive)
    n = len(y)
    t = np.arange(n) / fs
    mod = (wow * np.sin(2.0 * np.pi * wow_hz * t)
           + flutter * np.sin(2.0 * np.pi * flutter_hz * t))
    idx = np.clip(np.arange(n) * (1.0 + mod), 0.0, n - 1.001)
    y = np.interp(idx, np.arange(n), y)
    y = dsp.lp(y, bw, fs, order=2)
    if rng is not None and hiss_db > -120.0:
        y = y + dsp.white(n, rng) * (10.0 ** (hiss_db / 20.0))
    return y


def vinyl(x: np.ndarray, fs: int = FS, crackle_hz: float = 4.0,
          rumble: float = 0.02, hi: float = 9000.0,
          rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    y = dsp.hp(dsp.lp(x, hi, fs, order=2), 35.0, fs, order=2)
    n = len(y)
    if crackle_hz > 0:
        n_ev = int(max(0, rng.poisson(crackle_hz * n / fs)))
        if n_ev:
            pos = rng.integers(0, n, n_ev)
            amp = rng.uniform(0.01, 0.18, n_ev) * rng.choice([-1.0, 1.0], n_ev)
            imp = np.zeros(n)
            np.add.at(imp, pos, amp)
            kern = np.exp(-np.arange(max(2, int(fs * 0.004))) / (fs * 0.0009))
            y = y + np.convolve(imp, kern)[:n] * 0.8
    if rumble > 0:
        y = y + dsp.lp(dsp.white(n, rng), 60.0, fs, order=2) * rumble
    return y


def scratch(src: np.ndarray, rate_curve: np.ndarray, fs: int = FS,
            friction: float = 0.06, rng: np.random.Generator | None = None,
            start: int | None = None) -> np.ndarray:
    """Turntable gesture: a playback-rate curve over our own material (Hansen 2010).

    `start` is the sample the needle begins at. It defaults to the middle of the
    source, because a backspin starting at index 0 has nowhere to travel.
    """
    rng = rng or np.random.default_rng(0)
    n = len(src)
    rate = np.asarray(rate_curve, dtype=np.float64)
    if start is None:
        start = n // 2
    # SIGNED integration: a negative rate must move the read head backwards.
    # Taking abs() here makes every backspin and reverse stroke play forwards.
    idx = np.clip(start + np.cumsum(rate), 0.0, n - 1.001)
    out = np.interp(idx, np.arange(n), src)
    if friction > 0:
        vel = np.abs(rate_curve[:len(out)]) if len(rate_curve) >= len(out) else np.resize(rate_curve, len(out))
        noise = dsp.bp(dsp.white(len(out), rng), 2200.0, fs, order=2, q=2.0)
        out = out + noise * (friction * vel)
    return out


# --------------------------------------------------------------------------
# space: synthesized impulse responses we own outright
# --------------------------------------------------------------------------
def synth_ir(fs: int = FS, t60: float = 1.2, size: float = 0.4,
             pre_s: float = 0.008, bands: int = 7,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Early reflections + band-split velvet-noise tail. No recorded IR files."""
    rng = rng or np.random.default_rng(0)
    n = int(max(0.05, t60 * 1.15) * fs)
    t = np.arange(n) / fs
    ir = np.zeros(n)

    pre = int(pre_s * fs)
    spread = 0.004 + 0.05 * float(np.clip(size, 0.0, 1.0))
    for k in range(9):
        pos = pre + int(rng.uniform(0.001, spread) * fs)
        if pos < n:
            ir[pos] += (0.75 ** k) * rng.uniform(0.35, 1.0) * rng.choice([-1.0, 1.0])

    tail = np.zeros(n)
    for b in range(bands):
        lo = 80.0 * (2.0 ** b)
        hi = min(lo * 2.0, fs * 0.45)
        if lo >= hi:
            break
        fc = float(np.sqrt(lo * hi))
        t60b = t60 / (1.0 + 0.30 * b)          # highs die faster
        src = dsp.velvet(n, period=max(6, int(fs / 3000)), rng=rng)
        tail += dsp.bp(src, fc, fs, order=2, q=1.1) * np.exp(
            -np.log(1000.0) * t / max(t60b, 1e-3)
        )

    peak = float(np.max(np.abs(tail))) or 1.0
    ir = ir + (tail / peak) * 0.6
    ir = np.concatenate([np.zeros(pre), ir[:-pre] if pre else ir])
    ir *= np.exp(-np.log(1000.0) * t / max(t60, 1e-3))
    ir = dsp.fade(ir, fs, ms=30.0)
    return ir / (np.max(np.abs(ir)) + 1e-12)


def conv(x: np.ndarray, ir: np.ndarray, mix: float = 0.3,
         fs: int = FS, wet_trim: bool = True) -> np.ndarray:
    wet = signal.fftconvolve(x, ir, mode="full")[:len(x)]
    if wet_trim:
        wet = dsp.fade(wet, fs, ms=3.0)
    # Normalise the wet path to the dry peak BEFORE mixing. Without this the
    # convolution's summed energy, not the mix parameter, decides the level:
    # the tail becomes the file's peak and the dry attack lands 10-13 dB under it.
    wp = float(np.max(np.abs(wet))) if wet.size else 0.0
    dp = float(np.max(np.abs(x))) if np.size(x) else 0.0
    if wp > 1e-9 and dp > 1e-9:
        wet = wet * (dp / wp)
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet


def reverb(x: np.ndarray, fs: int = FS, t60: float = 1.2, size: float = 0.4,
           mix: float = 0.3, rng: np.random.Generator | None = None) -> np.ndarray:
    return conv(x, synth_ir(fs=fs, t60=t60, size=size, rng=rng), mix=mix, fs=fs)


# --------------------------------------------------------------------------
# standard chains
# --------------------------------------------------------------------------
def chain_clean(x: np.ndarray, fs: int = FS) -> np.ndarray:
    return dsp.fade(dsp.hp(x, 25.0, fs, order=2), fs, ms=4.0)


def chain_dirty(x: np.ndarray, fs: int = FS,
                rng: np.random.Generator | None = None) -> np.ndarray:
    y = tape(x, fs=fs, rng=rng)
    y = vinyl(y, fs=fs, rng=rng)
    return dsp.fade(y, fs, ms=4.0)
