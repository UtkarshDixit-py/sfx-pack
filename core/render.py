"""Layer assembly, render entry point, and the master/delivery chain.

Targets (briefs/RESEARCH.md):
  one-shots      peak-normalised, -6 dBFS default (LUFS unreliable < 400 ms)
  beds/loops     integrated loudness, -24 LKFS home / -18 portable (ASWG-R001)
  everything     true peak <= -1 dBTP (BS.1770-5)
  24-bit master  plain rounding is inaudible; TPDF dither only for 16-bit derivatives
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import dsp, fx
from .dsp import FS

TARGET_PEAK_DBFS = -6.0
TRUE_PEAK_CEIL_DBTP = -1.0
LOOP_LUFS_HOME = -24.0


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
@dataclass
class Layer:
    buf: np.ndarray
    gain: float = 1.0
    offset_s: float = 0.0
    pan: float = 0.0          # -1 left .. +1 right (only used when stereo)
    label: str = ""


@dataclass
class Render:
    buf: np.ndarray
    fs: int = FS
    stereo: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return len(self.buf) / self.fs


def mix(layers: list[Layer], fs: int = FS, stereo: bool = False) -> np.ndarray:
    if not layers:
        return np.zeros(int(fs * 0.1))
    total = 0
    for L in layers:
        off = int(round(L.offset_s * fs))
        total = max(total, off + len(L.buf))
    if total == 0:
        return np.zeros(int(fs * 0.1))
    if not stereo:
        out = np.zeros(total)
        for L in layers:
            off = int(round(L.offset_s * fs))
            out[off:off + len(L.buf)] += L.buf * L.gain
        return out
    out = np.zeros((total, 2))
    for L in layers:
        off = int(round(L.offset_s * fs))
        a = (float(np.clip(L.pan, -1.0, 1.0)) + 1.0) * np.pi / 4.0
        gl, gr = np.cos(a), np.sin(a)
        seg = L.buf * L.gain
        out[off:off + len(seg), 0] += seg * gl * np.sqrt(2.0)
        out[off:off + len(seg), 1] += seg * gr * np.sqrt(2.0)
    return out


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    return np.stack([x, x], axis=1)


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * np.log10(p) if p > 0 else -np.inf


def true_peak_dbtp(x: np.ndarray, factor: int = 4) -> float:
    y = signal.resample_poly(np.asarray(x, dtype=np.float64), factor, 1, axis=0)
    return peak_dbfs(y)


def integrated_lufs(x: np.ndarray, fs: int = FS) -> float | None:
    try:
        import pyloudnorm as pyln
    except ImportError:
        return None
    if len(x) / fs < 0.4:          # gating blocks make short clips meaningless
        return None
    meter = pyln.Meter(fs)
    try:
        return float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))
    except Exception:
        return None


# --------------------------------------------------------------------------
# master chain
# --------------------------------------------------------------------------
def normalize_peak(x: np.ndarray, target_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    p = peak_dbfs(x)
    if not np.isfinite(p):
        return x
    return x * (10.0 ** ((target_dbfs - p) / 20.0))


def normalize_lufs(x: np.ndarray, fs: int = FS, target: float = LOOP_LUFS_HOME) -> np.ndarray:
    lufs = integrated_lufs(x, fs)
    if lufs is None:
        return normalize_peak(x, -12.0)
    y = x * (10.0 ** ((target - lufs) / 20.0))
    tp = true_peak_dbtp(y)
    if tp > TRUE_PEAK_CEIL_DBTP:
        y *= 10.0 ** ((TRUE_PEAK_CEIL_DBTP - tp) / 20.0)
    return y


def master(x: np.ndarray, fs: int = FS, loop: bool = False,
           target_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    duration = y.shape[0] / fs
    if duration < 0.5:
        # A 6 Hz one-pole removes the *local* mean, which on a 35 Hz kick inside
        # a 200 ms file is the kick itself. For short one-shots remove the true
        # global DC offset instead and leave the waveform intact.
        y = y - y.mean(axis=0, keepdims=True)
    elif not loop:
        # A loop's DC was removed circularly by the family; a zero-state blocker
        # here would leave a step at the seam the family cannot compensate for.
        y = dsp.dc_block(y, fs)
    if not loop:
        # A loop must stay seamless, so it gets no edge fade at all; the family
        # is responsible for its own crossfade.
        y = dsp.fade_asym(y, fs, in_ms=0.4)
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 0.95:
        y = y * (0.95 / peak)      # keep soft_clip a no-op instead of shaping hot renders
    y = fx.soft_clip(y, 0.995)
    # soft_clip can reintroduce a small mean shift on asymmetric waveforms
    y = y - y.mean(axis=0, keepdims=True)
    y = normalize_lufs(y, fs) if loop else normalize_peak(y, target_dbfs)
    tp = true_peak_dbtp(y)
    if tp > TRUE_PEAK_CEIL_DBTP:
        y *= 10.0 ** ((TRUE_PEAK_CEIL_DBTP - tp) / 20.0)
    return y


def _tpdf_dither(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) - rng.random(n)) / 32768.0


def save_wav(path: str | Path, x: np.ndarray, fs: int = FS, bits: int = 24,
             dither: bool = False, rng: np.random.Generator | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(x, dtype=np.float64)
    if bits == 16:
        if dither:
            rng = rng or np.random.default_rng(0)
            if y.ndim == 1:
                y = y + _tpdf_dither(len(y), rng)
            else:
                y = y + _tpdf_dither(y.shape[0], rng)[:, None]
        sf.write(str(path), np.clip(y, -1.0, 1.0), fs, subtype="PCM_16")
    else:
        sf.write(str(path), np.clip(y, -1.0, 1.0), fs, subtype="PCM_24")
    return path


# --------------------------------------------------------------------------
# entry point used by every family and by the batch driver
# --------------------------------------------------------------------------
def render_family(gen, params: dict, seed: int, fs: int = FS,
                  stereo: bool | None = None) -> tuple[np.ndarray, dict]:
    """Call a family's generate(); returns (mastered buffer, metadata)."""
    rng = dsp.rng_for(seed)
    out = gen(params=params, rng=rng, fs=fs)
    if isinstance(out, Render):
        buf, meta = out.buf, dict(out.meta)
        stereo = out.stereo if stereo is None else stereo
    elif isinstance(out, list):
        buf, meta, stereo = mix(out, fs=fs, stereo=bool(stereo)), {}, bool(stereo)
    else:
        buf, meta = np.asarray(out), {}
        stereo = bool(buf.ndim == 2) if stereo is None else stereo
    if stereo and buf.ndim == 1:
        buf = to_stereo(buf)
    buf = np.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0)
    loop = bool(meta.get("loop", False))
    buf = master(buf, fs=fs, loop=loop)
    meta.update(
        seed=int(seed),
        seconds=round(len(buf) / fs, 4),
        peak_dbfs=round(peak_dbfs(buf), 2),
        true_peak_dbtp=round(true_peak_dbtp(buf), 2),
        lufs=integrated_lufs(buf, fs),
        channels=2 if buf.ndim == 2 else 1,
    )
    return buf, meta
