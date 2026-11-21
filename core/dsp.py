"""Synthesis primitives.

Everything is deterministic: a family receives an np.random.Generator and derives
all randomness from it, so (family, params, seed) -> bit-identical audio.

References (see briefs/RESEARCH.md):
  polyBLEP/BLEP       Stilson & Smith 1996; Valimaki 2005
  modal synthesis     van den Doel & Pai 2003; O'Brien et al. 2002
  Karplus-Strong      Jaffe & Smith 1983
  FM                  Chowning 1973
  granular            Truax 1988
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal

FS = 48_000


# --------------------------------------------------------------------------
# frequency helpers
# --------------------------------------------------------------------------
def const(n: int, f: float) -> np.ndarray:
    return np.full(int(n), float(f), dtype=np.float64)


def sweep(n: int, f0: float, f1: float, kind: str = "exp") -> np.ndarray:
    """Frequency curve over n samples. 'exp' is the musically useful one."""
    t = np.linspace(0.0, 1.0, int(n))
    if kind == "exp" and f0 > 0 and f1 > 0:
        return f0 * (f1 / f0) ** t
    return f0 + (f1 - f0) * t


def rng_for(seed: int) -> np.random.Generator:
    """One canonical way to turn an int seed into a generator."""
    return np.random.Generator(np.random.PCG64(int(seed)))


# --------------------------------------------------------------------------
# oscillators
# --------------------------------------------------------------------------
def _phase(f: np.ndarray, fs: int = FS, phase0: float = 0.0):
    if np.ndim(f) == 0:
        # Silent footgun: a scalar frequency has no length, and returning a
        # 1-sample array makes partial stacks vanish without an error.
        raise ValueError(
            "oscillator frequency must be an array: pass np.full(n, f0) "
            "or use dsp.const(n, f0)"
        )
    f = np.asarray(f, dtype=np.float64)
    dt = f / fs
    ph = (phase0 + np.cumsum(dt)) % 1.0
    return ph, dt


def polyblep(ph: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """2-point polynomial band-limited step residual."""
    out = np.zeros_like(ph)
    d = np.maximum(np.abs(dt), 1e-12)
    lo = ph < d
    if lo.any():
        x = ph[lo] / d[lo]
        out[lo] = x + x - x * x - 1.0
    hi = ph > 1.0 - d
    if hi.any():
        x = (ph[hi] - 1.0) / d[hi]
        out[hi] = x * x + x + x + 1.0
    return out


def saw(f, fs: int = FS, phase0: float = 0.0) -> np.ndarray:
    ph, dt = _phase(f, fs, phase0)
    return 2.0 * ph - 1.0 - polyblep(ph, dt)


def square(f, fs: int = FS, duty: float = 0.5, phase0: float = 0.0) -> np.ndarray:
    ph, dt = _phase(f, fs, phase0)
    s = np.where(ph < duty, 1.0, -1.0)
    s = s + polyblep(ph, dt)
    s = s - polyblep((ph + 1.0 - duty) % 1.0, dt)
    return s


def sine(f, fs: int = FS, phase0: float = 0.0) -> np.ndarray:
    ph, _ = _phase(f, fs, phase0)
    return np.sin(2.0 * np.pi * ph)


def tri(f, fs: int = FS, phase0: float = 0.0, harmonics: int = 9) -> np.ndarray:
    """Band-limited triangle, additive so each partial can be Nyquist-checked."""
    ph, _ = _phase(f, fs, phase0)
    f_arr = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(ph)
    for k in range(1, harmonics + 1):
        n = 2 * k - 1
        safe = (n * f_arr) < fs * 0.45
        term = np.zeros_like(ph)
        term[safe] = ((-1.0) ** (k + 1)) * np.sin(2.0 * np.pi * n * ph[safe]) / (n * n)
        out += term
    return out * (8.0 / np.pi ** 2)


def fm(fc, ratio: float, index_env: np.ndarray, fs: int = FS) -> np.ndarray:
    """FM pair with a time-varying modulation index.

    Carson's rule: upper sideband reaches fc*(1 + ratio*(1+index)). The index is
    clamped so the outermost sideband stays below Nyquist instead of folding back.
    """
    fc_a = np.asarray(fc, dtype=np.float64)
    fm_rate = fc_a * float(ratio)
    allowed = np.maximum(0.0, (fs * 0.46 - fc_a) / np.maximum(fm_rate, 1e-9) - 1.0)
    idx = np.minimum(np.asarray(index_env, dtype=np.float64), allowed)
    phc, _ = _phase(fc_a, fs)
    phm, _ = _phase(fm_rate, fs)
    return np.sin(2.0 * np.pi * phc + idx * np.sin(2.0 * np.pi * phm))


def karplus(freq: float, n: int, fs: int = FS, decay: float = 0.9995,
            bright: float = 0.5, exc: np.ndarray | None = None,
            rng: np.random.Generator | None = None, tune: bool = True) -> np.ndarray:
    """Extended Karplus-Strong.

    Block-vectorised: within a block of N samples every recurrence input comes
    from before the block, so the whole string renders without a per-sample loop.

    tune=True makes the pitch exact. The two-tap loop filter (a, b) with a+b=1
    has a group delay of b samples, so the true loop delay is N+b. Solving
    N+b = fs/f removes the cents-flat error of integer-N tuning (~4 cents at C5).
    """
    if tune:
        D = fs / float(freq)
        N = max(2, int(np.floor(D)))
        b = D - N
        if b < 0.02:                       # keep both taps meaningful
            N -= 1
            b += 1.0
        b = float(np.clip(b, 0.02, 0.98))
    else:
        N = max(2, int(round(fs / float(freq))))
        b = float(np.clip(bright, 0.0, 1.0))
    n = int(n)
    buf = np.zeros(n + N + 1)
    if exc is None:
        if rng is None:
            raise ValueError("karplus needs exc or rng")
        exc = rng.standard_normal(N) * np.hanning(N)
    m = min(N, n)
    buf[1:1 + m] = np.asarray(exc)[:m]
    a = 1.0 - b
    for s in range(N, n, N):
        e = min(s + N, n)
        L = e - s
        y1 = buf[s + 1 - N:s + 1 - N + L]   # y[n-N]
        y2 = buf[s - N:s - N + L]           # y[n-N-1]
        buf[s + 1:e + 1] = decay * (a * y1 + b * y2)
    return buf[1:n + 1]


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------
def white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(int(n))


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    w = rng.standard_normal(n)
    spec = np.fft.rfft(w)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    spec /= np.sqrt(f)
    spec[0] = 0.0            # dc bin was being scaled by 1/sqrt(1) and left in
    out = np.fft.irfft(spec, n)
    return out / (np.max(np.abs(out)) + 1e-12)


def brown(n: int, rng: np.random.Generator) -> np.ndarray:
    # detrend() RETURNS the detrended signal; subtracting it left only the ramp,
    # so every seed produced the same straight line. Assign, don't subtract.
    out = signal.detrend(np.cumsum(rng.standard_normal(int(n))))
    return out / (np.max(np.abs(out)) + 1e-12)


def velvet(n: int, period: int = 48, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sparse +/-1 impulses, one per period on average (Valimaki 2017 tail source)."""
    rng = rng or np.random.default_rng(0)
    out = np.zeros(int(n))
    idx = rng.integers(0, max(1, period), size=int(n // max(1, period)) + 1)
    idx = idx + np.arange(idx.size) * max(1, period)
    idx = idx[idx < n]
    out[idx] = rng.choice([-1.0, 1.0], size=idx.size)
    return out


# --------------------------------------------------------------------------
# envelopes
# --------------------------------------------------------------------------
def env_exp(n: int, fs: int = FS, attack: float = 0.002,
            tau: float = 0.2, curve: float = 1.0) -> np.ndarray:
    t = np.arange(int(n)) / fs
    a = np.clip(t / max(attack, 1e-6), 0.0, 1.0)
    if curve != 1.0:
        a = a ** curve
    return a * np.exp(-np.maximum(t - 0, 0) / max(tau, 1e-6))


def env_adsr(n: int, fs: int = FS, a: float = 0.003, d: float = 0.08,
             s: float = 0.6, r: float = 0.2) -> np.ndarray:
    n = int(n)
    t = np.arange(n) / fs
    total = n / fs
    env = np.zeros(n)
    adsr = [a, d, max(0.0, total - a - d - r), r]
    marks = np.cumsum(adsr)
    for i in range(n):
        ti = t[i]
        if ti < marks[0]:
            env[i] = ti / max(a, 1e-6)
        elif ti < marks[1]:
            k = (ti - marks[0]) / max(d, 1e-6)
            env[i] = 1.0 + (s - 1.0) * k
        elif ti < marks[2]:
            env[i] = s
        else:
            k = (ti - marks[2]) / max(r, 1e-6)
            env[i] = s * max(0.0, 1.0 - k)
    return env


def dc_block(x: np.ndarray, fs: int = FS, fc: float = 6.0) -> np.ndarray:
    """One-pole DC blocker. Asymmetric square/pulse waves carry real DC offset."""
    r = float(np.exp(-2.0 * np.pi * fc / fs))
    b = np.array([1.0, -1.0])
    a = np.array([1.0, -r])
    return signal.lfilter(b, a, x, axis=0)


def fade(x: np.ndarray, fs: int = FS, ms: float = 5.0) -> np.ndarray:
    """Mandatory anti-click fade. Every rendered file goes through this."""
    m = int(fs * ms / 1000.0)
    if m <= 1 or x.size < 2 * m:
        return x
    y = np.asarray(x, dtype=np.float64).copy()
    r = np.linspace(0.0, 1.0, m)
    if y.ndim > 1:
        r = r[:, None]
    y[:m] *= r
    y[-m:] *= r[::-1]
    return y


def fade_asym(x: np.ndarray, fs: int = FS, in_ms: float = 0.4,
              out_ms: float | None = None) -> np.ndarray:
    """Asymmetric fade: keep the attack intact, taper only the tail.

    A symmetric 4 ms fade eats the first 4 ms of a drum hit, which is precisely
    where its click lives. The start needs microseconds to kill the DC step; the
    end needs milliseconds to kill the cutoff click.
    """
    y = np.asarray(x, dtype=np.float64).copy()
    n = y.size
    if n < 4:
        return y
    mi = max(1, int(fs * in_ms / 1000.0))
    mi = min(mi, n // 4)
    if out_ms is None:
        out_ms = max(2.0, min(8.0, 1000.0 * n / fs * 0.15))
    mo = max(1, int(fs * out_ms / 1000.0))
    mo = min(mo, n // 4)
    ri = np.linspace(0.0, 1.0, mi)
    ro = np.linspace(1.0, 0.0, mo)
    if y.ndim > 1:                    # stereo beds: ramps must be per-channel
        ri = ri[:, None]
        ro = ro[:, None]
    if mi > 1:
        y[:mi] *= ri
    if mo > 1:
        y[-mo:] *= ro
    return y


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def _sos(kind: str, fs: int, f: float, order: int, q: float):
    nyq = fs / 2.0
    if kind == "lp":
        wn = min(max(f, 20.0), nyq * 0.99)
        return signal.butter(order, wn, btype="low", fs=fs, output="sos")
    if kind == "hp":
        wn = min(max(f, 10.0), nyq * 0.99)
        return signal.butter(order, wn, btype="high", fs=fs, output="sos")
    lo = max(f / max(q, 0.1), 20.0)
    hi = min(f * max(q, 0.1), nyq * 0.98)
    if hi <= lo * 1.05:
        hi = min(lo * 1.05, nyq * 0.98)
    return signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def lp(x, f: float, fs: int = FS, order: int = 4) -> np.ndarray:
    return signal.sosfilt(_sos("lp", fs, float(f), order, 1.0), x, axis=0)


def hp(x, f: float, fs: int = FS, order: int = 4) -> np.ndarray:
    return signal.sosfilt(_sos("hp", fs, float(f), order, 1.0), x, axis=0)


def bp(x, f: float, fs: int = FS, order: int = 2, q: float = 1.4) -> np.ndarray:
    return signal.sosfilt(_sos("bp", fs, float(f), order, q), x, axis=0)


def moving_bp(x: np.ndarray, f_curve: np.ndarray, fs: int = FS,
              q: float = 1.4, bands: int = 16) -> np.ndarray:
    """Bandpass whose centre follows a curve (air, whooshes, risers).

    Implemented as a band split with a per-sample Gaussian crossfade, NOT as
    overlap-add chunks. Chunked filtering restarts each chunk's filter state,
    which injects a comb at fs/chunk (a 47 Hz buzz at 1024-sample chunks).
    Every band is filtered over the whole signal, so there is no seam anywhere.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    fc = np.asarray(f_curve, dtype=np.float64)
    if fc.size != n:
        fc = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, fc.size), fc)
    fc = np.clip(fc, 30.0, fs * 0.47)

    lo_edge, hi_edge = max(30.0, fs / 2000.0), fs * 0.48
    edges = np.geomspace(lo_edge, hi_edge, bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    sigma = max(0.12, 0.90 / max(q, 0.3))        # log-frequency half-width
    log_fc = np.log(fc)

    out = np.zeros(n)
    for i in range(bands):
        sos = signal.butter(2, [edges[i], edges[i + 1]], btype="band",
                            fs=fs, output="sos")
        w = np.exp(-0.5 * ((log_fc - np.log(centers[i])) / sigma) ** 2)
        if w.max() < 1e-3:
            continue                              # the curve never visits this band
        out += signal.sosfilt(sos, x) * w
    return out


# --------------------------------------------------------------------------
# modal synthesis (impacts, booms, shatter)
# --------------------------------------------------------------------------
MATERIALS = {
    # name: (mode ratios, relative decay multipliers, inharmonicity spread)
    "metal":   (np.array([1.0, 2.76, 5.40, 8.93, 13.34]),
                np.array([1.00, 0.62, 0.45, 0.34, 0.25]), 0.004),
    "glass":   (np.array([1.0, 2.32, 4.25, 7.10, 10.60]),
                np.array([0.85, 0.55, 0.40, 0.30, 0.20]), 0.003),
    "wood":    (np.array([1.0, 1.94, 3.12, 4.60]),
                np.array([0.70, 0.42, 0.26, 0.16]), 0.010),
    "concrete": (np.array([1.0, 1.63, 2.71, 4.02, 6.10]),
                 np.array([0.45, 0.32, 0.22, 0.15, 0.10]), 0.020),
    "ceramic": (np.array([1.0, 2.05, 3.71, 5.89]),
                np.array([0.60, 0.40, 0.27, 0.18]), 0.007),
    "plastic": (np.array([1.0, 2.41, 4.05]),
                np.array([0.35, 0.20, 0.12]), 0.014),
}


def modal_bank(f0: float, material: str, tau0: float = 0.35,
               size: float = 1.0, rng: np.random.Generator | None = None,
               jitter: float = 0.0) -> list[tuple[float, float, float]]:
    """Build a resonator bank: [(freq_hz, tau_s, gain), ...]."""
    ratios, decays, spread = MATERIALS[material]
    rng = rng or np.random.default_rng(0)
    modes = []
    for i, (r, dm) in enumerate(zip(ratios, decays)):
        f = f0 * r * (1.0 + spread * i)
        if jitter:
            f *= 1.0 + rng.uniform(-jitter, jitter)
        # bigger objects ring lower and longer
        f = f / max(size, 0.05)
        modes.append((float(f), float(tau0 * dm * size), float(1.0 / (1.0 + i))))
    return modes


def modal(x: np.ndarray, modes: list[tuple[float, float, float]],
          fs: int = FS) -> np.ndarray:
    """Drive a bank of two-pole resonators with x (O'Brien: contact accel as input)."""
    y = np.zeros_like(x)
    for f, tau, g in modes:
        if f <= 20.0 or f >= fs / 2.0 - 200.0 or tau <= 0:
            continue
        r = float(np.exp(-1.0 / (max(tau, 1e-4) * fs)))
        b1 = 2.0 * r * float(np.cos(2.0 * np.pi * f / fs))
        b2 = -r * r
        # A 2-pole resonator's impulse-response peak scales like 1/(1-r). Without
        # this normalisation a long-tau mode returns amplitudes in the hundreds,
        # which then overdrives every downstream nonlinearity.
        y += (g * (1.0 - r)) * signal.lfilter([1.0], [1.0, -b1, -b2], x)
    return y


def contact_burst(n: int, fs: int = FS, hardness: float = 0.7,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Excitation for a modal bank. Hard hits approach an impulse, soft hits smear."""
    rng = rng or np.random.default_rng(0)
    n = int(n)
    noise = rng.standard_normal(n)
    cutoff = 800.0 + 12_000.0 * float(np.clip(hardness, 0.0, 1.0))
    burst = lp(noise, cutoff, fs, order=2)
    return burst * env_exp(n, fs, attack=0.0005, tau=0.002 + 0.02 * (1.0 - hardness))


# --------------------------------------------------------------------------
# granular
# --------------------------------------------------------------------------
def moving_lp(x: np.ndarray, f_curve: np.ndarray, fs: int = FS,
              block: int = 128) -> np.ndarray:
    """Lowpass whose cutoff follows a curve, with filter state carried across blocks.

    Restarting the filter at each block edge is what puts a buzz at fs/block into
    the signal; carrying `zi` is what keeps it clean. (A bandpass cannot stand in
    for this on bass material: it removes the fundamental.)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    fc_arr = np.asarray(f_curve, dtype=np.float64)
    zi = None
    for s in range(0, len(x), block):
        e = min(s + block, len(x))
        fc = float(np.clip(fc_arr[min(s, len(fc_arr) - 1)], 20.0, fs * 0.45))
        sos = signal.butter(2, fc, btype="low", fs=fs, output="sos")
        seg, zi = signal.sosfilt(sos, x[s:e], zi=zi)
        y[s:e] = seg
    return y


def grain(buf: np.ndarray, start: int, length: int, rate: float,
          fs: int = FS, shape: str = "hann") -> np.ndarray:
    """One grain. rate is a playback-rate (pitch) multiplier."""
    idx = np.arange(int(length)) * float(rate)
    src = start + idx
    valid = (src >= 0) & (src < len(buf) - 1)
    out = np.zeros(int(length))
    if valid.any():
        s = np.interp(src[valid], np.arange(len(buf)), buf)
        win = {"hann": np.hanning, "gauss": lambda k: signal.windows.gaussian(k, k / 6.0)}.get(shape, np.hanning)(int(length))
        out[valid] = s * win[valid]
    return out


def granulate(src: np.ndarray, n_out: int, density_hz: float, fs: int = FS,
              rng: np.random.Generator | None = None,
              grain_ms: tuple[float, float] = (20.0, 90.0),
              pitch: tuple[float, float] = (1.0, 1.0),
              position: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Sparse grain cloud over a buffer we synthesised ourselves. Never fed samples."""
    rng = rng or np.random.default_rng(0)
    out = np.zeros(int(n_out))
    period = max(1, int(fs / max(1e-3, density_hz)))
    for onset in range(0, int(n_out), period):
        if rng.random() > 0.85:
            continue
        gl = int(fs * rng.uniform(*grain_ms) / 1000.0)
        if gl < 4:
            continue
        p = rng.uniform(*position)
        start = int(p * max(1, len(src) - gl - 2))
        g = grain(src, start, gl, rng.uniform(*pitch), fs)
        e = min(len(out), onset + g.size)
        out[onset:e] += g[:e - onset]
    return out


# --------------------------------------------------------------------------
# scale helpers (12-key outputs)
# --------------------------------------------------------------------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((float(m) - 69.0) / 12.0)


def note_hz(note: str, octave: int = 4) -> float:
    idx = NOTE_NAMES.index(note)
    return midi_to_hz(12 * (octave + 1) + idx)
