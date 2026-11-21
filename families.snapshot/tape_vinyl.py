"""Family: tape_vinyl. Tape transport artefacts and vinyl surface noise.

This is a character/texture family, not a synth family.  Two sources of material
are synthesised here (never loaded):

  _music_src   a detuned saw chord with two Karplus plucks -> "something playing"
  _stab        a bright saw stab with a noise transient -> scratch material

and two transports drive them:

  _play(src, rate)      signed playback-rate curve (cumsum of rate).  Rate is in
                        samples per sample, so negative rates play backwards.
  fx.tape               the analog character stage (hysteresis, wow, flutter,
                        gap-loss LP, hiss) that every tape render goes through.

CORE ISSUE (worked around, see the .md): `fx.scratch` integrates |rate_curve|,
so it cannot represent a backspin or a baby-scratch reversal - a negative rate
plays forward.  `_play` here integrates the signed curve instead.

Loop policy: the two loop archetypes set meta={"loop": True} and are delivered
mono.  They are synthesised as two decorrelated channels (that is the texture
that makes a hiss/crackle bed sit wide when a game engine widens it) and summed
to mono, because `core.render.master -> dsp.fade` cannot handle a 2-D buffer:
`y[:m] *= r` raises a broadcast error on (n, 2).  That is a CORE ISSUE in the
.md.  `master` also applies its mandatory 4 ms anti-click fade to loops, so a
loop seam always has that tiny dip; also a render-stage property, not something
a family can remove.
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "tape_vinyl"
DESCRIPTORS = ["crackle", "scratch", "backspin", "tapestop", "hiss", "vinyl"]
ARCHETYPES = ("tape_stop", "tape_start", "tape_degrade", "tape_hiss",
              "tape_hiss_loop", "wobble", "scratch", "backspin",
              "vinyl_crackle_loop", "vinyl_pop")
DESCR_OF = {"tape_stop": "tapestop", "tape_start": "tapestop", "wobble": "tapestop",
            "tape_degrade": "hiss", "tape_hiss": "hiss", "tape_hiss_loop": "hiss",
            "scratch": "scratch", "backspin": "backspin",
            "vinyl_crackle_loop": "crackle", "vinyl_pop": "vinyl"}


def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    a = float(np.max(np.abs(x))) if np.size(x) else 0.0
    if a <= 1e-12:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) * (peak / a)


def _bed(n: int, rng: np.random.Generator, kind: str) -> np.ndarray:
    """Coloured noise floor: the difference between tape hiss and vinyl surface."""
    if kind == "brown":
        return dsp.brown(n, rng) * 0.7
    if kind == "white":
        return dsp.white(n, rng) * 0.22
    return dsp.pink(n, rng)


def _release(x: np.ndarray, fs: int, ms: float = 20.0) -> np.ndarray:
    m = min(int(fs * ms / 1000.0), max(2, len(x) // 10))
    if m > 2:
        x = x.copy()
        x[-m:] *= np.cos(np.linspace(0.0, np.pi / 2.0, m)) ** 2
    return x


def _loopify(x: np.ndarray, fs: int, ms: float = 180.0) -> np.ndarray:
    """Equal-length crossfade of the tail into the head; trims the tail."""
    x = np.asarray(x, dtype=np.float64)
    m = min(int(fs * ms / 1000.0), max(4, len(x) // 4))
    r = np.linspace(0.0, 1.0, m)
    out = x.copy()
    out[:m] = x[:m] * r + x[-m:] * (1.0 - r)
    return out[:-m]


def _play(src: np.ndarray, rate: np.ndarray, start: float = 0.0) -> np.ndarray:
    """Signed-rate playback of our own buffer. rate in samples/sample."""
    rate = np.asarray(rate, dtype=np.float64)
    # 2 ms smoothing: kinks in a hand-drawn rate curve otherwise chirp
    m = max(1, int(0.002 * FS))
    if rate.size > 2 * m:
        rate = np.convolve(rate, np.ones(m) / m, mode="same")
    idx = float(start) + np.cumsum(rate)
    n = len(src)
    valid = (idx >= 0.0) & (idx <= n - 1.001)
    out = np.interp(np.clip(idx, 0.0, n - 1.001), np.arange(n), src)
    if not valid.all():
        k = max(8, int(0.003 * FS))
        mask = np.convolve(valid.astype(np.float64), np.ones(k) / k, mode="same")
        out = out * np.clip(mask, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def _music_src(dur: float, fs: int, rng: np.random.Generator, root: float,
               bright: float = 1.0) -> np.ndarray:
    n = int(fs * dur)
    t = np.arange(n) / fs
    y = np.zeros(n)
    for i, r in enumerate((1.0, 1.26, 1.5, 2.0)):
        f = root * r * (1.0 + 0.0015 * i)
        vib = 1.0 + 0.0012 * np.sin(2.0 * np.pi * (4.5 + 0.7 * i) * t)
        y += dsp.saw(f * vib, fs) * (0.65 ** i)
    y *= dsp.env_adsr(n, fs, a=0.012, d=0.08, s=0.82, r=min(0.15, dur * 0.2))
    y = dsp.lp(y, 3800.0 * bright, fs, order=2)
    for j, frac in enumerate((0.0, 0.45)):
        off = int(frac * n)
        k = min(n - off, int(fs * 0.5))
        if k > 128:
            pl = dsp.karplus(root * 2.0, k, fs, decay=0.996, bright=0.55, rng=rng)
            y[off:off + k] += _norm(pl, 0.45 if j == 0 else 0.3)
    return _norm(y, 0.9)


def _perc_src(dur: float, fs: int, rng: np.random.Generator, root: float) -> np.ndarray:
    """A small beat: kick, hats, backbeat snare, low drone under it."""
    n = int(fs * dur)
    y = np.zeros(n)
    for pos in range(0, n, max(1, int(fs * 0.5))):
        k = max(64, min(n - pos, int(fs * 0.25)))
        sub = dsp.sine(dsp.sweep(k, 110.0, 45.0, "exp"), fs)
        sub *= dsp.env_exp(k, fs, attack=0.001, tau=0.06)
        _add_at(y, _norm(sub, 0.7), int(pos))
    for pos in range(int(fs * 0.125), n, max(1, int(fs * 0.25))):
        k = max(16, min(n - pos, int(fs * 0.04)))
        h = dsp.hp(dsp.white(k, rng), 6000.0, fs, order=2)
        h *= dsp.env_exp(k, fs, attack=0.0005, tau=0.012)
        _add_at(y, _norm(h, 0.2), int(pos))
    for pos in range(int(fs * 0.5), n, max(1, int(fs * 1.0))):
        k = max(64, min(n - pos, int(fs * 0.15)))
        sn = dsp.bp(dsp.white(k, rng), 1800.0, fs, order=2, q=1.0)
        sn *= dsp.env_exp(k, fs, attack=0.001, tau=0.05)
        sn += dsp.sine(dsp.const(k, 185.0), fs) * dsp.env_exp(k, fs, attack=0.001, tau=0.03) * 0.4
        _add_at(y, _norm(sn, 0.42), int(pos))
    t = np.arange(n) / fs
    y += dsp.saw(dsp.const(n, root * 0.5), fs) * 0.15 * dsp.env_adsr(
        n, fs, a=0.01, d=0.1, s=0.9, r=0.1)
    return _norm(y, 0.9)


def _bass_src(dur: float, fs: int, rng: np.random.Generator, root: float) -> np.ndarray:
    """A pumping bass line: one low saw, subdivided by an amplitude gate."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    f = root * 0.5 * (1.0 + 0.004 * np.sin(2.0 * np.pi * 5.5 * t))
    y = dsp.saw(f, fs) * 0.7 + dsp.sine(f, fs) * 0.5
    y *= dsp.env_adsr(n, fs, a=0.006, d=0.15, s=0.7, r=0.2)
    y = dsp.lp(y, 900.0, fs, order=2)
    rate = rng.uniform(2.0, 4.0)
    gate = (np.sin(2.0 * np.pi * rate * t + 1.2) > -0.2).astype(np.float64)
    m = max(4, int(fs * 0.006))
    gate = np.convolve(gate, np.hanning(m) / np.hanning(m).sum(), mode="same")
    return _norm(y * (0.45 + 0.65 * gate), 0.9)


def _vox_src(dur: float, fs: int, rng: np.random.Generator, root: float) -> np.ndarray:
    """Formant-swept 'voice' record: saw + breath through two moving formants."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    f = root * (1.0 + 0.010 * np.sin(2.0 * np.pi * rng.uniform(4.5, 6.5) * t))
    src = dsp.saw(f, fs) * 0.6 + dsp.white(n, rng) * 0.08
    f1 = 650.0 * (1.0 + 0.10 * np.sin(2.0 * np.pi * 0.7 * t))
    f2 = 1150.0 * (1.0 + 0.12 * np.sin(2.0 * np.pi * 0.5 * t + 1.0))
    y = (dsp.moving_bp(src, f1, fs, q=3.0, bands=14) * 0.8
         + dsp.moving_bp(src, f2, fs, q=3.0, bands=14) * 0.5)
    y *= dsp.env_adsr(n, fs, a=0.02, d=0.2, s=0.8, r=0.2)
    return _norm(y, 0.9)


def _noise_record(fs: int, rng: np.random.Generator, dur: float = 1.6) -> np.ndarray:
    """Bright moving-band noise 'record' for scratching noise instead of a stab."""
    n = int(fs * dur)
    y = dsp.moving_bp(dsp.white(n, rng), dsp.sweep(n, 600.0, 4200.0, "exp"),
                      fs, q=1.0, bands=12)
    y *= dsp.env_adsr(n, fs, a=0.01, d=0.3, s=0.6, r=0.3)
    return _norm(y, 0.85)


def _make_src(kind: str, dur: float, fs: int, rng: np.random.Generator,
              root: float, bright: float) -> np.ndarray:
    if kind == "perc":
        return _perc_src(dur, fs, rng, root)
    if kind == "bass":
        return _bass_src(dur, fs, rng, root)
    if kind == "vox":
        return _vox_src(dur, fs, rng, root)
    return _music_src(dur, fs, rng, root, bright)


def _stab(fs: int, rng: np.random.Generator, root: float, dur: float = 1.6) -> np.ndarray:
    n = int(fs * dur)
    t = np.arange(n) / fs
    y = np.zeros(n)
    for i, r in enumerate((1.0, 1.5, 2.0, 2.52)):
        f = root * r * (1.0 + 0.0015 * np.sin(2.0 * np.pi * (5.0 + i) * t))
        y += dsp.saw(f, fs) * (0.7 ** i)
    y *= dsp.env_exp(n, fs, attack=0.002, tau=dur * 0.45)
    y = dsp.lp(y, 6000.0, fs, order=2)
    k = int(fs * 0.02)
    tr = dsp.hp(dsp.white(k, rng), 2500.0, fs, order=2)
    tr *= dsp.env_exp(k, fs, attack=0.0003, tau=0.004)
    y[:k] += tr * 0.5
    return _norm(y, 0.9)


def _scratch_src(kind: str, fs: int, rng: np.random.Generator, root: float,
                 dur: float) -> np.ndarray:
    if kind == "vox":
        return _vox_src(dur, fs, rng, root)
    if kind == "noise":
        return _noise_record(fs, rng, dur)
    return _stab(fs, rng, root, dur=dur)


def _pop(fs: int, rng: np.random.Generator, freq: float, amp: float) -> np.ndarray:
    """One vinyl pop: damped 4 ms resonance + a small low thump."""
    k = int(fs * 0.035)
    t = np.arange(k) / fs
    ring = np.exp(-t / 0.0035) * np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    thump = 0.6 * np.exp(-t / 0.025) * np.sin(2.0 * np.pi * 70.0 * t)
    y = amp * (ring + thump)
    y[:4] += amp * np.array([1.0, -0.6, 0.35, -0.2])
    return y


def _add_at(out: np.ndarray, y: np.ndarray, pos: int) -> None:
    e = min(len(out), pos + len(y))
    if pos < len(out):
        out[pos:e] += y[:e - pos]


# --------------------------------------------------------------------------
# parameter spread
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    arch = str(rng.choice(ARCHETYPES,
                          p=[0.13, 0.10, 0.10, 0.10, 0.10, 0.08,
                             0.14, 0.09, 0.10, 0.06]))
    p = dict(archetype=arch,
             root=float(2.0 ** rng.uniform(np.log(90.0), np.log(400.0))),
             src_kind=str(rng.choice(["chord", "perc", "bass", "vox"],
                                     p=[0.40, 0.25, 0.20, 0.15])),
             noise_kind=str(rng.choice(["pink", "brown", "white"],
                                       p=[0.55, 0.30, 0.15])),
             drive=float(rng.uniform(1.3, 2.6)),
             wow=float(rng.uniform(0.0006, 0.0040)),
             flutter=float(rng.uniform(0.0002, 0.0018)),
             bw=float(rng.uniform(6000.0, 12000.0)),
             bright=float(rng.uniform(0.6, 1.2)))

    if arch == "tape_stop":
        p.update(dur=float(rng.uniform(0.70, 2.00)),
                 hold=float(rng.uniform(0.20, 0.42)),
                 stopwin=float(rng.uniform(0.34, 0.52)),
                 exp=float(rng.uniform(1.2, 2.8)),
                 rate_end=float(rng.uniform(0.02, 0.09)))
    elif arch == "tape_start":
        p.update(dur=float(rng.uniform(0.40, 0.95)),
                 run=float(rng.uniform(0.12, 0.35)),
                 wob=float(rng.uniform(0.01, 0.05)))
    elif arch == "tape_degrade":
        p.update(dur=float(rng.uniform(1.5, 3.0)),
                 wob=float(rng.uniform(0.004, 0.011)),
                 wob_hz=float(rng.uniform(0.4, 1.8)),
                 drops=int(rng.integers(1, 4)))
    elif arch == "tape_hiss":
        p.update(dur=float(rng.uniform(0.6, 1.5)),
                 hiss_db=float(rng.uniform(-24.0, -12.0)),
                 bw=float(rng.uniform(7000.0, 13000.0)))
    elif arch == "tape_hiss_loop":
        p.update(dur=float(rng.uniform(2.0, 4.0)),
                 hiss_db=float(rng.uniform(-26.0, -16.0)),
                 bw=float(rng.uniform(7000.0, 12000.0)))
    elif arch == "wobble":
        p.update(dur=float(rng.uniform(1.5, 3.0)),
                 wob=float(rng.uniform(0.004, 0.012)),
                 wob_hz=float(rng.uniform(0.4, 2.0)))
    elif arch == "scratch":
        p.update(src_kind=str(rng.choice(["stab", "vox", "noise"],
                                         p=[0.58, 0.25, 0.17])),
                 dur=float(rng.uniform(0.38, 1.00)),
                 gesture=str(rng.choice(["baby", "chirp", "transform"],
                                        p=[0.45, 0.35, 0.20])),
                 speed=float(rng.uniform(0.8, 1.35)),
                 friction=float(rng.uniform(0.03, 0.08)),
                 crackle=float(rng.uniform(0.6, 3.0)))
    elif arch == "backspin":
        p.update(src_kind=str(rng.choice(["stab", "vox", "noise"],
                                         p=[0.58, 0.25, 0.17])),
                 dur=float(rng.uniform(0.60, 1.80)),
                 decel=float(rng.uniform(0.22, 0.50)),
                 spin=float(rng.uniform(1.2, 2.2)),
                 friction=float(rng.uniform(0.03, 0.08)))
    elif arch == "vinyl_crackle_loop":
        p.update(dur=float(rng.uniform(2.0, 5.0)),
                 crackle=float(rng.uniform(2.0, 12.0)),
                 rumble=float(rng.uniform(0.01, 0.04)),
                 hi=float(rng.uniform(6500.0, 9000.0)))
    else:  # vinyl_pop
        p.update(dur=float(rng.uniform(0.30, 0.80)),
                 pops=int(rng.integers(1, 4)),
                 pop_amp=float(rng.uniform(0.5, 0.95)),
                 hiss=float(rng.uniform(0.008, 0.03)))
    return p


# --------------------------------------------------------------------------
# tape archetypes
# --------------------------------------------------------------------------
def _tape_stop(p, rng, fs, n):
    dur = n / fs
    src = _make_src(p["src_kind"], min(3.0, dur * 1.25), fs, rng, p["root"], p["bright"])
    t = np.arange(n) / fs
    hold = dur * p["hold"]
    stop = dur * p["stopwin"]
    u = np.clip((t - hold) / stop, 0.0, 1.0)
    rate = (1.0 - u) ** p["exp"]
    rate = np.where(t <= hold, 1.0, rate)
    rate = np.where(t > hold + stop, p["rate_end"], rate)
    out = _play(src, rate, start=0.0)

    # the replay head loses high frequency as the motor stalls
    w = np.clip((t - hold) / stop, 0.0, 1.0) * 0.75
    out = out * (1.0 - w) + dsp.lp(out, 2200.0, fs, order=2) * w
    tail_from = hold + stop
    out *= np.where(t > tail_from,
                    np.exp(-np.maximum(t - tail_from, 0.0) / (dur * 0.08)), 1.0)
    out = fx.tape(out, fs=fs, drive=p["drive"], wow=p["wow"],
                  flutter=p["flutter"], hiss_db=-62.0, bw=p["bw"], rng=rng)
    return [_mono_layer(out, 0.9, fs)]


def _tape_start(p, rng, fs, n):
    dur = n / fs
    src = _make_src(p["src_kind"], min(3.0, dur * 1.2), fs, rng, p["root"], p["bright"])
    t = np.arange(n) / fs
    run = p["run"]
    rate = 0.03 + 0.97 * np.clip(t / run, 0.0, 1.0) ** 0.7
    rate *= 1.0 + p["wob"] * np.sin(2.0 * np.pi * 3.2 * t) * np.exp(-t / max(run, 1e-3))
    out = _play(src, rate, start=0.06 * len(src))
    out *= np.clip(t / (run * 0.6), 0.0, 1.0)
    out = fx.tape(out, fs=fs, drive=p["drive"], wow=p["wow"],
                  flutter=p["flutter"], hiss_db=-62.0, bw=p["bw"], rng=rng)
    click = np.zeros(n)
    k = int(fs * 0.012)
    c = dsp.bp(dsp.white(k, rng), 1800.0, fs, order=2, q=1.4)
    c *= dsp.env_exp(k, fs, attack=0.0003, tau=0.004)
    click[:k] += _norm(c, 0.22)
    return [_mono_layer(out, 0.8, fs), Layer(click, 1.0, label="clunk")]


def _tape_degrade(p, rng, fs, n):
    dur = n / fs
    src = _make_src(p["src_kind"], min(4.0, dur * 1.1), fs, rng, p["root"], p["bright"])
    t = np.arange(n) / fs
    rate = (1.0 + p["wob"] * np.sin(2.0 * np.pi * p["wob_hz"] * t)
            + 0.25 * p["wob"] * np.sin(2.0 * np.pi * 9.0 * t))
    out = _play(src, rate, start=0.0)
    # dropouts: smooth notches where the tape loses contact
    for _ in range(int(p["drops"])):
        c = rng.uniform(0.15, 0.85) * dur
        w = rng.uniform(0.02, 0.06)
        notch = 1.0 - 0.85 * np.exp(-0.5 * ((t - c) / (w / 2.355)) ** 2)
        out *= notch
    out = fx.tape(out, fs=fs, drive=p["drive"], wow=p["wow"],
                  flutter=p["flutter"], hiss_db=-52.0, bw=p["bw"], rng=rng)
    return [_mono_layer(out, 0.9, fs)]


def _tape_hiss(p, rng, fs, n):
    bed = _bed(n, rng, p["noise_kind"]) * 0.02
    y = fx.tape(bed, fs=fs, drive=1.0, wow=0.0004, flutter=0.0002,
                hiss_db=p["hiss_db"], bw=p["bw"], rng=rng)
    y = dsp.lp(y, p["bw"], fs, order=2)
    fade = dsp.env_adsr(n, fs, a=0.05, d=0.1, s=0.95, r=min(0.2, n / fs * 0.3))
    return [_mono_layer(y * fade, 0.9)]


def _tape_hiss_loop(p, rng, fs, n):
    m = int(fs * 0.18)
    nn = n + m
    chans = []
    t = np.arange(nn) / fs
    for _ in range(2):
        bed = _bed(nn, rng, p["noise_kind"]) * 0.02
        y = fx.tape(bed, fs=fs, drive=1.0, wow=p["wow"], flutter=p["flutter"],
                    hiss_db=p["hiss_db"], bw=p["bw"], rng=rng)
        y = dsp.lp(y, p["bw"], fs, order=2)
        # slow "breathing" of the noise floor, 3 cycles per loop: still seamless
        y *= 1.0 + 0.10 * np.sin(2.0 * np.pi * 3.0 * t / (nn / fs) + rng.uniform(0, 6.28))
        chans.append(_loopify(y, fs, ms=180.0))
    L = min(len(chans[0]), len(chans[1]))
    buf = np.stack([chans[0][:L], chans[1][:L]], axis=1)
    return buf


def _wobble(p, rng, fs, n):
    dur = n / fs
    src = _make_src(p["src_kind"], min(4.0, dur * 1.15), fs, rng, p["root"], p["bright"])
    t = np.arange(n) / fs
    rate = (1.0 + p["wob"] * np.sin(2.0 * np.pi * p["wob_hz"] * t + 0.4)
            + 0.18 * p["wob"] * np.sin(2.0 * np.pi * 3.7 * p["wob_hz"] * t))
    out = _play(src, rate, start=0.0)
    out = fx.tape(out, fs=fs, drive=p["drive"], wow=0.0,
                  flutter=p["flutter"], hiss_db=-58.0, bw=p["bw"], rng=rng)
    out *= 1.0 + 0.08 * np.sin(2.0 * np.pi * p["wob_hz"] * 0.5 * t)
    return [_mono_layer(out, 0.9, fs)]


# --------------------------------------------------------------------------
# vinyl archetypes
# --------------------------------------------------------------------------
def _scratch(p, rng, fs, n):
    dur = n / fs
    src = _scratch_src(p["src_kind"], fs, rng, p["root"], 1.8)
    t = np.arange(n) / fs
    u = t / dur
    sp = p["speed"]
    rate = np.zeros(n)
    turns = []
    if p["gesture"] == "baby":
        rate = np.where(u < 0.42, sp, np.where(u < 0.84, -sp, 0.6 * sp))
        turns = [0.42 * dur, 0.84 * dur]
    elif p["gesture"] == "chirp":
        rate = np.where(u < 0.22, 2.1 * sp,
                        np.where(u < 0.44, -2.1 * sp,
                                 np.where(u < 0.60, 1.4 * sp, 0.0)))
        turns = [0.22 * dur, 0.44 * dur, 0.60 * dur]
    else:  # transform: slow forward motion, chopped by the fader
        rate = np.full(n, 0.85 * sp)

    out = _play(src, rate, start=0.04 * len(src))
    # a 3 ms dip at each direction change; real scratches detach there
    for tc in turns:
        out *= 1.0 - 0.85 * np.exp(-0.5 * ((t - tc) / 0.0025) ** 2)
    friction = dsp.bp(dsp.white(n, rng), 2200.0, fs, order=2, q=2.0)
    vel = np.convolve(np.abs(rate), np.ones(96) / 96, mode="same")
    out = out + friction * (p["friction"] * vel)
    if p["gesture"] == "transform":
        out = fx.stutter(out, fs=fs, rate_hz=float(rng.uniform(8.0, 15.0)),
                         duty=0.5, ramp_ms=1.0)
    out = fx.vinyl(out, fs=fs, crackle_hz=p["crackle"], rumble=0.004,
                   hi=9000.0, rng=rng)
    return [_mono_layer(out, 0.9, fs)]


def _backspin(p, rng, fs, n):
    dur = n / fs
    src = _scratch_src(p["src_kind"], fs, rng, p["root"], 1.6)
    t = np.arange(n) / fs
    t0 = dur * 0.10
    rate = np.where(t < t0, 0.9,
                    -p["spin"] * np.exp(-np.maximum(t - t0, 0.0) / p["decel"]))
    out = _play(src, rate, start=0.75 * len(src))
    friction = dsp.bp(dsp.white(n, rng), 2000.0, fs, order=2, q=2.0)
    vel = np.convolve(np.abs(rate), np.ones(96) / 96, mode="same")
    out = out + friction * (p["friction"] * vel)
    out *= np.clip(1.0 - 0.9 * np.maximum(t - dur * 0.75, 0.0) / (dur * 0.25), 0.0, 1.0)
    out = fx.vinyl(out, fs=fs, crackle_hz=1.2, rumble=0.004, hi=9000.0, rng=rng)
    return [_mono_layer(out, 0.9, fs)]


def _vinyl_crackle_loop(p, rng, fs, n):
    m = int(fs * 0.2)
    nn = n + m
    chans = []
    for _ in range(2):
        bed = _bed(nn, rng, p["noise_kind"]) * 0.22 + dsp.lp(dsp.brown(nn, rng), 80.0, fs, order=2) * 0.05
        y = fx.vinyl(bed, fs=fs, crackle_hz=p["crackle"], rumble=p["rumble"],
                     hi=p["hi"], rng=rng)
        # occasional bigger pop on top of the Poisson micro-crackle
        n_big = int(round(nn / fs * rng.uniform(0.4, 1.2)))
        for _j in range(n_big):
            k = _pop(fs, rng, rng.uniform(1200.0, 3600.0), rng.uniform(0.02, 0.07))
            _add_at(y, k, int(rng.integers(0, nn - 1)))
        chans.append(_loopify(y, fs, ms=200.0))
    L = min(len(chans[0]), len(chans[1]))
    return np.stack([chans[0][:L], chans[1][:L]], axis=1)


def _vinyl_pop(p, rng, fs, n):
    dur = n / fs
    bed = _bed(n, rng, p["noise_kind"]) * p["hiss"]
    y = fx.vinyl(bed, fs=fs, crackle_hz=0.0, rumble=0.002, hi=9000.0, rng=rng)
    times = [rng.uniform(0.008, 0.03)]
    for _ in range(int(p["pops"]) - 1):
        times.append(rng.uniform(0.15, 0.85) * dur)
    times.sort()
    for i, tc in enumerate(times):
        k = _pop(fs, rng, rng.uniform(900.0, 4200.0),
                 p["pop_amp"] * (1.0 if i == 0 else rng.uniform(0.5, 0.9)))
        _add_at(y, k, int(tc * fs))
    y = dsp.hp(y, 35.0, fs, order=2)
    return [_mono_layer(y, 0.9, fs)]


def _mono_layer(x: np.ndarray, peak: float, fs: int = FS) -> Layer:
    return Layer(_norm(dsp.dc_block(x, fs), peak), 1.0, label="main")


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    arch = p["archetype"]
    n = int(fs * p["dur"])
    loop = arch in ("tape_hiss_loop", "vinyl_crackle_loop")

    stereo = False
    if arch == "tape_hiss_loop":
        buf = _tape_hiss_loop(p, rng, fs, n)
        # CORE ISSUE: core.render.master -> dsp.fade does `y[:m] *= r`, which
        # crashes on a (n, 2) buffer. Two decorrelated channels are generated and
        # summed to a mono loop; summing keeps the density and avoids the crash.
        buf = buf.mean(axis=1)
    elif arch == "vinyl_crackle_loop":
        buf = _vinyl_crackle_loop(p, rng, fs, n)
        buf = buf.mean(axis=1)
    else:
        if arch == "tape_stop":
            layers = _tape_stop(p, rng, fs, n)
        elif arch == "tape_start":
            layers = _tape_start(p, rng, fs, n)
        elif arch == "tape_degrade":
            layers = _tape_degrade(p, rng, fs, n)
        elif arch == "tape_hiss":
            layers = _tape_hiss(p, rng, fs, n)
        elif arch == "wobble":
            layers = _wobble(p, rng, fs, n)
        elif arch == "scratch":
            layers = _scratch(p, rng, fs, n)
        elif arch == "backspin":
            layers = _backspin(p, rng, fs, n)
        else:                                   # vinyl_pop
            layers = _vinyl_pop(p, rng, fs, n)
        buf = mix(layers, fs=fs, stereo=False)
        if buf.size > n:
            buf = buf[:n]
        elif buf.size < n:
            buf = np.pad(buf, (0, n - buf.size))

    if not loop:
        buf = dsp.hp(buf, 32.0, fs, order=2)
        buf = dsp.dc_block(buf, fs, fc=8.0)
        buf = _release(buf, fs, ms=20.0)
    buf = dsp.fade(buf, fs, ms=4.0)
    # keep the pre-master peak under 1 so master's soft_clip is a no-op, then
    # remove any clip mean (a slow pitch-warped source can hold a DC lobe)
    buf = _norm(buf, 0.92)
    if buf.ndim == 2:
        buf = buf - buf.mean(axis=0, keepdims=True)
    else:
        buf = buf - float(np.mean(buf))

    return Render(buf=buf, fs=fs, stereo=stereo,
                  meta=dict(family=FAMILY, archetype=arch, loop=loop,
                            descriptor=DESCR_OF[arch],
                            drive=round(float(p["drive"]), 2),
                            bw=round(float(p["bw"]), 0)))
