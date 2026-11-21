"""Family: sci_fi. Teleports, energy, force fields, doors, machinery, holograms.

Design notes
------------
Every render is a small stack of explicitly normalised layers (the modal bank in
core is normalised by (1-r), so nothing can be assumed hot).  The palettes:

  teleport   moving_bp noise "suction" + inharmonic FM body + sub sweep
             + granular shimmer + arrival/launch transient
  energy     rising FM + rising moving_bp air + velvet crackle, landing on a
             modal-metal hit with a sub drop
  field      FM hum (integer harmonics) + wobbling band-passed noise sizzle,
             opened by a short engage transient
  door       servo sweep + hydraulic band + 2-3 metal modal mechanism clicks,
             a latch ring, and a thud on close
  machine    circular (wrap-around) metal clanks + tick + motor hum, mono loop
  holoscan   granular shimmer over an FM source + FM scanner pings + air bed

Two transients are deliberate exceptions to "peak within 0-15 ms": a teleport
*in* and an energy charge are risers whose loudest moment is the arrival.  In
both cases the sustained riser sits ~10 dB under the arrival hit, so the peak
is still a real transient, not a hiss plateau.
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "sci_fi"
DESCRIPTORS = ["teleport", "energy", "field", "door", "machine", "holoscan"]
ARCHETYPES = ("teleport", "energy", "field", "door", "machine", "holoscan")


# --------------------------------------------------------------------------
# small local helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    a = float(np.max(np.abs(x))) if np.size(x) else 0.0
    if a <= 1e-12:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) * (peak / a)


def _click(fs: int, rng: np.random.Generator, peak: float = 0.7,
           bright: float = 5200.0) -> np.ndarray:
    k = max(16, int(fs * 0.005))
    x = dsp.hp(dsp.white(k, rng), bright, fs, order=3)
    x *= dsp.env_exp(k, fs, attack=0.0002, tau=0.0016)
    return _norm(x, peak)


def _ping(fs: int, rng: np.random.Generator, f0: float, material: str = "metal",
          tau0: float = 0.22, peak: float = 0.6, hardness: float = 0.85,
          jitter: float = 0.0) -> np.ndarray:
    """One modal mechanism strike / latch / clank."""
    n = max(64, int(fs * min(1.4, tau0 * 4.0 + 0.03)))
    exc = dsp.contact_burst(n, fs, hardness, rng)
    y = dsp.modal(exc, dsp.modal_bank(f0, material, tau0=tau0, rng=rng,
                                      jitter=jitter), fs)
    y = dsp.hp(y, 70.0, fs, order=2)
    return _norm(y, peak)


def _sweep_noise(n: int, fs: int, rng: np.random.Generator, f0: float, f1: float,
                 q: float = 1.5, pink: bool = True, peak: float = 0.25) -> np.ndarray:
    src = dsp.pink(n, rng) if pink else dsp.white(n, rng)
    y = dsp.moving_bp(src, dsp.sweep(n, f0, f1, "exp"), fs, q=q, bands=16)
    return _norm(dsp.dc_block(y, fs), peak)


def _rise_fall(n: int, fs: int, t_hit: float, power: float = 1.6,
               tau: float = 0.09, floor: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fs
    h = min(max(t_hit, 0.01), (n / fs) - 0.005)
    rise = np.clip(t / h, 0.0, 1.0) ** power
    fall = np.exp(-np.maximum(t - h, 0.0) / max(tau, 1e-4))
    return floor + (1.0 - floor) * rise * fall


def _tilt(x: np.ndarray, tilt: float, fs: int) -> np.ndarray:
    """Bright/dark axis. tilt in [-1, 1]: +1 adds a lot of 3 kHz+ air, -1 removes it.

    One strong spectral axis is worth more for the pack than a dozen micro-varied
    renders: MFCC dedup is dominated by average spectral shape.
    """
    if abs(tilt) < 0.05:
        return x
    hi = dsp.hp(x, 2800.0, fs, order=2)
    return x + float(np.clip(tilt, -1.0, 1.0)) * 0.85 * hi


def _release(x: np.ndarray, fs: int, ms: float = 25.0) -> np.ndarray:
    """Longer end release than the 4 ms anti-click fade, so cut modal tails don't tick."""
    m = min(int(fs * ms / 1000.0), max(2, len(x) // 8))
    if m > 2:
        x = x.copy()
        x[-m:] *= np.cos(np.linspace(0.0, np.pi / 2.0, m)) ** 2
    return x


def _seamless_noise(n: int, fs: int, rng: np.random.Generator, color: str = "pink",
                    xf_ms: float = 50.0) -> np.ndarray:
    """Noise whose head has the pre-roll tail folded in, so it loops cleanly."""
    m = min(int(fs * xf_ms / 1000.0), max(4, n // 4))
    raw = dsp.pink(n + m, rng) if color == "pink" else dsp.white(n + m, rng)
    out = raw[:n].copy()
    r = np.linspace(0.0, 1.0, m)
    out[:m] = raw[:m] * r + raw[n:n + m] * (1.0 - r)
    return out


# --------------------------------------------------------------------------
# parameter spread
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    arch = str(rng.choice(ARCHETYPES, p=[0.22, 0.20, 0.17, 0.15, 0.12, 0.14]))
    p = dict(archetype=arch, dirt=float(rng.uniform(0.0, 1.0)),
             tilt=float(rng.uniform(-1.0, 1.0)),
             sub=float(rng.uniform(0.15, 0.70)),
             space=float(rng.uniform(0.04, 0.30)),
             space_t60=float(rng.uniform(0.4, 1.3)))

    if arch == "teleport":
        p.update(dur=float(rng.uniform(0.55, 1.35)),
                 direction=str(rng.choice(["in", "out"], p=[0.6, 0.4])),
                 f0=float(rng.uniform(220.0, 650.0)),
                 f1=float(rng.uniform(1000.0, 5500.0)),
                 ratio=float(rng.choice([1.41, 2.37, 3.51, 4.73])),
                 index=float(rng.uniform(1.5, 4.5)),
                 shimmer=float(rng.uniform(35.0, 95.0)),
                 q=float(rng.uniform(1.0, 2.2)))
    elif arch == "energy":
        flavor = str(rng.choice(["charge", "surge"], p=[0.65, 0.35]))
        p.update(flavor=flavor, dur=float(rng.uniform(0.85, 1.9)),
                 f0=float(rng.uniform(150.0, 380.0)),
                 f1=float(rng.uniform(700.0, 3600.0)),
                 ratio=float(rng.choice([1.41, 2.37, 3.51])),
                 index=float(rng.uniform(1.0, 3.5)),
                 hit_f=float(rng.uniform(450.0, 2600.0)),
                 crackle=float(rng.uniform(25.0, 90.0)),
                 hit_at=float(rng.uniform(0.18, 0.30) if flavor == "surge"
                              else rng.uniform(0.70, 0.84)))
    elif arch == "field":
        p.update(dur=float(rng.uniform(1.6, 2.5)),
                 flavor=str(rng.choice(["hum", "crackle"], p=[0.55, 0.45])),
                 hum_f=float(rng.uniform(55.0, 112.0)),
                 ratio=float(rng.choice([2.0, 3.0])),
                 trem=float(rng.uniform(0.4, 1.8)),
                 trem_depth=float(rng.uniform(0.08, 0.28)),
                 bp_c=float(rng.uniform(650.0, 3600.0)),
                 wob=float(rng.uniform(0.15, 0.7)),
                 engage_f=float(rng.uniform(600.0, 1500.0)))
    elif arch == "door":
        p.update(dur=float(rng.uniform(0.8, 1.6)),
                 opening=bool(rng.random() < 0.5),
                 servo0=float(rng.uniform(320.0, 640.0)),
                 servo1=float(rng.uniform(1300.0, 2400.0)),
                 thud_f=float(rng.uniform(48.0, 78.0)),
                 ring_f=float(rng.uniform(120.0, 620.0)),
                 material=str(rng.choice(["metal", "ceramic"],
                                         p=[0.7, 0.3])),
                 clicks=int(rng.integers(2, 4)))
    elif arch == "machine":
        period = float(rng.uniform(0.24, 0.42))
        reps = int(rng.integers(2, 6))
        p.update(period=period, reps=reps, dur=float(period * reps),
                 hum_f=float(rng.uniform(48.0, 96.0)),
                 clank_f=float(rng.uniform(180.0, 520.0)),
                 clank_tau=float(rng.uniform(0.08, 0.22)),
                 tick_bright=float(rng.uniform(2500.0, 6500.0)))
    else:  # holoscan
        p.update(dur=float(rng.uniform(0.9, 2.0)),
                 flavor=str(rng.choice(["pings", "shimmer", "sweep"],
                                       p=[0.42, 0.34, 0.24])),
                 n_pings=int(rng.integers(2, 7)),
                 ping_f=float(rng.uniform(900.0, 4200.0)),
                 ratio=float(rng.choice([2.37, 3.51, 4.73])),
                 density=float(rng.uniform(35.0, 95.0)),
                 air0=float(rng.uniform(400.0, 900.0)),
                 air1=float(rng.uniform(1800.0, 4800.0)),
                 up=bool(rng.random() < 0.6))

    # One log-uniform "device size" multiplier over every frequency. This is the
    # single strongest anti-clone lever: the same machinery at 0.65x and 1.5x is
    # a different device, not a re-seed of the same one.
    scale = float(np.exp(rng.uniform(np.log(0.65), np.log(1.5))))
    for k in ("f0", "f1", "hum_f", "bp_c", "engage_f", "servo0", "servo1",
              "thud_f", "ring_f", "clank_f", "ping_f", "air0", "air1", "hit_f"):
        if k in p:
            p[k] = float(np.clip(p[k] * scale, 30.0, 12000.0))
    p["scale"] = scale
    p["space"] = float(np.clip(p["space"] * rng.uniform(0.6, 1.6), 0.0, 0.40))
    return p


# --------------------------------------------------------------------------
# archetypes
# --------------------------------------------------------------------------
def _teleport(p, rng, fs, n):
    t = np.arange(n) / fs
    going_in = p["direction"] == "in"
    t_arrive = p["dur"] * (0.66 if going_in else 0.04)
    # riser / faller envelope: quiet suction under the arrival transient
    if going_in:
        shape = _rise_fall(n, fs, t_arrive, power=2.0, tau=0.11, floor=0.0)
    else:
        shape = np.exp(-np.maximum(t - t_arrive, 0.0) / (p["dur"] * 0.45))

    f_noise = dsp.sweep(n, p["f0"], p["f1"] if going_in else p["f0"] * 0.5, "exp")
    suction = _sweep_noise(n, fs, rng, p["f0"] * 1.2,
                           p["f1"] if going_in else p["f0"] * 0.4,
                           q=p["q"], pink=True, peak=0.22) * shape

    fc = dsp.sweep(n, p["f0"], p["f1"] * 0.6, "exp")
    idx_env = p["index"] * np.exp(-t / (p["dur"] * 0.35))
    body = dsp.fm(fc, p["ratio"], idx_env, fs)
    body = dsp.lp(body, 7000.0, fs, order=2)
    body = _norm(body * shape, 0.5)

    sub = dsp.sine(dsp.sweep(n, 120.0, 38.0, "exp"), fs) * shape
    sub = _norm(dsp.dc_block(sub, fs), 0.34 * p["sub"] / 0.42)

    # granular shimmer over our own FM material
    src = dsp.fm(dsp.const(int(fs * 0.5), p["f1"] * 0.5), p["ratio"],
                 np.full(int(fs * 0.5), p["index"] * 0.5), fs)
    src = _norm(src, 0.8)
    shim = dsp.granulate(src, n, p["shimmer"], fs, rng=rng,
                         grain_ms=(18.0, 60.0), pitch=(0.8, 1.6))
    shim = _norm(shim * shape, 0.14)

    hit = np.zeros(n)
    k = _click(fs, rng, peak=0.8, bright=4200.0)
    off = int(t_arrive * fs)
    e = min(n, off + k.size)
    hit[off:e] = k[:e - off]
    if going_in:
        ring = _ping(fs, rng, p["f1"] * 0.35, "glass", tau0=0.18, peak=0.45,
                     hardness=0.95)
        e2 = min(n, off + ring.size)
        hit[off:e2] += ring[:e2 - off]
    hit = _norm(hit, 0.82)
    return [Layer(suction, 1.0, label="suction"),
            Layer(body, 1.0, label="fm_body"),
            Layer(sub, 1.0, label="sub"),
            Layer(shim, 1.0, label="shimmer"),
            Layer(hit, 1.0, label="transient")]


def _energy(p, rng, fs, n):
    t_hit = p["dur"] * p["hit_at"]
    shape = _rise_fall(n, fs, t_hit, power=1.5, tau=0.10)

    fc = dsp.sweep(n, p["f0"], p["f1"], "exp")
    idx_env = p["index"] * np.linspace(0.4, 1.0, n) * shape
    rise = dsp.fm(fc, p["ratio"], idx_env, fs)
    rise = _norm(dsp.lp(rise, 6500.0, fs, order=2) * shape, 0.45)

    air = _sweep_noise(n, fs, rng, p["f0"] * 2.0, p["f1"] * 1.8,
                       q=1.7, pink=True, peak=0.24) * shape

    # crackle: velvet impulses through a bright band, density rising with the sweep
    vel = dsp.velvet(n, period=max(6, int(fs / 900.0)), rng=rng)
    crack = dsp.bp(vel, min(9000.0, p["f1"] * 3.0), fs, order=2, q=2.0)
    crack = _norm(crack * shape, 0.12)

    hit = np.zeros(n)
    hit_times = [t_hit]
    if p.get("flavor") == "surge":
        hit_times.append(p["dur"] * 0.74)      # a second, smaller discharge
    for hi, th in enumerate(hit_times):
        amp = 1.0 if hi == 0 else 0.55
        off = int(th * fs)
        kl = _click(fs, rng, peak=0.8 * amp, bright=5000.0)
        e = min(n, off + kl.size)
        hit[off:e] += kl[:e - off]
        ring = _ping(fs, rng, p["hit_f"] * (1.0 if hi == 0 else 1.35), "metal",
                     tau0=0.28, peak=0.65 * amp, hardness=0.9, jitter=0.01)
        e = min(n, off + ring.size)
        hit[off:e] += ring[:e - off]
        ksub = int(min(n - off, fs * 0.30))
        if ksub > 32:
            sub = dsp.sine(dsp.sweep(ksub, 115.0, 38.0, "exp"), fs)
            sub *= dsp.env_exp(ksub, fs, attack=0.001, tau=0.09)
            hit[off:off + ksub] += _norm(sub, 0.45 * amp * p["sub"] / 0.42)
    hit = _norm(hit, 0.85)
    return [Layer(rise, 1.0, label="fm_rise"),
            Layer(air, 1.0, label="air_rise"),
            Layer(crack, 1.0, label="crackle"),
            Layer(hit, 1.0, label="arrival")]


def _field(p, rng, fs, n):
    t = np.arange(n) / fs
    # snap the hum to an integer number of cycles over the file. A partial with a
    # non-integer cycle count leaves a non-zero clip mean, which the auditor reads
    # as DC offset (a half-rate sub partial was doing exactly that).
    hum_f = round(max(44.0, p["hum_f"]) * (n / fs)) / (n / fs)
    fc = dsp.const(n, hum_f)
    idx_env = np.full(n, 1.4 * p["ratio"] / 2.0)
    hum = dsp.fm(fc, p["ratio"], idx_env, fs) * 0.7
    hum += dsp.saw(dsp.const(n, 2.0 * hum_f), fs) * 0.25
    hum = dsp.lp(hum, 1400.0, fs, order=3)
    trem = 1.0 + p["trem_depth"] * np.sin(2.0 * np.pi * p["trem"] * t)
    if p.get("flavor") == "crackle":
        hum_g, sizz_g = 0.24, 0.36      # shield crackle dominates
    else:
        hum_g, sizz_g = 0.46, 0.12      # the hum itself dominates
    hum = _norm(dsp.env_adsr(n, fs, a=0.012, d=0.15, s=0.85, r=0.18)
                * hum * trem, hum_g)

    sizz = dsp.pink(n, rng)
    centre = p["bp_c"] * (1.0 + 0.35 * np.sin(2.0 * np.pi * p["wob"] * t))
    sizz = dsp.moving_bp(sizz, centre, fs, q=2.2, bands=16)
    sizz = _norm(dsp.env_adsr(n, fs, a=0.02, d=0.1, s=0.9, r=0.18)
                 * sizz, sizz_g)

    engage = np.zeros(n)
    kl = _click(fs, rng, peak=0.75, bright=4500.0)
    engage[:kl.size] += kl
    ping = _ping(fs, rng, p["engage_f"], "glass", tau0=0.12, peak=0.4)
    e = min(n, ping.size)
    engage[:e] += ping[:e]
    engage = _norm(engage, 0.8)
    return [Layer(hum, 1.0, label="hum"),
            Layer(sizz, 1.0, label="sizzle"),
            Layer(engage, 1.0, label="engage")]


def _door(p, rng, fs, n):
    opening = p["opening"]
    layers = []
    t_servo_end = p["dur"] * 0.75

    f = (dsp.sweep(n, p["servo0"], p["servo1"], "exp") if opening
         else dsp.sweep(n, p["servo1"], p["servo0"], "exp"))
    env = np.clip(np.arange(n) / (fs * t_servo_end), 0.0, 1.0)
    env = env * np.exp(-np.maximum(np.arange(n) / fs - t_servo_end, 0.0) / 0.12)
    servo = dsp.moving_bp(dsp.pink(n, rng), f, fs, q=1.6, bands=16)
    layers.append(Layer(_norm(servo * env, 0.26), 1.0, label="servo"))

    hyd_f = dsp.const(n, p["servo0"] * 0.6) * (
        1.0 + 0.12 * np.sin(2.0 * np.pi * 9.0 * np.arange(n) / fs))
    hyd = dsp.moving_bp(dsp.pink(n, rng), hyd_f, fs, q=2.6, bands=16)
    layers.append(Layer(_norm(hyd * env, 0.18), 1.0, label="hydraulic"))

    # mechanism clicks spread over the travel, latch ring at the end
    click_times = np.linspace(0.06, 0.52, p["clicks"]) * p["dur"]
    for i, tc in enumerate(click_times):
        c = _ping(fs, rng, p["ring_f"] * (1.0 + 0.35 * i), p["material"],
                  tau0=0.10 + 0.05 * i, peak=0.42, hardness=0.92, jitter=0.01)
        layers.append(Layer(c, 1.0, offset_s=float(tc), label=f"click{i}"))

    latch = _ping(fs, rng, p["ring_f"], p["material"], tau0=0.35,
                  peak=0.5, hardness=0.95, jitter=0.005)
    # latch rings from the end of the travel; the tail is allowed to run out
    latch_at = p["dur"] * (0.72 if opening else 0.10)
    layers.append(Layer(latch, 1.0, offset_s=float(latch_at), label="latch"))

    # the thud lands on the last 30 % of the gesture when the door closes
    kth = int(min(fs * 0.45, fs * p["dur"] * 0.30))
    sub = dsp.sine(dsp.sweep(kth, p["thud_f"] * 1.6, p["thud_f"], "exp"), fs)
    sub *= dsp.env_exp(kth, fs, attack=0.001, tau=0.10)
    sub = _norm(dsp.dc_block(sub, fs),
                (0.7 if not opening else 0.28) * p["sub"] / 0.42)
    thud_at = (p["dur"] * 0.70) if not opening else (p["dur"] * 0.04)
    layers.append(Layer(sub, 1.0, offset_s=float(thud_at), label="thud"))
    return layers


def _machine(p, rng, fs, n):
    nper = max(64, int(round(fs * p["period"])))
    # period is snapped to whole samples so every cycle lands on the same grid
    dur = n / fs
    hum_f = round(p["hum_f"] * dur) / dur          # integer cycles over the loop
    t = np.arange(n) / fs
    hum = dsp.saw(dsp.const(n, hum_f), fs) * 0.5 + dsp.sine(dsp.const(n, hum_f * 2.0), fs) * 0.4
    hum = dsp.lp(hum, 900.0, fs, order=3)
    hum = _norm(hum * (1.0 + 0.15 * np.sin(2.0 * np.pi * 4.0 * t)), 0.24)

    clank = np.zeros(n)
    tick = np.zeros(n)
    for j in range(p["reps"]):
        y = _ping(fs, rng, p["clank_f"] * rng.uniform(0.95, 1.06), "metal",
                  tau0=p["clank_tau"] * rng.uniform(0.9, 1.1), peak=0.62,
                  hardness=0.92, jitter=0.012)
        pos = j * nper
        idx = (pos + np.arange(y.size)) % n
        np.add.at(clank, idx, y)
        kt = _click(fs, rng, peak=0.2 * rng.uniform(0.8, 1.2),
                    bright=p["tick_bright"])
        pos = (j * nper + nper // 2) % n
        idx = (pos + np.arange(kt.size)) % n
        np.add.at(tick, idx, kt)

    air = _seamless_noise(n, fs, rng, "pink")
    air = dsp.moving_bp(air, dsp.const(n, 2400.0), fs, q=3.0, bands=16)
    return [Layer(hum, 1.0, label="motor"),
            Layer(_norm(clank, 0.65), 1.0, label="clank"),
            Layer(_norm(tick, 0.22), 1.0, label="tick"),
            Layer(_norm(air, 0.05), 1.0, label="air")]


def _holoscan(p, rng, fs, n):
    t = np.arange(n) / fs
    # granular shimmer over an FM source we synthesised
    sl = int(fs * 0.3)
    src = dsp.fm(dsp.const(sl, p["ping_f"] * 0.5), p["ratio"],
                 np.full(sl, 1.6), fs)
    src = _norm(dsp.lp(src, 8000.0, fs, order=2), 0.8)
    flav = p.get("flavor", "pings")
    ping_g, shim_g, air_g = {"pings": (0.62, 0.16, 0.08),
                             "shimmer": (0.18, 0.50, 0.10),
                             "sweep": (0.26, 0.14, 0.32)}[flav]
    shim = dsp.granulate(src, n, p["density"], fs, rng=rng,
                         grain_ms=(12.0, 55.0), pitch=(0.7, 1.7))
    shim = _norm(shim * dsp.env_adsr(n, fs, a=0.02, d=0.1, s=0.8,
                                     r=p["dur"] * 0.25), shim_g)

    up = True if flav == "sweep" else p["up"]
    f0, f1 = (p["air0"], p["air1"]) if up else (p["air1"], p["air0"])
    air = _sweep_noise(n, fs, rng, f0, f1, q=1.4, pink=True, peak=air_g)
    air *= dsp.env_adsr(n, fs, a=0.05, d=0.2, s=0.7, r=p["dur"] * 0.2)

    pings = np.zeros(n)
    for i in range(p["n_pings"]):
        tc = (0.10 + 0.72 * i / max(1, p["n_pings"] - 1)) * p["dur"]
        k = int(fs * 0.4)
        idx_env = 2.2 * np.exp(-np.arange(k) / (fs * 0.09))
        y = dsp.fm(dsp.const(k, p["ping_f"] * rng.uniform(0.85, 1.25)),
                   p["ratio"], idx_env, fs)
        y *= dsp.env_exp(k, fs, attack=0.002, tau=0.13)
        y = _norm(y, 0.5 * (1.0 - 0.16 * i))
        off = int(tc * fs)
        e = min(n, off + k)
        pings[off:e] += y[:e - off]
    pings = _norm(pings, ping_g)

    start = np.zeros(n)
    kl = _click(fs, rng, peak=0.45, bright=6000.0)
    start[:kl.size] += kl
    return [Layer(shim, 1.0, label="shimmer"),
            Layer(air, 1.0, label="air"),
            Layer(pings, 1.0, label="pings"),
            Layer(start, 1.0, label="start")]


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    arch = p["archetype"]
    if arch == "machine":
        nper = max(64, int(round(fs * p["period"])))
        n = nper * int(p["reps"])
        dur = n / fs
    else:
        n = int(fs * p["dur"])
        dur = p["dur"]

    if arch == "teleport":
        layers = _teleport(p, rng, fs, n)
        desc = "teleport"
        loop = False
    elif arch == "energy":
        layers = _energy(p, rng, fs, n)
        desc = "energy"
        loop = False
    elif arch == "field":
        layers = _field(p, rng, fs, n)
        desc = "field"
        loop = False
    elif arch == "door":
        layers = _door(p, rng, fs, n)
        desc = "door"
        loop = False
    elif arch == "machine":
        layers = _machine(p, rng, fs, n)
        desc = "machine"
        loop = True
    else:
        layers = _holoscan(p, rng, fs, n)
        desc = "holoscan"
        loop = False

    buf = mix(layers, fs=fs, stereo=False)
    # hard-trim to the stated duration: overhanging modal tails are cut, not left
    # to stretch the one-shot past the family's duration spec
    if buf.size > n:
        buf = buf[:n]
    elif buf.size < n:
        buf = np.pad(buf, (0, n - buf.size))

    buf = _tilt(buf, float(p["tilt"]), fs)

    if p["dirt"] > 0.6:
        buf = fx.sat_adaa(buf, 1.2 + 1.4 * p["dirt"])
    elif p["dirt"] > 0.3:
        buf = fx.sat_os(buf, 1.0 + 0.8 * p["dirt"]) if p["dirt"] > 0.45 else buf

    if p["space"] > 0.03:
        buf = fx.reverb(buf, fs=fs, t60=float(p["space_t60"]), size=0.35,
                        mix=float(p["space"]), rng=rng)

    buf = dsp.hp(buf, 40.0 if arch == "field" else 32.0, fs, order=2)
    buf = dsp.dc_block(buf, fs, fc=8.0)
    if not loop:
        buf = _release(buf, fs, ms=25.0)
    buf = dsp.fade(buf, fs, ms=4.0)
    # Keep the pre-master peak under 1.0: master's soft_clip is an asymmetric-in-
    # practice waveform shaper, and clipping an asymmetric hum puts a DC step into
    # the render. Normalise first, then take the mean out as the final operation
    # (low hums are not an integer number of cycles over the windowed file).
    buf = _norm(buf, 0.95)
    buf = buf - float(np.mean(buf))
    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, archetype=arch, loop=loop,
                            descriptor=desc, seconds=round(dur, 3),
                            flavor=p.get("flavor"), scale=round(float(p["scale"]), 3),
                            dirt=round(float(p["dirt"]), 3)))
