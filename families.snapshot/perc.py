"""Family: perc — glitch/electronic percussion one-shots.

Six types, each with its own synthesis recipe (never one recipe reskinned):
  tick      high-passed noise burst 3-9 kHz + tiny tone     (UI / HUD tick)
  click     band-passed noise + low anchor tone             (menu click)
  rim       short metal modal bank, hard contact            (rimshot)
  woodblock modal wood bank, soft-ish contact               (woodblock)
  blip      pitch-enveloped square, soft-bitcrushed          (digital blip)
  ping      metal modal ping + fast-decaying fundamental     (metallic ping)

All are 30-180 ms mono one-shots with a synthesised 1.6 ms broadband click on top
so the transient is inside the first 15 ms. Every layer is peak-normalised
explicitly. Pitch is the main seed variable: log-uniform over a wide band per
type, because these are the most-repeated sounds in a game.
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "perc"
DESCRIPTORS = ["tick", "click", "ping", "rim", "blip"]

TYPES = ("tick", "click", "rim", "woodblock", "blip", "ping")

# (duration_s, f0_lo_hz, f0_hi_hz)
TYPE_SPEC = {
    "tick":      (0.030, 0.055, 3000.0, 9000.0),
    "click":     (0.030, 0.055, 1200.0, 6000.0),
    "rim":       (0.035, 0.075, 600.0, 1900.0),
    "woodblock": (0.055, 0.120, 240.0, 900.0),
    "blip":      (0.050, 0.140, 300.0, 2000.0),
    "ping":      (0.090, 0.180, 900.0, 3200.0),
}


def _set_peak(x: np.ndarray, target: float) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p <= 0.0:
        return np.zeros_like(x)
    return x / p * float(target)


def _restore_attack(x: np.ndarray, fs: int, ms: float = 4.0,
                    final_ms: float = 1.0, cap: float = 40.0) -> np.ndarray:
    """CORE ISSUE workaround. render.master() fades EVERY one-shot in over 4 ms,
    which is a third of a 30 ms tick. Applied to our own faded buffer, this
    divides out that ramp (capped) so master's 4 ms fade lands as a 1 ms fade."""
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


def _click(n: int, fs: int, rng: np.random.Generator, gain: float) -> np.ndarray:
    k = max(8, int(fs * 0.0016))
    out = np.zeros(n)
    burst = dsp.hp(dsp.white(k, rng), 3500.0, fs, order=3)
    burst = burst * dsp.env_exp(k, fs, attack=0.00012, tau=0.00055)
    out[:k] = _set_peak(burst, 1.0)
    return out * float(gain)


def _modal_body(n: int, fs: int, f0: float, material: str, tau0: float,
                size: float, hardness: float, rng: np.random.Generator,
                jitter: float = 0.006) -> np.ndarray:
    exc = dsp.contact_burst(n, fs, hardness=hardness, rng=rng)
    modes = dsp.modal_bank(f0, material, tau0=tau0, size=size, rng=rng,
                           jitter=jitter)
    return _set_peak(dsp.modal(exc, modes, fs), 1.0)


def sample_params(rng: np.random.Generator) -> dict:
    typ = str(rng.choice(TYPES, p=[0.18, 0.14, 0.16, 0.14, 0.20, 0.18]))
    d_lo, d_hi, f_lo, f_hi = TYPE_SPEC[typ]
    dur = float(rng.uniform(d_lo, d_hi))
    f0 = float(np.exp(rng.uniform(np.log(f_lo), np.log(f_hi))))
    return dict(
        typ=typ,
        dur=dur,
        f0=f0,
        decay_scale=float(rng.uniform(0.60, 1.40)),
        size=float(rng.uniform(0.60, 1.40)),
        hardness=float(rng.uniform(0.55, 1.00)),
        noise=float(rng.uniform(0.10, 0.60)),
        click_gain=float(rng.uniform(0.35, 0.85)),
        drive=float(rng.uniform(0.0, 0.6)),
        hp_f=float(rng.uniform(90.0, 200.0)),
        ratio=float(rng.uniform(1.5, 3.0)),
        bits=int(rng.integers(5, 10)),
        rate_div=int(rng.choice([2, 3, 4, 6])),
        duty=float(rng.uniform(0.28, 0.62)),
        bright=float(rng.uniform(0.45, 0.95)),
    )


def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    typ = p["typ"]
    n = max(64, int(fs * p["dur"]))
    f0 = p["f0"]
    t = np.arange(n) / fs

    body = np.zeros(n)
    if typ == "tick":
        nz = dsp.hp(dsp.white(n, rng), f0, fs, order=3)
        nz *= dsp.env_exp(n, fs, attack=0.0002,
                          tau=p["dur"] * 0.18 * p["decay_scale"])
        tone = dsp.sine(dsp.const(n, f0 * 0.55), fs)
        tone *= dsp.env_exp(n, fs, attack=0.0002,
                            tau=p["dur"] * 0.10 * p["decay_scale"])
        body = nz + 0.35 * tone
    elif typ == "click":
        nb = dsp.bp(dsp.white(n, rng), f0, fs, order=2, q=1.1)
        nb *= dsp.env_exp(n, fs, attack=0.00015,
                          tau=p["dur"] * 0.22 * p["decay_scale"])
        tone = dsp.sine(dsp.const(n, f0 * 0.50), fs)
        tone *= dsp.env_exp(n, fs, attack=0.00015,
                            tau=p["dur"] * 0.14 * p["decay_scale"])
        body = nb + 0.50 * tone
    elif typ == "rim":
        body = _modal_body(n, fs, f0, "metal",
                           tau0=p["dur"] * 0.28 * p["decay_scale"],
                           size=p["size"], hardness=max(0.80, p["hardness"]),
                           rng=rng, jitter=0.006)
    elif typ == "woodblock":
        body = _modal_body(n, fs, f0, "wood",
                           tau0=p["dur"] * 0.55 * p["decay_scale"],
                           size=p["size"], hardness=p["hardness"],
                           rng=rng, jitter=0.020)
    elif typ == "blip":
        tau_p = float(rng.uniform(0.004, 0.014))
        fcurve = f0 * (1.0 + (p["ratio"] - 1.0) * np.exp(-t / tau_p))
        osc = dsp.square(fcurve, fs, duty=p["duty"])
        osc = dsp.lp(osc, min(15000.0, f0 * 12.0 + 2000.0), fs, order=2)
        osc = fx.bitcrush(osc, bits=p["bits"], rate_div=p["rate_div"],
                          mode="soft", fs=fs)
        osc *= dsp.env_exp(n, fs, attack=0.0004,
                           tau=p["dur"] * 0.35 * p["decay_scale"])
        sinep = dsp.sine(fcurve, fs) * 0.35
        sinep *= dsp.env_exp(n, fs, attack=0.0004,
                             tau=p["dur"] * 0.25 * p["decay_scale"])
        body = osc + sinep
    else:  # ping
        body = _modal_body(n, fs, f0, "metal",
                           tau0=p["dur"] * 0.55 * p["decay_scale"],
                           size=p["size"], hardness=max(0.85, p["hardness"]),
                           rng=rng, jitter=0.010)
        fund = dsp.sine(dsp.const(n, f0), fs)
        fund *= dsp.env_exp(n, fs, attack=0.0003,
                            tau=p["dur"] * 0.22 * p["decay_scale"])
        body = body + 0.40 * fund

    body = _set_peak(body, 1.0)

    # whichever type, a synthesised broadband click guarantees the attack
    clk = _click(n, fs, rng, p["click_gain"])

    buf = mix([
        Layer(body, gain=1.0, label="body"),
        Layer(clk, gain=1.0, label="click"),
    ], fs=fs, stereo=False)

    if p["drive"] > 0.02:
        buf = fx.sat_adaa(buf, 1.0 + p["drive"])
    buf = dsp.hp(buf, p["hp_f"], fs, order=2)
    buf = dsp.dc_block(buf, fs, fc=15.0)     # clicks leave a DC tail; kill it here
    buf = _set_peak(buf, 0.95)
    buf = dsp.fade(buf, fs, ms=4.0)
    buf = _restore_attack(buf, fs)
    buf = dsp.dc_block(buf, fs, fc=15.0)      # the restored head can carry offset
    buf = _set_peak(buf, 0.95)

    return Render(buf=buf, fs=fs, stereo=False,
                  meta=dict(family=FAMILY, type=typ, loop=False,
                            descriptor=f"{typ}_{int(f0)}",
                            hardness=round(p["hardness"], 3),
                            size=round(p["size"], 3)))
