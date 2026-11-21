"""Family: drum — one-shot kit voices for trap / drill / phonk / jersey club / boom bap.

Built from the recipes in briefs/CONTRACT.md:
  kick      pitch-enveloped sine (root ~28-46 Hz, start 45-72 Hz) + 2-12 ms click
  808 kick  x1.8-2.4 pitch drop, long driven body, optional beater knock
  snare     body 150-260 Hz with a short pitch drop + broad noise burst + room
  hat       highpassed / bandpassed noise 4-11 kHz with a randomized metallic stack
  clap      3-5 bandpassed noise bursts 7-14 ms apart, then a short tail
  rim       short wooden/plastic modal ring driven by a contact burst + stick tick
  ride      long metallic hat with a shimmering partial stack

Diversity is designed, not left to luck. Every archetype carries a named character
(sub / punch / knock / driven / thud kick; tight / roomy / crack / fat snare;
noise / metal / band / dark hat; tight / wide / snap clap) AND every continuous
parameter is wide inside the character: pitch-envelope depth and time, audible decay,
click hardness, drive, body resonance, noise/tone balance, band centre, room send.
That is what keeps 60 candidates from collapsing into a handful of timbres.

Structural rules (CONTRACT.md):
  * every layer is peak-normalised to an explicit target before the mix
  * transient first: click / noise burst starts at sample 0, body follows
  * mono, one-shot, highpass + fade at the end; distortion only where it earns it
  * final buffer scaled just under the master's soft-clip knee and DC-corrected,
    because an asymmetric click + soft_clip generates a real DC shift (see .md)

Contract:
    FAMILY, DESCRIPTORS, sample_params(rng), generate(params, rng, fs) -> Render
"""
from __future__ import annotations

import numpy as np

from core import dsp, fx
from core.dsp import FS
from core.render import Layer, Render, mix

FAMILY = "drum"
DESCRIPTORS = ["kick", "snare", "hat", "clap", "rim"]

ARCHETYPES = ("kick", "kick808", "snare", "hat_closed", "hat_open", "clap", "rim", "ride")
_P_ARCH = (0.18, 0.08, 0.21, 0.17, 0.09, 0.12, 0.08, 0.07)

# short UCS-safe words + a readable variant tag used in the filename
_DESCRIPTOR = {
    "kick": "kick",
    "kick808": "kick808",
    "snare": "snare",
    "hat_closed": "hat",
    "hat_open": "openhat",
    "clap": "clap",
    "rim": "rim",
    "ride": "ridehat",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """Explicit per-layer target. Never trust an emergent mix level."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-12:
        return np.zeros_like(x)
    return x * (peak / m)


def _click(n: int, fs: int, rng: np.random.Generator, peak: float = 0.35,
           hp_hz: float = 3000.0, ms: float = 5.0) -> np.ndarray:
    """Short highpassed noise burst: the attack that makes a drum read on phones."""
    k = int(np.clip(fs * ms / 1000.0, 8, n))
    out = np.zeros(n)
    nz = dsp.hp(dsp.white(k, rng), hp_hz, fs, order=2)
    out[:k] = _norm(nz * dsp.env_exp(k, fs, attack=0.0002, tau=0.0016), 1.0)
    return out * float(peak)


def _metal_ratios(rng: np.random.Generator) -> tuple[float, ...]:
    """Random inharmonic partial ratios: fixed ratios made hats sound alike."""
    ratios = [1.0]
    for _ in range(int(rng.integers(3, 6))):
        ratios.append(ratios[-1] * float(rng.uniform(1.15, 1.42)))
    return tuple(ratios)


def _metal_stack(n: int, fs: int, rng: np.random.Generator, base: float,
                 tau: float, amount: float, ratios: tuple[float, ...]) -> np.ndarray:
    """Inharmonic sine partials = the 808 cymbal / metallic hat character."""
    out = np.zeros(n)
    for i, r in enumerate(ratios):
        f = base * r
        if f >= fs * 0.44:
            continue
        env = dsp.env_exp(n, fs, attack=0.0002, tau=tau * (0.92 - 0.11 * i))
        out += dsp.sine(np.full(n, f), fs, phase0=float(rng.random())) * (1.0 / (1.0 + i)) * env
    out = dsp.hp(out, 4000.0, fs, order=2)
    return _norm(out, 0.6 * amount)


def _resonance(n: int, fs: int, rng: np.random.Generator, f0: float, tau0: float,
               level: float, hardness: float = 0.72,
               material: str | None = None) -> np.ndarray:
    """Short resonant body: a beater knock (kick) or a stick ring (rim)."""
    if material is None:
        material = str(rng.choice(("wood", "plastic", "ceramic"), p=(0.55, 0.30, 0.15)))
    modes = dsp.modal_bank(f0, material, tau0=tau0, size=float(rng.uniform(0.70, 1.10)),
                           rng=rng, jitter=0.03)
    exc = dsp.contact_burst(n, fs, hardness=hardness, rng=rng)
    return _norm(dsp.modal(exc, modes, fs), level)


def _room(buf: np.ndarray, fs: int, rng: np.random.Generator, t60: float,
          mix_amt: float) -> np.ndarray:
    if mix_amt <= 0.02:
        return buf
    return fx.reverb(buf, fs=fs, t60=float(t60), size=0.22,
                     mix=float(np.clip(mix_amt, 0.0, 0.45)), rng=rng)


def _finish(buf: np.ndarray, fs: int) -> np.ndarray:
    """Highpass -> scale under the soft-clip knee -> remove windowed DC -> fade.

    The master chain soft-clips at 0.995. A kick is asymmetric (bright click +
    low sine), so that nonlinearity shifts the mean past the QA DC limit. Keeping
    the peak at 0.94 means soft_clip is a no-op, and the explicit mean removal
    makes the file DC-free without touching the fundamental.
    """
    buf = dsp.hp(buf, 30.0, fs, order=2)
    # pre-empt the master's 6 Hz DC blocker: on a 35-50 Hz kick its local-mean
    # removal overshoots the peak by ~6 %, which wakes up soft_clip and shifts DC.
    buf = dsp.dc_block(buf, fs, fc=10.0)
    buf = _norm(buf, 0.94)
    buf = buf - float(np.mean(buf))
    return dsp.fade(buf, fs, ms=4.0)


# --------------------------------------------------------------------------
# parameter design
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    arch = str(rng.choice(ARCHETYPES, p=_P_ARCH))
    p: dict = dict(
        archetype=arch,
        dirt=float(rng.uniform(0.0, 1.0)),
        colour=float(rng.uniform(0.0, 1.0)),      # brightness / room character knob
    )

    if arch in ("kick", "kick808"):
        char = str(rng.choice(("sub", "punch", "knock", "driven", "thud"),
                              p=(0.18, 0.24, 0.20, 0.18, 0.20)))
        if arch == "kick":
            f1 = float(rng.uniform(28.0, 44.0))                  # root
            f0 = float(rng.uniform(45.0, 72.0))                  # start
            if f0 < f1 * 1.25:                                   # guarantee a real drop
                f0 = f1 * float(rng.uniform(1.25, 1.75))
            pt = float(rng.uniform(0.008, 0.040))                # 8-40 ms drop time
            decay = float(rng.uniform(0.12, 0.45))               # audible decay
            tau = decay / 3.0
            dur = float(np.clip(decay * 1.45, 0.16, 0.80))
        else:                                                    # 808: x1.8-2.4, long
            f1 = float(rng.uniform(26.0, 40.0))
            f0 = float(f1 * rng.uniform(1.80, 2.40))
            pt = float(rng.uniform(0.016, 0.060))
            decay = float(rng.uniform(0.25, 0.75))
            tau = decay / 3.4
            dur = float(np.clip(decay * 1.30, 0.35, 0.80))
        p.update(dur=dur, f0=f0, f1=f1, pitch_tau=pt, decay=decay, tau=tau, char=char,
                 click_ms=float(rng.uniform(2.0, 12.0)),
                 click_hp=float(rng.uniform(1500.0, 7000.0)))

        if char == "sub":
            p.update(click=float(rng.uniform(0.03, 0.12)), harm=float(rng.uniform(0.02, 0.10)),
                     knock=0.0, thud=0.0, res=float(rng.uniform(0.0, 0.10)),
                     drive=float(rng.uniform(1.0, 2.0)))
        elif char == "punch":
            p.update(click=float(rng.uniform(0.24, 0.55)), harm=float(rng.uniform(0.08, 0.30)),
                     knock=0.0, thud=0.0, res=float(rng.uniform(0.0, 0.18)),
                     drive=float(rng.uniform(1.1, 3.2)))
        elif char == "knock":
            p.update(click=float(rng.uniform(0.10, 0.30)), harm=float(rng.uniform(0.04, 0.16)),
                     knock=float(rng.uniform(0.15, 0.50)), thud=0.0,
                     knock_f=float(rng.uniform(150.0, 430.0)),
                     knock_tau=float(rng.uniform(0.012, 0.060)),
                     res=float(rng.uniform(0.0, 0.12)),
                     drive=float(rng.uniform(1.0, 2.6)))
        elif char == "driven":
            p.update(click=float(rng.uniform(0.08, 0.30)), harm=float(rng.uniform(0.15, 0.45)),
                     knock=0.0, thud=0.0, res=float(rng.uniform(0.0, 0.14)),
                     drive=float(rng.uniform(2.4, 8.0)))
        else:                                      # thud: bandpassed-noise body
            p.update(click=float(rng.uniform(0.06, 0.26)), harm=float(rng.uniform(0.02, 0.12)),
                     knock=0.0, thud=float(rng.uniform(0.25, 0.65)),
                     thud_f=float(rng.uniform(320.0, 1000.0)),
                     thud_tau=float(rng.uniform(0.020, 0.090)),
                     res=float(rng.uniform(0.10, 0.38)),
                     drive=float(rng.uniform(1.0, 3.0)))
        # body resonance: a second short modal layer at 90-320 Hz, every character
        p.update(res_f=float(rng.uniform(90.0, 320.0)),
                 res_tau=float(rng.uniform(0.015, 0.100)))

    elif arch == "snare":
        char = str(rng.choice(("tight", "roomy", "crack", "fat"), p=(0.28, 0.22, 0.26, 0.24)))
        decay = float(rng.uniform(0.09, 0.32))                   # audible noise decay
        p.update(char=char, decay=decay,
                 dur=float(np.clip(decay * 1.45, 0.14, 0.44)),
                 tone_f=float(rng.uniform(150.0, 260.0)),
                 tone_drop=float(rng.uniform(0.68, 0.94)),
                 tone_tau=float(rng.uniform(0.015, 0.090)),
                 tone_lvl=float(rng.uniform(0.20, 0.80)),
                 noise_lvl=float(rng.uniform(0.55, 1.00)),
                 bp_f=float(rng.uniform(1200.0, 5000.0)),
                 noise_tau=float(rng.uniform(0.020, 0.110)),
                 room=float(rng.uniform(0.02, 0.40)),
                 snap=float(rng.uniform(0.10, 0.45)),
                 pink=bool(rng.random() < 0.5),
                 crack=0.0)
        if char == "tight":
            p.update(tone_lvl=float(rng.uniform(0.25, 0.60)), noise_tau=float(rng.uniform(0.020, 0.060)),
                     noise_lvl=float(rng.uniform(0.70, 1.00)), bp_f=float(rng.uniform(2400.0, 5000.0)),
                     room=float(rng.uniform(0.02, 0.12)), pink=False)
        elif char == "roomy":
            p.update(tone_lvl=float(rng.uniform(0.20, 0.50)), noise_tau=float(rng.uniform(0.055, 0.110)),
                     noise_lvl=float(rng.uniform(0.60, 0.95)), bp_f=float(rng.uniform(1200.0, 2600.0)),
                     room=float(rng.uniform(0.15, 0.40)), pink=True,
                     decay=float(rng.uniform(0.18, 0.32)))
        elif char == "crack":
            p.update(tone_lvl=float(rng.uniform(0.20, 0.48)), noise_tau=float(rng.uniform(0.025, 0.070)),
                     noise_lvl=float(rng.uniform(0.75, 1.00)), bp_f=float(rng.uniform(2500.0, 5000.0)),
                     room=float(rng.uniform(0.03, 0.16)), pink=False,
                     crack=float(rng.uniform(0.18, 0.50)))
        else:                                      # fat
            p.update(tone_f=float(rng.uniform(150.0, 200.0)), tone_lvl=float(rng.uniform(0.55, 0.80)),
                     tone_tau=float(rng.uniform(0.045, 0.095)), noise_tau=float(rng.uniform(0.035, 0.100)),
                     noise_lvl=float(rng.uniform(0.55, 0.90)), bp_f=float(rng.uniform(1200.0, 2200.0)),
                     room=float(rng.uniform(0.02, 0.12)), pink=True)

    elif arch in ("hat_closed", "hat_open", "ride"):
        char = str(rng.choice(("noise", "metal", "band", "dark"), p=(0.24, 0.32, 0.24, 0.20)))
        if arch == "hat_closed":
            decay = float(rng.uniform(0.018, 0.110))
            dur = float(np.clip(decay * 1.50, 0.05, 0.17))
        elif arch == "hat_open":
            decay = float(rng.uniform(0.080, 0.350))
            dur = float(np.clip(decay * 1.40, 0.15, 0.50))
        else:                                      # ride
            decay = float(rng.uniform(0.250, 0.750))
            dur = float(np.clip(decay * 1.15, 0.40, 0.80))
        tau = decay / (3.4 if char == "dark" else 3.0)
        p.update(dur=dur, decay=decay, tau=tau, char=char,
                 metal=float(rng.uniform(0.45, 1.0)) if char in ("metal", "dark") else 0.0,
                 metal_f=float(rng.uniform(3500.0, 8500.0)),
                 hp_f=float(rng.uniform(4000.0, 6500.0) if char == "dark" else rng.uniform(5000.0, 11000.0)),
                 band_f=float(rng.uniform(6000.0, 12000.0)),
                 band_q=float(rng.uniform(1.2, 2.4)),
                 ratios=_metal_ratios(rng),
                 shimmer=float(rng.uniform(1.5, 8.0)) if arch == "ride" else 0.0)
        if char == "band":
            p["metal"] = float(rng.uniform(0.0, 0.40))

    elif arch == "clap":
        char = str(rng.choice(("tight", "wide", "snap"), p=(0.40, 0.34, 0.26)))
        p.update(char=char, dur=float(rng.uniform(0.16, 0.36)),
                 bp_f=float(rng.uniform(800.0, 3000.0)),
                 q=float(rng.uniform(1.0, 2.0)),
                 bursts=int(rng.integers(3, 6)),
                 gap=float(rng.uniform(0.007, 0.014)),
                 tail_tau=float(rng.uniform(0.030, 0.140)),
                 room=float(rng.uniform(0.02, 0.25)),
                 snap=float(rng.uniform(0.10, 0.50)))
        if char == "tight":
            p.update(bp_f=float(rng.uniform(1600.0, 2600.0)), bursts=int(rng.integers(4, 6)),
                     gap=float(rng.uniform(0.007, 0.010)), tail_tau=float(rng.uniform(0.030, 0.070)),
                     room=float(rng.uniform(0.02, 0.10)))
        elif char == "wide":
            p.update(bp_f=float(rng.uniform(800.0, 1400.0)), bursts=3,
                     gap=float(rng.uniform(0.011, 0.014)), tail_tau=float(rng.uniform(0.070, 0.140)),
                     room=float(rng.uniform(0.10, 0.25)))
        else:                                      # snap
            p.update(bp_f=float(rng.uniform(2400.0, 3000.0)), bursts=int(rng.integers(4, 6)),
                     gap=float(rng.uniform(0.008, 0.012)), tail_tau=float(rng.uniform(0.030, 0.080)),
                     room=float(rng.uniform(0.02, 0.10)))

    else:                                          # rim
        char = str(rng.choice(("wood", "plastic"), p=(0.65, 0.35)))
        p.update(char=char,
                 dur=float(rng.uniform(0.06, 0.18)),
                 f0=float(rng.uniform(380.0, 1400.0)),
                 material=("wood" if char == "wood" else "plastic"),
                 tau0=float(rng.uniform(0.006, 0.060)),
                 size=float(rng.uniform(0.40, 1.30)),
                 click=float(rng.uniform(0.25, 0.80)),
                 click_hp=float(rng.uniform(1800.0, 6000.0)),
                 tick_lvl=float(rng.uniform(0.08, 0.50)),
                 tick_f=float(rng.uniform(1600.0, 3600.0)))
    return p


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def _gen_kick(p: dict, rng: np.random.Generator, fs: int, arch: str) -> np.ndarray:
    n = int(fs * p["dur"])
    t = np.arange(n) / fs
    # instantaneous pitch: fast exponential drop to the root, phase-continuous
    f = p["f1"] + (p["f0"] - p["f1"]) * np.exp(-t / p["pitch_tau"])
    attack = 0.0009 if arch == "kick" else 0.0025
    body = dsp.sine(f, fs) * dsp.env_exp(n, fs, attack=attack, tau=p["tau"])
    body = _norm(body, 0.88)
    # octave-up harmonic: audible weight on phone/laptop speakers
    harm = dsp.sine(f * 2.0, fs) * dsp.env_exp(n, fs, attack=0.001, tau=p["tau"] * 0.35)
    harm = _norm(harm, p["harm"])
    cl = _click(n, fs, rng, peak=p["click"], hp_hz=p["click_hp"], ms=p["click_ms"])

    layers = [Layer(body, label="body"), Layer(harm, label="harm"), Layer(cl, label="click")]
    if p["knock"] > 0:
        kn = _resonance(n, fs, rng, p["knock_f"], p["knock_tau"], p["knock"], hardness=0.72)
        layers.append(Layer(kn, label="knock"))
    if p["thud"] > 0:
        th = dsp.bp(dsp.white(n, rng), p["thud_f"], fs, order=2, q=1.3)
        th *= dsp.env_exp(n, fs, attack=0.0006, tau=p["thud_tau"])
        layers.append(Layer(_norm(th, p["thud"]), label="thud"))
    if p["res"] > 0:
        rs = _resonance(n, fs, rng, p["res_f"], p["res_tau"], p["res"], hardness=0.80)
        layers.append(Layer(rs, label="res"))

    buf = mix(layers, fs=fs, stereo=False)

    drive = float(p["drive"])
    if drive > 1.25:
        buf = fx.sat_adaa(buf, drive)
    return buf


def _gen_snare(p: dict, rng: np.random.Generator, fs: int) -> np.ndarray:
    n = int(fs * p["dur"])
    tone_f = dsp.sweep(n, p["tone_f"], p["tone_f"] * p["tone_drop"], "exp")
    t1 = dsp.sine(tone_f, fs) * dsp.env_exp(n, fs, attack=0.0005, tau=p["tone_tau"])
    t2 = dsp.sine(tone_f * 1.83, fs) * dsp.env_exp(n, fs, attack=0.0005, tau=p["tone_tau"] * 0.45)

    src = dsp.pink(n, rng) if p["pink"] else dsp.white(n, rng)
    noise = dsp.bp(src, p["bp_f"], fs, order=2, q=1.15)
    noise *= dsp.env_exp(n, fs, attack=0.0004, tau=p["noise_tau"])
    snap = dsp.hp(dsp.white(n, rng), 4800.0, fs, order=2)
    snap *= dsp.env_exp(n, fs, attack=0.0003, tau=0.018)
    cl = _click(n, fs, rng, peak=0.30, hp_hz=3200.0, ms=4.0)

    layers = [Layer(_norm(t1, p["tone_lvl"]), label="body"),
              Layer(_norm(t2, 0.20 * p["tone_lvl"]), label="body_h"),
              Layer(_norm(noise, p["noise_lvl"]), label="noise"),
              Layer(_norm(snap, p["snap"]), label="snap"),
              Layer(cl, label="click")]
    if p["crack"] > 0:
        cr = dsp.bp(dsp.white(n, rng), float(rng.uniform(450.0, 950.0)), fs, order=2, q=1.6)
        cr *= dsp.env_exp(n, fs, attack=0.0003, tau=0.025)
        layers.append(Layer(_norm(cr, p["crack"]), label="crack"))

    buf = mix(layers, fs=fs, stereo=False)
    if p["dirt"] > 0.55:
        buf = fx.sat_adaa(buf, 1.2 + 1.6 * p["dirt"])
    t60 = 0.20 + 0.40 * p["colour"] if p["char"] == "roomy" else 0.08 + 0.25 * p["colour"]
    return _room(buf, fs, rng, t60=t60, mix_amt=p["room"])


def _gen_hat(p: dict, rng: np.random.Generator, fs: int, arch: str) -> np.ndarray:
    n = int(fs * p["dur"])
    env = dsp.env_exp(n, fs, attack=0.0002, tau=p["tau"])

    if p["char"] == "band":
        src = dsp.bp(dsp.white(n, rng), p["band_f"], fs, order=2, q=p["band_q"])
    else:
        src = dsp.hp(dsp.white(n, rng), p["hp_f"], fs, order=4)
    noise = _norm(src * env, 0.90)
    layers = [Layer(noise, label="noise")]

    if p["metal"] > 0.15:
        met = _metal_stack(n, fs, rng, p["metal_f"], p["tau"], p["metal"], p["ratios"])
        layers.append(Layer(met, label="metal"))

    buf = mix(layers, fs=fs, stereo=False)

    if arch == "ride" and p["shimmer"] > 0:
        t = np.arange(n) / fs
        lfo = 0.78 + 0.22 * np.sin(2.0 * np.pi * p["shimmer"] * t + rng.uniform(0.0, 6.28))
        buf = buf * lfo

    if p["dirt"] > 0.78:
        buf = fx.sat_adaa(buf, 1.15 + 0.7 * p["dirt"])
    return buf


def _gen_clap(p: dict, rng: np.random.Generator, fs: int) -> np.ndarray:
    n = int(fs * p["dur"])
    src = dsp.bp(dsp.white(n, rng), p["bp_f"], fs, order=2, q=p["q"])

    env = np.zeros(n)
    last_on = 0
    for k in range(p["bursts"]):
        jitter = rng.uniform(-0.6, 0.6) * fs * 0.0012
        on = int(np.clip(k * p["gap"] * fs + jitter, 0, n - 4))
        last_on = max(last_on, on)
        env[on:] += dsp.env_exp(n - on, fs, attack=0.0002, tau=0.0045) * (1.0 - 0.12 * k)
    env[last_on:] += dsp.env_exp(n - last_on, fs, attack=0.002, tau=p["tail_tau"]) * 0.45
    env /= max(float(np.max(env)), 1e-9)

    body = _norm(src * env, 0.90)
    snap = dsp.hp(dsp.white(n, rng), 4200.0, fs, order=2)
    snap *= dsp.env_exp(n, fs, attack=0.0002, tau=0.020)
    snap = _norm(snap, p["snap"])

    buf = mix([Layer(body, label="bursts"), Layer(snap, label="snap")], fs=fs, stereo=False)
    if p["dirt"] > 0.55:
        buf = fx.sat_adaa(buf, 1.2 + 1.0 * p["dirt"])
    return _room(buf, fs, rng, t60=0.13 + 0.18 * p["colour"], mix_amt=p["room"])


def _gen_rim(p: dict, rng: np.random.Generator, fs: int) -> np.ndarray:
    n = int(fs * p["dur"])
    modes = dsp.modal_bank(p["f0"], p["material"], tau0=p["tau0"], size=p["size"],
                           rng=rng, jitter=0.02)
    exc = dsp.contact_burst(n, fs, hardness=0.82, rng=rng)
    body = _norm(dsp.modal(exc, modes, fs), 0.80)

    tick = dsp.bp(dsp.white(n, rng), p["tick_f"], fs, order=2, q=2.5)
    tick *= dsp.env_exp(n, fs, attack=0.0002, tau=0.006)
    tick = _norm(tick, p["tick_lvl"])
    cl = _click(n, fs, rng, peak=p["click"], hp_hz=p["click_hp"], ms=3.0)

    buf = mix([Layer(body, label="modal"),
               Layer(tick, label="tick"),
               Layer(cl, label="click")], fs=fs, stereo=False)
    if p["dirt"] > 0.70:
        buf = fx.sat_adaa(buf, 1.1 + 0.9 * p["dirt"])
    return buf


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def generate(params: dict, rng: np.random.Generator, fs: int = FS) -> Render:
    p = params
    arch = p["archetype"]

    if arch in ("kick", "kick808"):
        buf = _gen_kick(p, rng, fs, arch)
    elif arch == "snare":
        buf = _gen_snare(p, rng, fs)
    elif arch in ("hat_closed", "hat_open", "ride"):
        buf = _gen_hat(p, rng, fs, arch)
    elif arch == "clap":
        buf = _gen_clap(p, rng, fs)
    elif arch == "rim":
        buf = _gen_rim(p, rng, fs)
    else:
        raise ValueError(f"unknown drum archetype {arch!r}")

    buf = _finish(buf, fs)

    # frequency tag for the descriptor: the most identifying pitch of the voice
    tag = p.get("f0", p.get("hp_f", p.get("bp_f", p.get("tone_f", p.get("metal_f", 0.0)))))
    descriptor = f"{_DESCRIPTOR[arch]}{int(round(float(tag)))}"
    return Render(
        buf=buf, fs=fs, stereo=False,
        meta=dict(family=FAMILY, archetype=arch, character=p.get("char", arch),
                  loop=False, mono=True, descriptor=descriptor,
                  dirt=round(p["dirt"], 3), dur=round(p["dur"], 4)),
    )
