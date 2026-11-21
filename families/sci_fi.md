# sci_fi — teleports, energy, force fields, doors, machinery, holograms

`families/sci_fi.py` · descriptors: `teleport, energy, field, door, machine, holoscan`
· mono one-shots 0.57–2.40 s (spec 0.4–2.5); `machine` is a mono loop

## Archetypes

| archetype | share | layers |
|---|---|---|
| `teleport` | 22 % | `moving_bp` noise suction sweep, inharmonic FM body, 120→38 Hz sub, granular shimmer over an FM source, arrival/lift click + glass ring |
| `energy` | 20 % | rising FM (`index` grows with the sweep), rising air band, velvet crackle, then a metal modal hit + sub drop. `charge` = one hit near the end; `surge` = hit at 18–30 % + second smaller discharge at 74 % |
| `field` | 17 % | FM hum (integer-cycle) + saw harmonic, two flavours: `hum` (hum dominates) and `crackle` (wobbling band-passed sizzle dominates), opened by an engage click + ping |
| `door` | 15 % | servo `moving_bp` sweep ±, hydraulic low band, 2–3 metal/ceramic mechanism clicks, latch ring, thud on close |
| `machine` | 12 % | motor hum, wrap-around metal clanks every period, half-period tick, seam-folded air; `meta={"loop": True}` |
| `holoscan` | 14 % | granular shimmer + FM scanner pings + air bed; flavours `pings`, `shimmer`, `sweep` change which layer dominates |

Every archetype gets a log-uniform device-size multiplier (`scale` 0.65–1.5) applied to
all of its frequencies, and a bright/dark `tilt` (−1…+1). Those two axes are what keep
60 candidates from being 60 recolours.

## Parameter spread (designed per archetype, not sampled blindly)

| field | range |
|---|---|
| dur | teleport 0.55–1.35, energy 0.85–1.9, field 1.6–2.5, door 0.8–1.6, machine period×(2–5) 0.48–2.1, holoscan 0.9–2.0 |
| FM ratio | 1.41 / 2.37 / 3.51 / 4.73 (teleport, energy, holoscan), 2.0 / 3.0 (field) |
| FM index | 1.0–4.5 (sweeps decay with time or grow with the riser) |
| sweeps | `dsp.moving_bp` centres 300–5500 Hz, q 1.0–2.2 |
| door | servo 320–2400 Hz, ring 120–620 Hz, thud 48–78 Hz, open/close 50/50 |
| machine | hum 48–96 Hz, clank 180–520 Hz, τ 80–220 ms |
| dirt | 0–1: >0.6 tape (drive 2.0–2.9, hiss −70…−62), >0.45 oversampled tanh, >0.3 ADAA |
| space | mix 0.02–0.40, t60 0.4–1.3 s, synthesized IR only |

## Level structure / finish

Every layer is peak-normalised to an explicit target before `mix` (body ≤ 0.5, sub ≤ 0.34·subgain,
air ≤ 0.25, transient 0.82, modal hit 0.62–0.85). A sustained riser sits ~10 dB under its
arrival hit, so the peak is always a transient. Finish chain: tilt → dirt → reverb →
`hp 32 Hz` (`hp 40 Hz` for `field`) → `dc_block 8 Hz` → 25 ms release → 4 ms fade →
peak 0.95 → subtract mean. Master then normalises to −6 dBFS (−24 LKFS for the loop).

Two transients are deliberate exceptions to "peak in the first 15 ms": `teleport in` and
`energy` are risers whose loudest moment is the arrival. Their sustained layers stay under
the hit, and the hit itself is a real 1–2 ms transient. `door` is a gesture in the same
sense: mechanism clicks open it, but the loudest event is the latch ring / closing thud at
the end of the travel (measured peak 0.52–1.13 s over the 60 set). Every non-gesture
archetype peaks in the first 0–70 ms (`field` engage, `machine` clank at t=0, `holoscan`
start click).

## Acceptance (this machine, current `qa.dedup` with perceptual blocks)

| run | result |
|---|---|
| `--count 12` | exit 0, **12/12 pass**, 3 near-dups |
| `--count 60` | exit 0, **60/60 pass**, 2 near-dups, 58/60 unique (target ≥ 27) |

Nothing was listened to; the human does that.

## CORE ISSUES

1. **`master` soft-clip can write DC into a hum.** `render.master` runs `soft_clip(0.995)`
   before peak normalisation. Any render that arrives with peak > 0.995 (a mix of hum + hit
   + reverb easily does) is shaped by `tanh`, and an asymmetric waveform gains a DC step —
   one `field` candidate measured `dc = 0.004`, 4× the audit limit. Worked around by
   peak-normalising to 0.95 *before returning* so the soft-clip is a no-op. Affects every
   family whose layer sum can exceed full scale.
2. **`qa.audit` dc_offset measures the clip mean, not DC.** Any partial whose frequency is
   not an integer number of cycles over the window (after the master's 4 ms end fade shortens
   the effective window) reads as DC. A 55 Hz hum over 1.8 s is enough. Worked around by
   snapping the field hum to an integer cycle count *and* removing the mean as the last
   operation after every fade. A proper fix is to measure a true DC estimate (e.g. mean of a
   high-passed signal) rather than the raw mean.
3. **`master` fades loops.** `render.master` applies `fade(ms=4)` to loop material too, so the
   `machine` loop always has a 4 ms dip at the seam. The family builds the loop circularly
   (clank tails wrap via modulo, air noise is seam-folded), but a truly gapless loop cannot
   leave this pipeline. Suggest skipping the fade when `loop=True` (or using a crossfaded
   loop-point check).
4. **`mix` extends to the longest layer.** Event layers with an `offset_s` (door latch, thud)
   push the buffer past the requested duration; the family hard-trims to `n` and applies a
   25 ms release so the cut modal tails do not tick. Worth documenting in `mix`.
