# stab

Pitched melodic stabs, plucks, bells and chords. Every candidate is a single root
note synthesised at its own frequency — **no take is pitch-shifted into the 12
notes**. Chords are built by transposing inside the render (root + third + fifth)
and are named by the root.

## Interface

```python
FAMILY = "stab"
DESCRIPTORS = ["stab", "pluck", "chord", "bell"]
sample_params(rng) -> dict            # includes note, key, octave
generate(params, rng, fs) -> Render   # mono, 0.20-0.80 s
```

## 12-note coverage

The note comes from a module-level round-robin cursor, `note = NOTE_NAMES[c % 12]`;
`generate` resolves it with `dsp.note_hz(note, octave)` and every voice is
synthesised at that frequency. Register (octave) **varies per draw** over 3/4/5
(≈130-988 Hz) as a diversity axis, so the same note recurs at different octaves
across a 60-draw set. `generate` is a pure function of `params`; a `--count 12`
run covers C..B exactly once, `--count 60` is 12 notes × 5 draws.

## Archetypes (weights)

| archetype | method | descriptor |
|---|---|---|
| pluck (35%) | `dsp.karplus` extended Karplus-Strong, noise excitation | `pluck_<note>` |
| stab (30%) | marimba-like stack: sine + 2f (+ optional 3f), fast decay | `stab_<note>` |
| chord (20%) | root + minor/major third + fifth, bright stacked tones | `chord_<note>` |
| bell (15%) | `dsp.fm` inharmonic ratio, decaying index | `bell_<note>` |

## Parameters (per seed)

* note (round-robin), octave ∈ {3, 4, 5}; duration 0.20-0.80 s.
* pluck: Karplus decay 0.988-0.9997, brightness 0.10-0.90, click 0.03-0.12.
* stab: decay tau 0.04-0.24 s, 2f gain 0.18-0.55, 3f gain 0-0.28, 2 or 3 partials.
* bell: FM ratio {1.41, 2.0, 2.4, 2.76, 3.5, 4.0, 5.0}, index 1-9 decaying over
  0.015-0.15 s, decay 0.04-0.28 s, click 0.02-0.10.
* chord: quality maj/min, decay 0.06-0.26 s, detune 0-10 cents, strum 0-8 ms,
  2f/3f gains 0.15-0.45 / 0-0.18, click 0.02-0.09.
* shared extracts: lowpass "tone" 1.5-17 kHz and optional highpassed edge 0-0.35
  at 700-4000 Hz; a short highpassed click under every archetype.

Every layer is normalised to an explicit peak target. Delivered at 0.9 peak (see
CORE ISSUES #2); final `hp` at 45 Hz.

## Diversity

core's dedup is a **min-over-blocks** comparison of {MFCC mean, MFCC std, log-mel,
24-pt RMS envelope, timbre scalars} at 0.93. Melodic material at octave 3-5 is far
more separable in these blocks than sub-bass, so register + recipe variation does
most of the work: Karplus decay/brightness, FM ratio/index, partial count, chord
gains/strum, decay, tone and edge. Two draws of a note therefore differ in recipe
and often in octave.

Measured (seed0 = 1000):

* `--count 12`: 12/12 pass audit, 10 unique of 12 (target ≥ 5) — **OK**
* `--count 60`: 60/60 pass audit, 35 unique of 60 (target ≥ 27) — **OK**

## CORE ISSUES

1. **DC offset from the master soft-clip.** At 1.0 peak `render.master`'s
   `soft_clip(0.995)` can engage and shift the mean on an asymmetric low tone,
   which `qa.audit` reads as `dc_offset`. Fix used here (and the bass families):
   deliver at **0.9 peak** + final `dc_block`. `render.master` / `qa.py` are core
   and were not edited.
2. **`dsp.karplus` integer-delay pitch quantisation.** The delay line is rounded
   to `round(fs/freq)` samples, so the pluck pitch is quantised to `fs/N` — a few
   cents at octave 4, more at octave 5. This is intrinsic to the physical model
   (and the brief specifies `dsp.karplus`), so it is documented, not patched.
3. No missing core function otherwise; no workarounds beyond #1.
