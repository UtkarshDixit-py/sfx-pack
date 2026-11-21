# bass_synth

Pitched synth-bass one-shots in four archetypes. Every candidate is a single root
note synthesised at its own frequency — **no take is pitch-shifted into the 12
notes**.

## Interface

```python
FAMILY = "bass_synth"
DESCRIPTORS = ["bass", "acid", "reese", "fm"]
sample_params(rng) -> dict            # includes note, key, octave
generate(params, rng, fs) -> Render   # mono, 0.30-1.20 s
```

## 12-note coverage

Same scheme as the other pitched families: the note comes from a module-level
round-robin cursor, `note = NOTE_NAMES[c % 12]`, and the same call index `c`
drives the diversity stratification. `generate` resolves the note with
`dsp.note_hz(note, octave)` (octave 2 = C2..B2, 65.4-123.5 Hz) and oscillates at
that frequency. `generate` is a pure function of `params`, so stored params
re-render exactly. A `--count 12` run covers C..B exactly once; `--count 60` is
12 notes × 5 draws.

## Archetypes

| archetype | signal | descriptor |
|---|---|---|
| acid | saw/square → resonant lowpass, cutoff swept by an envelope | `acid_<note>` |
| reese | two saws detuned 3-40 cents (+ soft sub), static lowpass | `reese_<note>` |
| lead | duty-square/saw with vibrato → static lowpass | `bass_<note>` |
| fm | carrier at root, decaying modulation index, ratio 0.5-7 → lowpass | `fm_<note>` |

Weights: acid 32%, reese 25%, lead 25%, fm 18%.

## Parameters (per seed)

* duration 0.30-1.20 s; attack 2-15 ms (capped at 15 ms to keep the loudest
  moment inside the transient-first window); decay curve 0.5-2.5.
* **envelope shape** `(c + c//12) % 4` ∈ {pluck, ring, hold, trem} — see Diversity.
  pluck/ring decay tau from a low-discrepancy sequence (0.05-0.22 s / 0.50-1.20 s);
  hold = flat sustain + gated release; trem = sustained with 4-10 Hz tremolo,
  depth 0.35-0.85.
* acid: cutoff start 4-30× root, end 1.2-5× root, sweep 0.03-0.40 s, Q 1.5-9.0,
  drive 1.2-3.2, wave saw/square, duty 0.15-0.70, click 0.05-0.24.
* reese: detune 3-40 cents, lowpass 2-16× root, sub 0.10-0.50, drive 1.0-2.6.
* lead: wave square/saw, duty 0.15-0.70, lowpass 2-14× root, vibrato 0-1%,
  3-8 Hz, drive 1.0-2.4, click 0-0.10.
* fm: ratio {0.5, 1, 1.5, 2, 3, 5, 7}, index 1-9 decaying over 0.02-0.30 s,
  lowpass 2-16× root, pitch drop 1.2-2.8× over 10-35 ms, drive 1.0-2.4.
* shared extras: spectral tilt lowpass 400-15 kHz on a golden-ratio sequence;
  optional 2f/3f bandpass boosts (h2 0-0.35, h3 0-0.20); optional highpassed edge
  0-0.40 at 600-4000 Hz.

## Diversity

core's dedup is a **min-over-blocks** comparison of {MFCC mean, MFCC std, log-mel,
24-pt RMS envelope, timbre scalars} at 0.93. Measurement of the 60-candidate set
showed the `timbre` block is saturated (every pair ≥0.93) and the spectral blocks
stay similar for any harmonic bass; the **envelope** block is the real separator.
So the amplitude envelope is treated as a first-class design axis: the four shapes
are stratified by `(c + c//12) % 4`, which hands all four to the 12 notes of a
short run *and* to the five draws of each note in a 60-draw run. Decay and tilt
follow low-discrepancy (Kronecker) sequences so they never bunch.

Measured (seed0 = 1000):

* `--count 12`: 12/12 pass audit, 11 unique of 12 (target ≥ 5) — **OK**
* `--count 60`: 60/60 pass audit, 35 unique of 60 (target ≥ 27) — **OK**

## CORE ISSUES

1. **No time-varying resonant lowpass in core.** `moving_bp` is a bandpass and
   would strip the fundamental a bass needs, so acid uses a local `_swept_lp`: an
   RBJ cookbook 2-pole lowpass whose cutoff is updated every 128 samples while the
   biquad state (`zi`) is carried, giving a continuous, alias-free sweep with the
   fundamental intact. Block-wise rather than per-sample for Python speed. This is
   a documented workaround, not a core edit.
2. **DC offset from the master soft-clip.** At 1.0 peak `render.master`'s
   `soft_clip(0.995)` engages and shifts the mean of a low-frequency one-shot,
   which `qa.audit` reports as `dc_offset` (this family failed 11/12 that way at
   first, dc ≈ +0.004). Fix: deliver at **0.9 peak** + final `dc_block`. Same
   root cause as worker A's `sub`.
3. **Dedup block saturation (metric limitation).** The `timbre` block (5 scalars,
   mean-centred) barely discriminates harmonic bass; envelope vs spectral blocks
   do the work. Not patched — `qa.py` is core. Informed the envelope-shape design.
