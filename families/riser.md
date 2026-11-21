# riser — build, tension, sweep, uplift

Mono one-shots, 1.0-4.0 s. `DESCRIPTORS = ["riser","build","sweep","tension","uplift"]`.

A riser is a texture that accelerates. The spine of every render is a
band-passed noise sweep whose centre climbs exponentially (`dsp.moving_bp`),
with amplitude `t^k`. On top of that the family covers five *structures* and
six *spectral voices*, so 60 candidates are 60 different gestures, not 60
filter settings.

## Structure (the gesture)

| struct  | weight | what it is |
|---------|--------|------------|
| noise   | 0.38 | the classic sweep + a tonal stack that locks on a root in the last 12-32 % |
| shepard | 0.15 | octave-spaced sines with Gaussian log-frequency windows (Shepard 1964) — the endless-rise illusion |
| roll    | 0.18 | accelerating hit train, 1.2 Hz -> 14-28 Hz (drum-roll build) |
| grain   | 0.14 | accelerating granular chime cloud, 6-14 grains/s -> 40-80/s, grains rise with time |
| siren   | 0.15 | dominant tonal glide over 2-4 octaves with deep vibrato (18-60 cents) |

Only "noise" is the textbook riser; the others are commercial stalwarts
(trailer rolls, sci-fi Shepard lifts, granular shimmer). If you want the pack
to be strictly classic risers, set the `STRUCTS` weights in `sample_params`.

## Layers (noise structure)

| layer | band | role |
|-------|------|------|
| bed    | 30 Hz - 16 kHz pink/white/brown | keeps the LTAS continuous; without it a 300 Hz->12 kHz sweep leaves an interior spectral hole and the audit fails |
| band   | moving centre, q 1.2-2.8 | the sweep itself |
| band2  | centre x 1.5-4 (42 % of candidates) | double-sweep width |
| fill   | fixed ~800 Hz q 4 | body / low-mid closure |
| low    | root/4 sine glide | weight on small speakers |
| tone   | additive chord, glides to the root, locks, vibrato deepens toward the lock | "it goes somewhere" |
| presence | fixed resonant formant 450-4800 Hz | the voice of the riser |
| air    | HP 5.2 kHz, opens as t^3.5-6.5 | 2-6 kHz air band |
| ticks  | accelerating 2.5 Hz -> 10-24 Hz click train | optional build layer |
| sub    | 38-56 Hz sine | optional rumble |

Every layer is peak-normalised to an explicit target (`_norm`) before mixing.
Nothing is left to emerge from the sum.

### Why the band layer is RMS-flattened

A moving bandpass over pink/brown noise is 20+ dB hotter at the bottom of the
sweep than at the top. Peak-normalising the layer then leaves the entire upper
sweep under the tonal layer — the render sounds like a tone with a lump at the
start, not a riser. `_flat_rms` divides out the 70 ms local RMS, so the band
stays at a constant loudness while its centre climbs. This was the single
biggest quality fix in the family.

### Other decisions

- **Tonal lock.** `_pitch_curve` glides root/4..root/8 up to the root over
  `1-lock_frac`, then holds. Root notes A2-E4. Vibrato depth ramps as `t^2`
  so the tension is at the top.
- **Chords, not one interval set.** Eight chord/interval stacks (unison,
  octave, fifth, major, minor, tritone, bright stack, sub-octave) chosen per
  candidate, plus one random upper harmonic. Two "tonal" risers are not the
  same chord at another root.
- **Envelope shapes.** `pow` (t^k), `exp` (late surge), `staged` (slow stage
  plus accelerated second stage — the two-step trailer build). Same endpoints,
  different shoulder.
- **Finishing.** final HP at `max(30, hpf)` order 4 (hpf is log-distributed
  25-3000 Hz: some renders are thin air-only risers), mean removal, re-fade.
- **Peak target 0.85 before master.** `master()` soft-clips above 0.995 and
  tanh compresses an asymmetric peak unequally, which walks DC into the file.
  Returning at 0.85 keeps soft_clip a no-op; master re-normalises to -6 dBFS.

## Acceptance

```
gen_candidates.py riser --count 12   -> 12/12 pass audit, 1 near-dup, 11/12 unique [OK]
gen_candidates.py riser --count 60   -> 60/60 pass audit, 20 near-dups, 40/60 unique (target 27) [OK]
```

Energy profile: every candidate peaks in the last 30 % (deciles 7-9). `holes`
max 0, `alias_db` -99 (no mirrored peaks found), crest 9.2-23.4 dB.

Note on the contract's "loudest moment within 0-15 ms" rule: that rule is for
impacts. A riser's loudest moment is at the end **by definition**; the
transition/sweep archetypes instead land on a hard stop or a short ring-out.

## CORE ISSUES

1. **`dsp.brown` returns a straight line.** It does
   `out -= signal.detrend(out)`, but `signal.detrend` *already* returns the
   detrended residual, so the subtraction re-adds the linear trend. Measured:
   two seeds correlate 1.000000, and the linear-fit residual is 7.9e-16.
   Every "brown" layer was identical across seeds (and was a ramp, not noise).
   Workaround: local `_brown()` builds 1/f^2 noise by spectral shaping.
   Suggested fix: `out = signal.detrend(np.cumsum(white))` (no `-=`).
2. **`dsp.pink` carries a random DC offset.** The spectrum is divided by
   `sqrt(f)` with `f[0] = 1.0`, so the DC bin is left at full white-noise
   amplitude. Measured mean of a 200k-sample pink buffer: 0.062 (white: 0.003).
   Workaround: every render is DC-blocked, mean-removed, and finished with a
   4th-order HP. Suggested fix: zero the DC bin (`spec[0] = 0`).
3. Core and tools changed three times during this run (dedup features, the
   audit's loop rule, `fade_asym`, the master loop policy). Every result above
   was re-rendered against the final state; not a defect, just be aware that
   numbers from earlier runs are not comparable.
