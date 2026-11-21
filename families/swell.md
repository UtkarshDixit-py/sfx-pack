# swell — reverse swells, whooshes, transitions, air

Mono one-shots, 0.4-2.5 s. `DESCRIPTORS = ["swell","whoosh","reverse","transition","air"]`.

Five gestures, each with its own envelope maths and layer set:

| type | weight | shape | content |
|------|--------|-------|---------|
| reverse   | 0.28 | t^k rise, hard stop (4-14 ms release), peak in the last decile | dense bright noise + an inharmonic metal partial cluster (1, 2.76, 5.40, 8.93, 13.34 x 280-900 Hz) + low "suck" |
| revsynth  | 0.18 | t^k rise, hard stop | a saw chord whose lowpass opens: dark copy (LP 550) crossfaded into a bright copy (LP 6.5 k) by u^2, plus a detuned twin |
| whoosh    | 0.24 | skewed bell, peak at 72-90 % of the file | moving bandpass q 1.6-3.0, sweeps up (60 %) or down (40 %); 0.4-1.6 s |
| transition| 0.16 | build to full level exactly when the hit lands, then clear out | noise rise into a bright transient in the last 60-140 ms: 3.2 kHz HP snap + ping at 1.4-4.2 kHz + sub thump, all decaying to the end |
| air       | 0.14 | smooth rise, long gentle landing (60-250 ms) | HP-noise air swell, no hard stop |

All types also carry a low-level broadband bed (LTAS continuity), a fixed
presence band, an air layer (5.2 kHz HP, opening as t^2.5-6), and a root/4 low
glide.

## Decisions

- **The transition hit is not at sample -1.** It starts 60-140 ms before the
  end and decays *out*: the rise lands on the transient, which then rings away.
  Placing the hit at the very last sample would leave the fade/edge logic to
  chop its tail. Peak is at 88-97 % of the file (measured).
- **Revsynth "filter opening"** is a crossfade between a dark and a bright copy
  of the same chord, because the core has no time-varying filter and wrapping
  `moving_bp` around a harmonic source would band-limit the low end. At 25-95 %
  open_frac the sweep never resolves before the stop, so it always reads as
  "not yet there".
- **Whooshes are mono only** (brief): no panning, no haas, no stereo widening.
  The sense of movement comes from the moving centre and the skewed bell only.
- **Bell skew.** `_bell` normalises `u^a (1-u)^b` so the maximum sits at the
  requested fraction (72-90 %); a symmetric bell reads as a pass *through* the
  listener, a late one reads as a transition *into* something.
- **No reverb on revsynth/reverse.** A tail after a hard stop destroys the
  stop; the partial cluster supplies the "space" instead.
- **Finishing.** Tilt LP (2.6-18 kHz), optional saturation, DC block at 12 Hz,
  4th-order HP at 25-1500 Hz, mean removal, re-fade, peak 0.85 before master
  (keeps `master()`'s soft-clip a no-op; it re-normalises to -6 dBFS).

## Parameter spread (per candidate)

`dur, fc0, fc1, k (1.4-4.0), q, tail_ms, bell_peak, hit_hz, hit_len, colour
(white/pink/brown), tilt_lp, presence_hz/gain/q, air_gain, air_pow, bed_gain,
low_gain, hpf, chord, open_frac, shimmer_gain/base, detune_cents, env_shape,
dirt, root_note`. Every seed changes several of these; the type itself changes
the layer set, which is what keeps the dedup clusters small.

## Acceptance

```
gen_candidates.py swell --count 12  -> 12/12 pass audit, 3 near-dups, 9/12 unique [OK]
gen_candidates.py swell --count 60  -> 60/60 pass audit, 20 near-dups, 40/60 unique (target 27) [OK]
```

Durations 0.40-2.49 s (brief: 0.4-2.5). `holes` max 0, `alias_db` -99,
peaks land in deciles 7-9 for every type.

## CORE ISSUES

1. **`dsp.brown` is a linear ramp, identical for every seed.** It does
   `out -= signal.detrend(out)`; `signal.detrend` already returns the
   detrended residual, so the trend is added back. Measured: seed-to-seed
   correlation 1.000000 at 200k samples; linear-fit residual 7.9e-16.
   Workaround: local `_brown()` (spectral 1/f^2). Suggested fix:
   `out = signal.detrend(out)`.
2. **`dsp.pink` has a random DC offset** (the DC bin is not attenuated:
   `f[0] = 1.0`, then `spec /= sqrt(f)`). Measured mean 0.062 on a 200k pink
   buffer. Every render is DC-blocked and mean-removed, so it is masked here,
   but it is a bad primitive for anyone who forgets. Suggested fix:
   `spec[0] = 0`.
3. `master()` takes a different branch for files < 0.5 s (global mean removal
   instead of the 6 Hz one-pole). Whooshes can be 0.40 s, so the shortest
   candidates are mastered slightly differently from the rest; this is benign
   for noise/tonal swells, but worth knowing when comparing levels.
