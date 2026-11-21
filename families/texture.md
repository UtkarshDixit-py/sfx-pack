# texture — stereo, seamless atmospheric beds

Stereo, `loop=True`, 4.0-11.5 s (never over the 12 s budget).
`DESCRIPTORS = ["texture","atmos","drone","bed","room","air"]`.

Six types: **room** (broadband tone, faint hum), **wind** (gusting band plus a
slow moving centre), **machinery** (50/60 Hz hum + motor beat), **drone**
(root/fifth/octave with detuned beating), **pad** (detuned saw chord, slowly
opening), **tension** (low drone + tritone/minor-2nd upper interval).

## Bed synthesis (McDermott-style texture statistics)

`_band_am` splits noise into 8-10 contiguous octave bands (45 Hz - 14 kHz,
q 1.30-1.65) and gives **each band its own slow random amplitude modulator**
(0.02-0.8 Hz depending on type, depth 0.08-0.65). Bands are RMS-normalised and
then given an explicit spectral slope of -1.5 to -5 dB/oct, so the result is a
shaped bed rather than white hiss while still covering every band (no spectral
holes: `holes = 0` across all 60).

The bed is built **twice with independent noise** — one per channel — and
panned ±0.55-0.95. That is real decorrelation, not a haas delay, and the mono
sum stays intact. Measured L/R correlation across the 60: 0.22-0.98, median
0.72; mono-sum level 0.82-1.00 of a single channel. Types with a dominant
centred tonal layer (drone, pad) sit at the high end, which is correct — they
should stay mono-compatible.

## Loop pipeline (order matters)

1. build layers at `dur + 150 ms`
2. **all shaping first**: tilt LP, HP, optional saturation/dull-LP. Any causal
   filter after the crossfade starts from zero state at the head, does not
   match its own state at the tail, and leaves an audible step at the loop
   point. This was a real defect in the first version (measured seam step 4-11x
   the p99.9 sample-to-sample difference); after the reorder it is <= 1.73x.
3. **equal-power crossfade** of the last 150 ms into the first 150 ms:
   `y[:F] = tail*sqrt(1-w) + head*sqrt(w)`. At the wrap point `y[0]` is the
   natural continuation of `y[-1]`, and by `y[F]` the head is back intact.
4. **tonal snapping**: every tonal frequency is rounded to a whole number of
   cycles over the loop (`snap(f) = round(f*n/fs)*fs/n`), including detuned
   twins and the hum harmonics, so tonal content is phase-continuous at the
   seam. Pitch error < 0.6 % at 4 s, < 0.2 % at 12 s.
5. **circular DC block**: run the 6 Hz blocker over two copies of the loop and
   keep the second, so the filter starts the loop with the state the loop's end
   leaves behind. A one-pass causal filter would put its startup step at the
   seam.
6. mean removal, peak target 0.80. `master()` skips its edge fade for loops, so
   the family deliberately does **not** fade its own ends.

Verification used: `|x[0] - x[-1]|` against the 99.9th percentile of
`|diff(x)|`. Across the 60-set: median 0.32x, max 1.73x — i.e. the wrap point
is indistinguishable from any other sample boundary.

## Acceptance

```
gen_candidates.py texture --count 12 -> 12/12 pass audit, 0 near-dups, 12/12 unique [OK]
gen_candidates.py texture --count 60 -> 60/60 pass audit, 1 near-dup, 59/60 unique (target 27) [OK]
```

Durations 4.03-11.45 s. Type mix at 60: room 13, wind 13, drone 13,
machinery 8, tension 7, pad 6. All at -24 LKFS (loop target), peaks about
-13 dBFS, `holes` 0, `alias_db` -99, `dc` ~ 0.

## CORE ISSUES

1. **`core.dsp.fade_asym` could not take a 2-D buffer (blocking).** Until it
   was fixed upstream mid-run, `y[:mi] *= np.linspace(0, 1, mi)` hit
   `(mi, 2) *= (mi,)` and raised, so `master()` failed and **every stereo
   render was skipped** by `gen_candidates.py` (0 rendered, 0 % pass). There is
   no way for a family to work around this from the outside: `render_family`
   always calls `master`. I did not patch core; I logged it and re-ran when
   core gained `ri = ri[:, None]` (current `fade_asym` handles `y.ndim > 1`).
   If that guard is ever removed, this family and every other stereo family
   will stop rendering.
2. **`dsp.brown` returns a linear ramp, identical for every seed.** It does
   `out -= signal.detrend(out)`; `detrend` already removes the trend, so the
   subtraction re-adds it (seed-to-seed correlation 1.000000, linear-fit
   residual 7.9e-16). For a stereo bed this meant L and R were perfectly
   correlated and the loop could never line up. Workaround: local `_brown()`
   (spectral 1/f^2). Suggested fix: `out = signal.detrend(out)`.
3. **`dsp.pink` carries a random DC offset.** `f[0] = 1.0` then
   `spec /= sqrt(f)` leaves the DC bin at full white-noise amplitude; measured
   mean 0.062 on a 200k buffer. On a loop a DC offset becomes a startup
   transient at the seam. Workaround: mean removal + circular DC block.
   Suggested fix: `spec[0] = 0`.
4. **Loop fade policy changed mid-run.** `master()` used to run
   `fade_asym(in 0.4 ms, out 8 ms)` on *every* file including loops, which put
   a level hole at the loop point (the 8 ms out-ramp averages -6 dB). It now
   skips the fade when `meta["loop"]` is true, and `qa.audit(..., loop=True)`
   exempts loops from the edge-click rule — both correct. My earlier
   `_seam_compensate` workaround was removed once the policy changed.
5. **Residual seam step from `master()`'s own DC block.** `master()` still runs
   `dsp.dc_block` on the loop from zero state after the family returns. Because
   the family has already removed sub-6 Hz energy circularly, the residual step
   is tiny (the measured seam ratio above includes it), but a core-side fix
   would be to skip `dc_block` when `loop=True` and let the family own it.
