# tape_vinyl — tape transport artefacts and vinyl surface noise

`families/tape_vinyl.py` · descriptors: `crackle, scratch, backspin, tapestop, hiss, vinyl`
· mono one-shots 0.31–4.9 s (spec 0.3–6 s); two loop beds with `meta={"loop": True}`

## What it is

A character/texture family, not a synth. Two pieces of material are synthesised in the
file and then driven by a transport model:

- `_music_src`, `_perc_src`, `_bass_src`, `_vox_src` — "something playing" (saw chord +
  Karplus plucks; a kick/hat/snare beat; a gated bass; a formant-swept voice)
- `_stab`, `_noise_record` — material to scratch

| archetype | share | source | transport |
|---|---|---|---|
| `tape_stop` | 13 % | chord / perc / bass / vox | rate 1 → (1−u)^1.2…2.8 → 0.02–0.09, 34–52 % stop window, HF loss blended in as the motor stalls |
| `tape_degrade` | 10 % | chord / perc / bass / vox | rate LFO 0.4–1.8 Hz at 0.4–1.1 % + 9 Hz flutter, 1–3 smoothed dropouts, `bw` 6–12 kHz |
| `tape_start` | 10 % | chord / perc / bass / vox | rate 0.03 → 1 over 120–350 ms with decaying wobble, start clunk |
| `tape_hiss` | 10 % | noise floor (pink/brown/white) | `fx.tape` hiss −24…−12 dB, `bw` 7–13 kHz, one-shot with ADSR |
| `tape_hiss_loop` | 10 % | two decorrelated noise floors | loop, 2–4 s, 3 cycles of slow "breathing" per loop |
| `wobble` | 8 % | chord / perc / bass / vox | own rate LFO 0.4–2.0 Hz at 0.4–1.2 %, plus `fx.tape(wow=0)` |
| `scratch` | 14 % | stab / vox / noise | baby (fwd/back), chirp (2.1× fwd/back + overshoot), transform (0.85× + 8–15 Hz fader chop); 3 ms direction dips, friction noise ∝ |rate| |
| `backspin` | 9 % | stab / vox / noise | +0.9 for 10 % of the file, then −1.2…−2.2 × exp(−t/0.22…0.50) |
| `vinyl_crackle_loop` | 10 % | pink/brown/white bed + rumble | `fx.vinyl` 2–12 crackle/s, `hi` 6.5–9 kHz, 0.4–1.2 extra big pops/s, loop |
| `vinyl_pop` | 6 % | surface bed | 1–3 damped 4 ms resonances (0.9–4.2 kHz) + 70 Hz thump under each |

## Tape and vinyl character

- Tape chain: `sat_adaa` drive 1.3–2.6 → wow 0.06–0.4 % → flutter 0.02–0.18 % → gap-loss
  LP at 6–12 kHz → hiss. Via `fx.tape` for the modelled stages; the wobble rate itself is
  generated in-family (see CORE ISSUES #3).
- Scratch playback uses a **signed** rate curve, integrated to a sample index and linearly
  interpolated: rate is in samples/sample, so a negative rate plays the source backwards.
  Index excursions out of the source are masked with a 3 ms ramp instead of being held.
- Vinyl: `fx.vinyl` for band-limit + Poisson crackle convolved with its 4 ms damped
  resonance; bigger pops use the family's own `_pop` (4 ms ring + 70 Hz thump + 4-sample
  click), which is the same physical model at a larger amplitude.

## Loops

The two loop archetypes are built as two decorrelated channels (a hiss/crackle bed widened
by a game engine needs decorrelated channels), tail-crossfaded 180–200 ms, then **summed
to mono for delivery**. They set `meta={"loop": True}`, so `master` normalises them to
−24 LKFS and true-peak-limits them.

## Transients

For impact-like cues the peak lands early: `vinyl_pop` 16–103 ms (first pop), `scratch`
7–660 ms (stroke-dependent), `tape_stop` 11–509 ms (source attack), `tape_start` 179–493 ms
(the moment the transport catches). The noise beds (`tape_hiss`, loops) and the slow
`wobble`/`backspin` gestures have no meaningful "impact"; their peak position is wherever
the surface noise happens to be largest.

## Finish chain (one-shots)

`hp 32 Hz` → `dc_block 8 Hz` → 20 ms release → 4 ms fade → peak 0.92 → subtract mean.
Master then normalises to −6 dBFS.

## Acceptance (this machine, current `qa.dedup` with perceptual blocks)

| run | result |
|---|---|
| `--count 12` | exit 0, **12/12 pass**, 5 near-dups |
| `--count 60` | exit 0, **60/60 pass**, 1 near-dup, 59/60 unique (target ≥ 27) |

Nothing was listened to; the human does that.

## CORE ISSUES

1. **Stereo renders crash in the master chain.** `core.render.master` calls
   `dsp.fade`, which is mono-only: `y[:m] *= r` raises
   `ValueError: operands could not be broadcast together with shapes (192,2) (192,)`.
   So any `Render(..., stereo=True)`, including every loop, fails inside
   `render_family` before it can be written. Worked around by generating two
   decorrelated channels, summing to mono and returning `stereo=False`. Fix: make
   `dsp.fade` broadcast the ramp over the last axis (`r[:, None]` when `y.ndim == 2`),
   and give `master`/`normalize_lufs` a stereo path. This blocks the pack's stated
   "stereo for beds/loops" policy at the platform level.
2. **`fx.scratch` cannot play backwards.** It computes
   `idx = cumsum(np.abs(rate_curve))`, so a negative rate is treated as positive: a
   backspin or the reverse half of a baby scratch plays forward. This family implements
   its own `_play` with `cumsum(rate_curve)` (signed) and uses it for every scratch
   gesture. Fix: sign-aware integration inside `fx.scratch` (clamp to `[0, n-1]`).
3. **`fx.tape` has fixed transport rates.** `wow` and `flutter` only scale the depth of a
   hardcoded 0.7 Hz and 8.5 Hz sine; there is no way to ask for a 1.6 Hz wobble or a
   slower flutter through the API, even though the contract's parameter ranges
   (wow 0.5–2 Hz) imply it. The `wobble` archetype generates its own rate curve and then
   calls `fx.tape(wow=0)`. Fix: `wow_hz` / `flutter_hz` arguments.
4. **`master` fades loops.** The mandatory 4 ms fade is applied to loop material too, so
   every bed has a 4 ms dip at the seam even though the family crossfades its own loop
   point. Fix: skip the fade when `meta["loop"]` is true.
5. **`qa.audit` dc_offset measures the clip mean.** Pitch-warped sources (tape stop,
   backspin) hold a DC lobe when the transport slows, and the master end-fade shortens the
   window further. The family removes the mean after every fade; a real DC estimate (mean
   after a high-pass) would make the rule meaningful.
