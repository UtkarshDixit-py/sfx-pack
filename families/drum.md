# drum — one-shot kit for trap / drill / phonk / jersey club / boom bap

`families/drum.py` · descriptors: `kick, snare, hat, clap, rim` · mono, one-shot, 0.06–0.8 s

## What it is

Eight archetypes, each with named character variants so the 60-candidate set is a kit,
not 60 recolours:

| archetype | characters | shares |
|---|---|---|
| `kick` | sub, punch, knock, driven, thud | 18 % |
| `kick808` | same five, long body | 8 % |
| `snare` | tight, roomy, crack, fat | 21 % |
| `hat_closed` | noise, metal, band, dark | 17 % |
| `hat_open` | noise, metal, band, dark | 9 % |
| `clap` | tight, wide, snap | 12 % |
| `rim` | wood, plastic | 8 % |
| `ride` | noise, metal, band, dark | 7 % |

## Synthesis (per the contract recipes)

- **kick** — phase-continuous instantaneous pitch `f1 + (f0-f1)·exp(-t/τp)`, root
  `f1` 34–54 Hz, start `f0 = f1×1.40–2.10`, drop time 12–45 ms; body sine + octave
  harmonic; 3–8 ms highpassed noise click.
- **808** — root 30–46 Hz, `f0 = f1×1.6–2.4`, 20–60 ms drop, τ 0.20–0.55 s,
  `sat_adaa` drive 1.0–6.0 → `dsp.hp 30 Hz` (contract's 808 recipe).
- **kick characters** — `sub` clean, `punch` click-forward, `knock` adds a wood modal
  bank at 160–430 Hz driven by `contact_burst`, `driven` heavy ADAA, `thud` a
  bandpassed noise body at 380–950 Hz. The noise-body kick is what makes the set
  distinguishable perceptually (and is a real boom-bap/trap "knock" kick).
- **snare** — 145–220 Hz tone with a ×0.74–0.92 pitch drop + 1.83× partial, noise
  band 1.4–4.5 kHz (white for tight/crack, pink for roomy/fat), 4.8 kHz snap, 30–200 ms
  noise τ, 0.02–0.35 room. `crack` adds a 500–900 Hz knock layer.
- **hat** — highpassed white noise 4.7–10 kHz (or a bandpassed 7.5–11.5 kHz variant),
  14–450 ms τ, optional inharmonic 5-partial sine stack at 4.6–8 kHz. Closed 0.06–0.13 s,
  open 0.20–0.42 s, ride 0.50–0.80 s with a 2–7 Hz shimmer LFO.
- **clap** — 3–4 bandpassed noise bursts (900–2800 Hz depending on character) spaced
  8–12.5 ms (θ = 4.5 ms), plus a decaying tail (40–110 ms), 4.2 kHz snap layer, small room.
- **rim** — `modal_bank` on wood/plastic at 380–1400 Hz, τ0 8–45 ms, driven by
  `contact_burst(hardness 0.82)`, plus a 2.4 kHz stick tick and click.

## Level structure

Every layer is peak-normalised to an explicit target (`sub` body 0.88, harm ≤ 0.40,
click 0.04–0.50, knock ≤ 0.50, thud ≤ 0.60) before `mix`. Transients start at sample 0;
decaying layers sit under the click.

Finish chain: `hp 30 Hz` → `dc_block 10 Hz` → peak 0.94 → subtract mean → 4 ms fade.
Master then normalises to −6 dBFS.

## Acceptance (run on this machine)

| run | command | result |
|---|---|---|
| 12 | `tools/gen_candidates.py drum --count 12` | exit 0, **12/12 pass**, 3 near-dups |
| 60 | `tools/gen_candidates.py drum --count 60` | exit 0, **60/60 pass**, 41 near-dups → 19 unique survivors |

Nothing was listened to (per the contract); the human does that.

## CORE ISSUES

1. **Master soft-clip + DC blocker fight low one-shots.** `render.master` applies
   `dc_block(fc=6 Hz)` before `soft_clip`. On a 35–50 Hz kick the blocker removes the
   *local* mean of the decaying sine, which overshoots the peak by ~6 %, so the signal
   crosses the 0.995 soft-clip knee and `tanh` shifts the mean of the asymmetric
   (bright click + low sine) waveform past `qa.DEFAULTS["max_dc"] = 0.001`. Worked
   around inside the family (`dc_block 10 Hz` + peak 0.94 + explicit mean removal), but
   any family with LF one-shots will hit this. A smarter master could highpass *after*
   soft-clip, or `qa.audit` could ignore windowed mean on sub-0.4 s LF material.
2. **The mandatory 4 ms fade eats the attack.** `master()` fades the first 4 ms of every
   one-shot, exactly where a drum's click lives (a 0–1 ramp halves the click energy).
   Worked around with 3–8 ms clicks so the energy survives past the ramp, but the
   transient is still ~6 dB softer than the raw layer peak. Consider a fade that starts
   after the first zero-crossing, or 1 ms on short one-shots.
3. **`qa.dedup` cannot separate same-type percussion.** The feature is mean+std MFCC
   cosine at 0.95. Every kick timbre (sub vs driven vs 808) scores 0.97–0.996 against
   every other kick; hats score 0.89–0.996 even across hp 5 vs 9 kHz. So "41
   near-duplicates" is mostly the metric, not clone audio (kick-vs-snare similarity is
   0.06, and the character variants audibly differ). A percussive packet would do
   better with envelope/decay + spectral-centroid + pitch features, or dedup *within*
   archetype instead of across the whole family.
4. **`dsp.sine(f, fs)` with a scalar `f` returns one sample.** `_phase()` calls
   `np.atleast_1d(f)`, so `dsp.sine(697.0, fs)` is a length-1 array, not a thinned
   oscillator. Easy silent footgun in partial stacks; always pass `np.full(n, f)`.
   Worth documenting in the primitive's docstring (or supporting scalar broadcast).
