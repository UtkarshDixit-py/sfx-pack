# bass_808

Pitched 808 sub-bass one-shots. Every candidate is a single root note synthesised
at its own frequency — **no take is pitch-shifted into the 12 notes**.

## Interface

```python
FAMILY = "bass_808"
DESCRIPTORS = ["808", "glide", "distorted"]
sample_params(rng) -> dict      # includes note, key, octave
generate(params, rng, fs) -> Render   # mono, 0.25-1.8 s
```

`generate` is a pure function of `params` (plus the seed's rng for the click
noise), so re-rendering a stored param dict reproduces the audio exactly.

## 12-note coverage

`sample_params` pulls the note from a module-level round-robin cursor:
`note = NOTE_NAMES[cursor % 12]`. The cursor `c` also drives the spread of the
strongest diversity axes (below). Consequences:

* a `--count 12` run covers `C, C#, …, B` exactly once — this is why the note is
  taken from the call index rather than a blind `rng.choice`, which would almost
  never cover all 12 in 12 draws;
* a `--count 60` run is 12 notes × 5 draws each;
* the note is resolved in `generate` with `dsp.note_hz(note, octave)` (octave 1 =
  C1..B1, 32.7-61.7 Hz) and the fundamental sine is synthesised at that pitch.

## Signal / chain

Pitch-enveloped sine core → `fx.sat_adaa` drive → optional highpassed edge →
optional sub-octave → click transient. Final `hp` at 25 Hz, `dc_block`, delivered
at 0.9 peak (see CORE ISSUES #2).

* pitch envelope: settles from `start_mult` (1.5-2.5x) down to the root; settle
  time is a per-draw attack class — fast 15-35 ms, snap 40-70 ms, slow 80-120 ms;
* glide variant: slides in from a musical interval (−12, −7, −5, +5, +7, +12
  semitones) over 50-180 ms instead of the 2x drop;
* drive 1.5-6.0 (clean / punchy / growl / bright characters); bright adds a
  highpassed copy for phone-speaker presence;
* optional sub-octave layer (42% of draws, gain 0.20-0.50);
* optional 2f / 3f harmonic partials (h2 0-0.40, h3 0-0.25) for extra timbre spread;
* click: 3-9 ms highpassed noise burst, hardness 800-6000 Hz.

## Parameters (per seed)

| axis | range |
|---|---|
| note / octave | round-robin note, octave 1 |
| length | short 0.25-0.55 s, mid 0.65-1.10 s, long 1.20-1.80 s; scaled down for higher notes (low notes ring longer) |
| decay tau | absolute 0.07-1.60 s, decoupled from length |
| decay curve | 0.4-2.6 (exponential shape) |
| pitch settle / depth | 15-120 ms / 1.5-2.5x |
| gesture | plain 50%, glide 25%, distorted 25% |
| drive / character | 1.5-6.0 over clean/punchy/growl/bright |
| click | level 0.10-0.40, HP 800-6000 Hz, 3-9 ms |
| sub-octave | on/off (42%), gain 0.20-0.50 |
| partials | h2 0-0.40, h3 0-0.25 |

## Diversity

core's dedup is a **min-over-blocks** comparison of {MFCC mean, MFCC std, log-mel,
24-pt RMS envelope, timbre scalars}, clustered at 0.93. For sub-bass the spectral
blocks are nearly pitch-invariant, so the spread is built on length bands,
decoupled absolute decay, attack class (pitch settle), drive/character, click
hardness, sub-octave, partials and sub-octave. Two draws of the same note differ
in length **and** attack character, not only loudness.

Measured with the current tool (seed0 = 1000):

* `--count 12`: 12/12 pass audit, 8 unique of 12 (target ≥ 5) — **OK**
* `--count 60`: 60/60 pass audit, 33 unique of 60 (target ≥ 27) — **OK**

## CORE ISSUES

1. **No time-varying resonant lowpass in core.** `moving_bp` is a bandpass and
   would drop a bass fundamental. Not needed by this family; `bass_synth` works
   around it locally with an RBJ biquad. Flagged for whoever needs it.
2. **DC offset from the master soft-clip.** Delivering at peak 1.0 makes
   `render.master`'s `soft_clip(0.995)` engage; on a low-frequency one-shot that
   nonlinearity shifts the mean and `qa.audit` reports `dc_offset` (seen on
   `bass_synth`, ~0.004 absolute). Workaround used here and in the sibling
   families: deliver at **0.9 peak** (below the soft-clip ceiling) plus a final
   `dc_block`. Same root cause documented by worker A on `sub`.
3. **Pitch-blind spectral dedup blocks.** The old mean+std MFCC compared
   different sub-bass notes at ≈0.99 similarity (mel floor 50 Hz, STFT bin 47 Hz).
   The current block dedup mostly fixes this, but its `timbre` block is still
   near-useless for harmonic bass (every pair ≥0.93) — the **envelope** block is
   the effective separator. Not patched: `qa.py` is core. This is why the design
   invests in envelope/attack spread.
