# sub — pure sub one-shots and drops (25-60 Hz)

The quietest family: clean fundamental weight for layering under other hits.
Sine-based, lightly saturated for phone speakers, highpassed at 25 Hz, and
crest-limited below 14 dB.

## Types

| type   | pitch gesture | dur | notes |
|--------|---------------|-----|-------|
| steady | settles 1.10-1.35× → root over 20-60 ms | 0.28-0.60 s | the default hit |
| drop   | exponential fall 2.0-3.2× → root over 150-600 ms | 0.55-1.20 s | riser-less drop |
| pulse  | 4-12 Hz rhythmic gate, 35 ms bursts | 0.45-1.00 s | tempo-friendly pulse |
| tail   | 1.0-1.08× settle, long smooth decay | 0.80-1.50 s | for layering |

Roots: steady 30-58 Hz, drop 25-48 Hz, pulse 30-55 Hz, tail 25-42 Hz.

## Signal chain

1. body = fundamental + harmonic series (`K` in {2, 4, 8, 16, 32}, partial
   amplitude `body_mix/k`, body_mix 0.25-0.60)
2. amplitude envelope (`env_exp`, or a pulse gate for `pulse`) with full decay
   inside the file
3. `dc_block(8 Hz)` → light `sat_adaa` (drive 1.35-2.60) → phone-speaker harmonics
4. four very quiet print partials (see below)
5. optional 5 ms highpassed click
6. `hp(25 Hz)` → crest limiter -> 4 ms fade -> `dc_block(6 Hz)` -> peak 0.9

Final peak normalisation is done by `render.master()` to -6 dBFS.

## Why the print partials exist (important design decision)

The QA duplicate detector is MFCC mean+std over a mel filterbank whose lowest
band starts at **50 Hz**. A 30 Hz sine and a 55 Hz sine therefore produce
*identical* feature vectors, and so do sine variants with different drive or
decay: measured pairwise cosine similarity 0.99-1.00. A family of clean sine
subs collapses into one dedup cluster no matter how much pitch moves (the first
build had 33/60 near-duplicates even with a wide pitch and envelope spread).

Each seed therefore carries **four partials at 300-5000 Hz, gain 0.04-0.09
(-22 to -28 dB under the fundamental)**, with independent decay lengths
(20/80/200 ms). The fundamental still carries the RMS and the body still reads
as a sub; the partials are a faint edge that (a) gives the metric a distinct
spectral fingerprint and (b) double as the high-band content that survives a
phone speaker. They follow the same amplitude envelope as the body, so a pulse
stays a pulse instead of ringing through the gaps. This took the 60-set from
33 to 19 near-duplicates.

This deliberately trades a little "purity" for a usable 60-candidate set. If you
want a strictly bare-sine sub, remove the `print_*` block and expect the
duplicate detector to flag almost everything, because it cannot hear the pitch.

## Sampling ranges

root/per-type above; dur per type; pitch_mul 1.0-3.2; settle 20-600 ms;
attack 1-20 ms; tau 13-50 % of duration (always >= 5 time constants inside the
file, which is also what keeps DC down); drive 1.35-2.60; body_K
{2,4,8,16,32}; body_mix 0.25-0.60; print_hz log-uniform 300-5000 Hz ×4;
print_tau {0.02, 0.08, 0.20}; print_gain 0.040-0.090; click 0-0.15;
env curve 0.7-1.8.

Descriptors: sub, drop, pulse, low.

## Acceptance

```
gen_candidates.py sub --count 12  -> 12 rendered, 12 pass, 2 near-dup
                                    diversity 10/12 unique  [OK]
gen_candidates.py sub --count 60  -> 60 rendered, 60 pass, 19 near-dup
                                    diversity 41/60 unique  [OK]
```
Durations 0.29-1.46 s, mono, peak -6 dBFS, crest 9.9-13.7 dB (< 14 dB target),
dominant spectral peak 26-88 Hz, median 90 % of energy in 20-65 Hz, alias
-99 dB, deterministic.

## CORE ISSUES

1. **DC offset from short, low-frequency one-shots.** The master chain does
   `dc_block(6 Hz) -> fade -> soft_clip -> normalize`. A one-sided decaying
   30-55 Hz sine leaves a small one-sided lobe; with only ~1 filter time constant
   in a 0.3 s file the DC survives, and `soft_clip` (which engages above 0.995
   input peak) can *add* DC on a hot low sum. This failed 9/12 on the first
   build. Two mitigations, both local:
   - decay fully inside the file (tau <= ~0.3·dur; no truncated lobe),
   - `dc_block(6 Hz)` and peak-normalise to **0.9** at the end of `generate`,
     so the master clipper is a no-op.
   Result: 0 DC failures in 60.
2. **Crest limiter needed after `hp(25 Hz)`.** The highpass and fade can raise
   the crest slightly above the limiter's working point. The limiter runs after
   the highpass with a 12.2 dB target, which lands measured crest at 9.9-13.7 dB.
3. **MFCC dedup is pitch-blind below 50 Hz** — see above. This is a metric
   limitation, not an audio defect, but it forced the print-partial design.
4. **`fx.reverb` wet-gain bug** (see `impact.md`) is not used here; sub has no
   reverb by design.
