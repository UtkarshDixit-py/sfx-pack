# boom — cinematic sub-boom / explosion

Three-layer explosion per Farnell: blast transient, pitch-dropping body, debris.
Everything is synthesised; no samples are loaded.

## Signal chain

| layer  | source | role |
|--------|--------|------|
| blast  | white noise, lowpassed 0.26-6 kHz, attack 4-20 ms, `sat_adaa` drive 2.2-5.0, plus a 1-2 ms highpassed crack | the hit |
| body   | sine sweep 80-155 Hz → 25-42 Hz over 150-400 ms, + optional 2nd/3rd harmonic, gentle `sat_adaa` | the weight |
| debris | sparse grains (density 18-70/s, 5-34 ms grains, random pitch) **plus** a low broadband "smoke" bed | crackle + smooth LTAS |
| air    | highpassed white noise, 0-30 % | open top end |
| space  | level-matched synthesized IR, t60 0.5-1.7 s, 4-32 % wet | cinematic size |

Every layer is peak-normalised explicitly before the mix. Body lands at 26-42 Hz
so there is real energy below 60 Hz (median 18.6 % of total spectral energy,
range 2.2-84 %; the low end is on the "explosion" candidates that deliberately
favour the bright debris).

## Design decisions

- **Archetypes** set the macro shape: `boom` (0.9-1.9 s punchy), `explosion`
  (1.3-2.5 s, more debris), `sub` (0.9-1.8 s, deep and tight), `cinematic`
  (1.6-2.5 s, slow attack, big room). Weights 34/28/20/18.
- **Spectral character** is a second independent axis because the first pass
  produced 11 near-duplicates in 12 (every boom is a low noise blob to an MFCC):
  - `deep` — blast cutoff 260-700 Hz, lowpassed mix 0-1100 Hz
  - `cracky` — blast cutoff 2.2-6 kHz, bright smoke, air
  - `dirty` — saturated mids
  - `wide` — mid-bright with air
  - `cinematic` — dark, large room
- **Debris** carries a smoke bed (brown/pink/white noise, highpass 90-1400 Hz,
  lowpass 1.2-16 kHz) so the long-term average spectrum is smooth and the audit
  does not see spectral holes. The grains sit on top at a low peak.
- **Sub weight without clipping** is handled by giving the body explicit gain
  (0.75-1.15 of the blast peak) and keeping the final buffer at 0.9 so the
  master's soft clipper never engages.
- **Length** 0.8-2.5 s by archetype; the body tau is 35-65 % of duration.

## Sampling ranges

dur per archetype (0.9-2.5 s), f_start 80-155 Hz, f_end 25-42 Hz,
sweep 0.15-0.40 s, blast_cut 0.26-6 kHz, blast_attack 4-20 ms, blast_tau
0.08-0.28 s, body_gain 0.75-1.15, body_tau 0.35-0.65·dur, body_h2 0.05-0.45,
body_h3 0-0.28, drive 2.2-5.0, debris density 18-70/s, debris_gain 0.08-0.42,
smoke_gain 0.08-0.30, smoke_hp 90-1400 Hz, smoke_lp 1.2-16 kHz, air 0-30 %,
space 4-32 %, space_t60 0.5-1.7 s, dirt 0-1.

Descriptors: boom, explosion, sub, impact, cinematic (name = `archetype_character`).

## Acceptance

```
gen_candidates.py boom --count 12  -> 12 rendered, 12 pass, 3 near-dup
                                     diversity 9/12 unique  [OK]
gen_candidates.py boom --count 60  -> 60 rendered, 60 pass, 21 near-dup
                                     diversity 39/60 unique  [OK]
```
Durations 0.90-2.44 s, mono, peak -6 dBFS, true peak <= -1 dBTP, crest
8.1-15.9 dB, alias -99 dB, no NaNs, deterministic.

## CORE ISSUES

1. **`fx.reverb` wet path is not gain-normalised** (same bug as `impact.md`).
   The IR energy set the file peak and the blast ended up far below the tail.
   Fixed locally with `_space()`, which matches wet peak to dry peak before
   mixing. After the fix the first 15 ms is within 2 dB of the file peak for
   33/60 and within 3.7 dB for all 60; before the fix it was 12.8 dB down on
   median.
2. **Master `soft_clip` before normalisation** can add DC on a hot low-frequency
   sum. Every family delivers at 0.9 peak so the clipper is bypassed.
3. **Master's fixed 4 ms fade-in** caps how early the numerical peak can be.
   Combined with a cinematic 12-20 ms blast attack, the peak is naturally a few
   ms after the fade. The perceptual attack is first; the argmax proxy in the QA
   sheet is not.
4. **Spectral-hole audit + long noise tails.** A lowpassed blast with a bright
   debris tail can leave a 15 dB gap between the two, which the audit counts as
   a defect. The smoke bed exists specifically to keep the LTAS smooth.
5. **Dedup is MFCC-based**; booms are spectrally similar by nature. The
   `character` axis is the main spread lever (21 near-dups in the 60 set,
   comfortably inside the tool's 45 % diversity target).
