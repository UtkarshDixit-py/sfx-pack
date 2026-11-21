# retro — 8-bit / chiptune game sounds

`families/retro.py` · descriptors: `coin, jump, powerup, hit, gameover, select, blip`
· mono one-shots 0.05–0.78 s (spec 0.05–0.8)

## Archetypes

| archetype | share | recipe | descriptor |
|---|---|---|---|
| `coin` | 13 % | two square notes (B5/A5/G5 → E6/F6/D6), 24–38 % / rest split | `coin` |
| `jump` | 12 % | band-limited sweep 150–260 → 550–950 Hz, optional triangle sub | `jump` |
| `powerup` | 11 % | 5–8 note major-pentatonic run from C5ish, optional fifth layer | `powerup` |
| `1up` | 7 % | same run, own contour (E5 G5 E6 C6 D6 G6 pattern) | `powerup` |
| `hit` | 11 % | sweep 420–950 → 80–190 Hz + white/velvet noise burst + 120→55 Hz sub | `hit` |
| `gameover` | 10 % | six-note descending minor (80 %) or major arpeggio, long last note | `gameover` |
| `select` | 10 % | two short notes, +5/+7/+12 semitones | `select` |
| `laser` | 10 % | exponential sweep down 1200–2600 → 140–340 Hz, saw or square, + click | `blip` |
| `blip` | 9 % | one note 1.1–2.6 kHz, 50–100 ms | `blip` |
| `door` | 7 % | 65–135 Hz square+saw buzz, 18–42 Hz AM, noise grit, two clicks | `select` |

Notes are sequenced one at a time with the oscillator phase reset per note (that is how a
sound chip retriggers a channel), each with a 1.5 ms attack and a 3 ms release. There is
no `dsp.triangle`, so a band-limited triangle is built as
`sine − sine(3f)/9 + sine(5f)/25`; it is used for jump/gameover timbres.

## Parameter spread

- transpose `shift` −2…+2 semitones; coin/jump/select/laser root ±varies per archetype
- duty 0.125 / 0.25 / 0.5 / 0.75; vibrato 0 / 0.4 / 0.8 / 1.4 %
- gameover timbre square or triangle; hit noise white or velvet; jump sub 0–0.4
- **bitcrush**: bits 6–8 (mostly 8), `rate_div` 1–3 (mostly 2), always `mode="harsh"`.
  This is the one family where alias is the product; `qa.alias_db` will be high by design.

## Finish chain

`mix` the lead (+ fifth/sub/click layers, each peak-normalised) → **`fx.bitcrush` (harsh)** →
`dc_block 15 Hz` → `hp 60 Hz` → 3 ms fade → peak 0.95 → subtract mean.
Master then normalises to −6 dBFS. The mean removal is needed because 8-bit quantisation
of an asymmetric duty cycle leaves a real offset.

Transient policy: impact-like cues (`hit`, `door`, `jump`, `laser`, `blip`, `coin`,
`select`) peak in the first 3–20 ms. The jingles (`gameover`, `powerup`, `1up`) are
melodic arpeggios, so the loudest note can land anywhere in the sequence; the first note
still has a crisp 1.5 ms attack and the sequence is the product, not a decorative tail.

## Acceptance (this machine, current `qa.dedup` with perceptual blocks)

| run | result |
|---|---|
| `--count 12` | exit 0, **12/12 pass**, 2 near-dups |
| `--count 60` | exit 0, **60/60 pass**, 6 near-dups, 54/60 unique (target ≥ 27) |

Nothing was listened to; the human does that.

## CORE ISSUES

1. **`qa.spectral_holes` false-positives on harmonic combs.** A duty-0.5 square has *only*
   odd harmonics, so there are real valleys between them (118 → 236 → 354 Hz…). Once the
   `harsh` bitcrush adds broadband alias and lifts `flatness` over 0.05, the hole detector
   starts counting those valleys: one `door` candidate scored 2 holes. Worked around by
   adding a saw layer (all harmonics) and a low-level wideband air bed to the door. The
   detector is right for noise material but should not be applied to a signal that is
   still fundamentally harmonic — its `flatness` guard (0.05) is too permissive for
   bitcrushed tones.
2. **Bitcrush + quantisation leaves DC.** `fx.bitcrush` rounds to 2^(bits−1) steps; on an
   asymmetric duty cycle or a gated noise burst the rounded mean is not zero. The family
   removes the mean after all fades. `fx.bitcrush` itself could subtract its output mean.
3. **The master 4 ms fade is 8 % of the shortest hit.** A 50 ms `blip` loses its first and
   last 4 ms. Acceptable at this length, but the fade is proportionally large; 1 ms for
   sub-100 ms one-shots would be safer.
4. **No triangle oscillator in `core.dsp`.** Built from three `dsp.sine` partials
   (band-limited). Cheap, but a first-class `dsp.triangle` would make chiptune families
   cleaner and would let the triangle share the polyBLEP path.
