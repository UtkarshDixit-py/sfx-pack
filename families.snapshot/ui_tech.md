# ui_tech — phones, notifications, buttons, toggles, scans, dial tones

`families/ui_tech.py` · descriptors: `tap, toggle, notify, alert, scan, charge, unlock`
· mono, one-shot, 30–300 ms, bright by design (tiny speakers have no bass)

## Types and durations

| type | duration | synthesis | descriptor |
|---|---|---|---|
| tap | 30–60 ms | 1.6–3 kHz sine with a ×0.72–0.95 drop, 4.5 kHz click | `tap` |
| toggle | 60–120 ms | two blips 30–50 ms apart, second at ×1.18–1.36 (on) or ÷ (off) | `toggle` |
| confirm | 120–200 ms | 2–3 note rising arpeggio, roots 740–1318 Hz, maj3/P4/P5 steps | `notify` |
| error | 150–280 ms | two descending square buzzes 420–800 Hz, duty 0.28–0.58 | `alert` |
| notify | 180–300 ms | inharmonic bell, one of 3 partial sets, optional second strike | `notify` |
| message | 100–180 ms | soft two-note "ding-dong", 850–1200 Hz ×0.66–0.80 | `notify` |
| unlock | 130–220 ms | rising sweep 700–1000 Hz × 2.2–3.2 | `unlock` |
| scan | 200–300 ms | 6–9 short blips 24–34 ms apart, 1100–1500 Hz × 1.7–2.2 | `scan` |
| dial | 100–200 ms | DTMF dual tone (697–941 Hz row + 1209–1633 Hz column) + 2nd harmonics | `notify` |
| charge | 180–300 ms | rising sweep 400–700 Hz × 2.8–4.2 + a high shimmer partial | `charge` |

The brief's ten types map onto its seven descriptors; the closest fitting word is used
for each, and the mapping is stated above. `dial` is a signalling tone → `notify`;
`error` → `alert`; `confirm`/`message` → `notify`.

## Design choices

- **Brightness is enforced, not hoped for.** Final chain highpasses at 180–260 Hz, so
  nothing useful is lost on a phone speaker. Measured spectral centroids land at
  0.8–2.5 kHz per type (tap 1.8 k, toggle 2.5 k, scan 1.9 k, message ~0.9 k).
- **Every onset is a transient.** Each note is a separately normalised segment with a
  0.9–4 ms attack, and later notes are never louder than the first (transient first).
- **Bells use inharmonic partial sets** (1 : 2.41 : 3.71 : 5.21, 1 : 2.76 : 5.40 : 8.93,
  1 : 2.05 : 3.71 : 5.89) rather than a stack of octaves, which is what makes a
  notification read as "device" instead of "musical note".
- **Click layer** (1–1.6 ms, highpassed noise) on tap / toggle / error / dial /
  unlock / charge gives the "screen tap" edge that the sine alone lacks.
- Finish: `hp 180–260` → `dc_block 20 Hz` → peak 0.94 → mean removal → 3 ms fade.
  The master's soft-clip never engages, so the click asymmetry cannot shift DC.

## Acceptance

| run | command | result |
|---|---|---|
| 12 | `tools/gen_candidates.py ui_tech --count 12` | exit 0, **12/12 pass**, 1 near-dup |
| 60 | `tools/gen_candidates.py ui_tech --count 60` | exit 0, **60/60 pass**, 27 near-dups → 33 unique survivors |

Nothing was listened to here; the human does that.

## CORE ISSUES

1. **Descriptor vocabulary is narrower than the type list.** `briefs/wE.md` names ten UI
   types but seven descriptors; `dial`, `confirm`, `message` and `error` have no exact
   word. I mapped them to the nearest allowed descriptor and documented it in code
   (`_DESCRIPTOR`). If filenames need one word per type, the descriptor set should be
   extended with `dial`, `confirm`, `message`, `error`.
2. **`qa.dedup` groups variants that are audibly different.** Bell/arpeggio variants
   cluster at 0.95+ cosine on mean+std MFCC even when pitch sets differ; the 27
   near-dups are within-type (confirm/toggle, unlock/charge, error) rather than clones.
   Same root cause as the drum family: the metric has no timing/envelope information.
3. **`dsp.sine(f, fs)` with a scalar `f` yields a 1-sample array** (`_phase` does
   `np.atleast_1d`). It is easy to write `dsp.sine(697.0, fs)` expecting a tone and get
   silence; every call site here passes `np.full(n, f)`. Worth a docstring note or
   scalar broadcast in the primitive.
4. **Very short candidates bypass loudness checks.** `render.integrated_lufs` returns
   `None` under 0.4 s, so `qa.audit` cannot flag a too-quiet 50 ms tap. Fine for this
   family (peak-normalised), but a family that outputs quiet-by-nature short cues would
   never be caught by the automated pass.
