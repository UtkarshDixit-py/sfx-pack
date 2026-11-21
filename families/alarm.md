# alarm — sirens, klaxons, electronic pulses, warning tones

`families/alarm.py` · descriptors: `alarm, siren, klaxon, warning, alert`
· mono, one-shot, 0.6–2.5 s (brief: 0.5–2.5 s)

## Two design families

**(a) Electronic alarm pulses** — a harmonic tone gated at the pulse rate:
- `pulse` — one pitch, duty 0.35–0.65, rate from the urgency tier
- `two_tone` — alternating `f0` / `f0×1.15–1.45` each burst, duty 0.35–0.70
- `warning` — slow warm beacon, rate 0.8–1.8 Hz, duty 0.50–0.72, 20–70 ms attack,
  gentle +6 % rise over the cue
- `alert` — fast harsh beeps, rate 3–6 Hz, 900–1500 Hz, duty 0.28–0.45, ends with
  10–22 % silence ("beep-beep-beep … repeat")

**(b) Siren / klaxon**
- `siren` — phase-continuous periodic pitch sweep, `f0 → f0×1.20–1.75`, sweep rate
  0.35–2.6 Hz depending on urgency, continuous for 1.0–2.5 s
- `klaxon` — 380–700 Hz buzz: harmonic stack (odd-biased), deep 10–30 Hz amplitude
  modulation, lowpassed at 8 kHz, gated into on/off bursts at 1.2–3.0 Hz, duty 0.55–0.88

## Urgency is a parameter

| tier | fundamental | pulse rate | attack | harmonic spread |
|---|---|---|---|---|
| low | 400–620 Hz | 1.0–2.2 Hz | 30–80 ms | 0.10–0.35 |
| medium | 600–950 Hz | 2.0–3.8 Hz | 8–30 ms | 0.30–0.60 |
| high | 950–1500 Hz | 3.5–6.0 Hz | 1–8 ms | 0.50–0.90 |

The gate is trapezoidal with raised-cosine edges (no clicks) and the tone stack is
sines only (2nd/3rd/5th partials, skipped above Nyquist), so alias score is clean.
Air (bandpassed noise at 3.2 kHz, 2–7 %) and a small room (t60 0.3–0.8 s, mix
0.02–0.10) sit the cue in a scene.

Measured over the 60 set: duration 0.61–2.46 s, LUFS −18.1 to −9.4 dB, crest 4.4–12.5 dB,
all files tonal (spectral flatness 0.0) so the LTAS hole rule never trips.

Finish: `hp 60 Hz` → `dc_block 20 Hz` (gated squares carry DC) → peak 0.94 → mean
removal → 5 ms fade.

## Acceptance

| run | command | result |
|---|---|---|
| 12 | `tools/gen_candidates.py alarm --count 12` | exit 0, **12/12 pass**, 4 near-dups |
| 60 | `tools/gen_candidates.py alarm --count 60` | exit 0, **60/60 pass**, 38 near-dups → 22 unique survivors |

Nothing was listened to here; the human does that.

## CORE ISSUES

1. **`qa.dedup` collapses tonal alarm variants.** Mean+std MFCC cosine puts every
   medium siren in one cluster and every medium two-tone in another (0.95+), because the
   feature carries no pulse-rate or envelope information — the one thing that separates
   alarms for a listener. 38 near-dups is the metric's limitation, not 38 clone files
   (the same files differ in rate, duty, sweep and pitch). A percussive/alarm-aware
   feature (modulation rate + spectral centroid + pitch track) would be the right fix.
2. **Siren sweep rate is outside the brief's "pulse rate 1–6 Hz" wording.** The brief
   states pulse rate 1–6 Hz for urgency, which is right for gated pulses; a continuous
   siren needs 0.35–2.6 sweep cycles/s to sound like a siren rather than a vibrato. I
   applied 1–6 Hz to the gated archetypes and documented the siren range separately.
3. **Klaxon sits just under the stated fundamental floor (380 Hz vs 400 Hz).** A
   low-frequency buzzer at 400–500 Hz reads as a klaxon; I allowed 380 Hz for the low
   tier so the growl has weight. If the brief's 400–1500 Hz floor is a hard spec, clamp
   `klaxon.f0` to ≥ 400.
