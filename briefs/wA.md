# Worker A — impact, boom, sub

Read `briefs/CONTRACT.md` first, then `families/laser.py`. Build THREE families.

## impact
Hard surface collisions. Modal resonator bank, not noise bursts.
Materials (use dsp.MATERIALS or your own ratio sets): metal, wood, glass, concrete, ceramic, plastic.
- Drive the bank with `dsp.contact_burst(...)`; hardness 0.2 (soft thud) to 1.0 (sharp crack).
- Layers: click transient (0-3 ms) + modal body + sub 60-90 Hz + air tail.
- Size scales the fundamental DOWN and the decay LONGER (bigger object = lower, longer).
- Target length 0.25-1.2 s. Mono. Descriptors: metal, wood, glass, stone, plastic, crack.
- Vary: material, size, hardness, decay, brightness. 60 candidates must not be 60 clones.

## boom
Cinematic sub-boom / explosion, 3 layers (Farnell):
1. blast: saturated lowpassed noise burst, 5-20 ms attack
2. body: sine/noise sweep 120 Hz -> 30 Hz over 150-400 ms
3. debris: sparse filtered noise grains for 0.3-1.5 s
- Target 0.8-2.5 s. Mono. Must have real sub weight (energy below 60 Hz) without clipping.
- Descriptors: boom, explosion, sub, impact, cinematic.

## sub
Pure sub one-shots and drops. Sine-based, 25-60 Hz.
- Types: steady sub hit, sub drop (pitch falls), sub pulse, sub tail.
- Use sat_adaa lightly for harmonics so it survives phone speakers, then hp at 25 Hz.
- Descriptors: sub, drop, pulse, low.
- IMPORTANT: these are the quietest family. Peak-normalise but keep crest factor sane (< 14 dB).

Run the acceptance test for each family. Write `families/<family>.md` with params, choices, CORE ISSUES.
