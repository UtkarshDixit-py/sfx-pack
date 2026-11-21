# Worker C — riser, swell, texture

Read `briefs/CONTRACT.md` first, then `families/laser.py`. Build THREE families.

## riser
Build/tension rises.
- bandpassed noise with a moving centre (`dsp.moving_bp`) from low to high, amplitude ~ t^k (k=2..4).
- Add a tonal layer that locks pitch near the top so it feels like it's going somewhere.
- Optionally a rising pulse/ticking layer at the end.
- 1.0-4.0 s. Mono. Descriptors: riser, build, sweep, tension, uplift.

## swell
Reverse swells, whooshes, transitions.
- Reverse-envelope noise/synth: rise then hard stop, or rise into a bright transient.
- Also: whoosh passes (moving bandpass across the stereo field is NOT allowed - these are mono).
- 0.4-2.5 s. Mono. Descriptors: swell, whoosh, reverse, transition, air.

## texture
Atmospheric beds and drones. THIS FAMILY IS STEREO AND LOOPABLE.
- Use `mix([...], stereo=True)` and `Render(..., stereo=True, meta={"loop": True})`.
- Per-band amplitude modulation on pink noise (McDermott-style texture statistics).
- Types: room tone, wind, machinery hum, dark drone, airy pad, tension bed.
- Make loops SEAMLESS: generate longer than needed and crossfade the last 150 ms into the start.
- 4-12 s each. Stereo. Descriptors: texture, atmos, drone, bed, room, air.
- Length matters for the 1 GB budget: no textures over 12 s.

Run the acceptance test for each family. Write `families/<family>.md` with params, choices, CORE ISSUES.
