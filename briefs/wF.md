# Worker F — sci_fi, retro, tape_vinyl

Read `briefs/CONTRACT.md` first, then `families/laser.py`. Build THREE families.

## sci_fi
Teleports, energy, holograms, doors, machinery, force fields.
- Use FM (inharmonic ratios), moving bandpass sweeps, ring-mod-like textures, granular shimmer.
- Types: teleport in/out, energy charge, force field hum, door open/close, machinery loop (mono),
  hologram shimmer, scanner.
- 0.4-2.5 s. Mono. Descriptors: teleport, energy, field, door, machine, holoscan.

## retro
Chiptune / 8-bit game sounds.
- Strict band-limited square/triangle (use dsp.square/dsp.saw), fast arpeggios, pitch sweeps.
- Use `fx.bitcrush(x, bits=8, rate_div=2, mode='harsh')` at the END for an authentic 8-bit crunch
  (this is the one family where aliasing is the point).
- Types: coin, jump, powerup, hit, gameover, select, laser, blip, door, 1up.
- 0.05-0.8 s. Mono. Descriptors: coin, jump, powerup, hit, gameover, select, blip.

## tape_vinyl
Tape and vinyl artefacts. This is a TEXTURE/CHARACTER family.
- tape: `fx.tape` with varying drive/wow/flutter/hiss; types: tape stop, tape start, tape hiss loop,
  tape degradation, wobble.
- vinyl: `fx.vinyl` crackle + `fx.scratch` with realistic turntable rate curves (baby, chirp,
  transform, backspin). For scratch, synthesise a source buffer first (a stab or noise) then
  drive it with a rate curve.
- Types: vinyl crackle loop (loopable), scratch, backspin, tape stop, tape hiss, vinyl pop.
- 0.3-6 s. Mono EXCEPT loops which may be stereo with meta={"loop": True}.
- Descriptors: crackle, scratch, backspin, tapestop, hiss, vinyl.

Run the acceptance test for each family. Write `families/<family>.md` with params, choices, CORE ISSUES.
