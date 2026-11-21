# Worker B — glitch, stutter, perc

Read `briefs/CONTRACT.md` first, then `families/laser.py`. Build THREE families.

## glitch
Digital damage. Combine `fx.bitcrush` (both modes), `dsp.granulate`, dropout gates, and buffer-repeat.
- Build a short synthetic SOURCE buffer first (square/saw/noise burst), then destroy it.
- Two characters: 'harsh' (unfiltered sample-hold, aliasing) and 'soft' (pre-LPF).
- Variations: bit reduction 4-10 bits, rate division 2-32, stutter repeats, random dropouts.
- 0.15-1.0 s. Mono. Descriptors: glitch, crush, corrupt, datamosh, stutter.

## stutter
Tempo-locked gated chops for trap/jersey/drill.
- Always derive the rate from a BPM: `rate_hz = bpm/60 * subdivision` (16th at 140 = 9.33 Hz).
- Put a real source under the gate: a synth stab or noise hit, not silence.
- Ramps of 1-2 ms on the gate edges or it clicks.
- 0.3-1.2 s. Mono. Descriptors: stutter, chop, gate, jersey, drill.

## perc
Glitch/electronic percussion one-shots.
- Types: ticks, clicks, rim-ish, woodblock, blips, digital blips, metallic pings.
- Very short (20-180 ms), high transient clarity, minimal low end.
- These are the most-repeated sounds in a game: vary pitch/frequency per seed a lot.
- Descriptors: tick, click, ping, rim, blip.

Run the acceptance test for each family. Write `families/<family>.md` with params, choices, CORE ISSUES.
