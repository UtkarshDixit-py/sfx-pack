# Worker D — bass_808, bass_synth, stab

Read `briefs/CONTRACT.md` first, then `families/laser.py`. Build THREE families.

## ALL THREE ARE PITCHED: every family must render all 12 notes, and `sample_params`
## must carry a `note` field ("C","C#",... "B") plus `key` for the manifest.
## Render each note at its own root frequency via `dsp.note_hz(note, octave)`.
## NEVER pitch-shift one render into 12 keys.
## Filename descriptors must include the note, e.g. descriptor="808_A".

## bass_808
- sine core with a fast pitch envelope (start 2x the root, fall to root in 20-60 ms).
- drive with `fx.sat_adaa` (2.0-6.0) so it has harmonics on small speakers.
- long-ish tail 0.6-1.8 s, decay scaled with note (low notes ring longer).
- add a short click at onset for definition.
- descriptors: 808_C, 808_D ... plus glide and distorted variants.

## bass_synth
- saw/square (band-limited, use dsp.saw/dsp.square) + lowpass with an env-swept cutoff.
- types: acid (resonant), reese (detuned pair), mono lead bass, FM bass.
- 0.3-1.2 s. Mono. descriptors: bass_C, acid_C, reese_C, fm_C.

## stab
- melodic stabs, plucks and chords. Use `dsp.karplus` (plucked) and `dsp.fm` (bell/metallic)
  and short marimba-like stacks (sine + 2f + 3f, fast decay).
- all 12 notes, 0.2-0.8 s, mono.
- descriptors: stab_C, pluck_C, chord_C (chord = root + minor/major third + fifth, still named by root).

Budget note: 12 notes x several types x 60 candidates adds up fast. Keep stabs and basses
under 1.2 s so the pack stays within 1 GB.

Run the acceptance test for each family. Write `families/<family>.md` with params, choices, CORE ISSUES.
