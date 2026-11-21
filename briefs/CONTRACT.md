# Worker contract — sfx-pack

You are one of six parallel workers building a commercial game SFX pack.
Repo root: /Users/utkarshdixit/sfx-pack.  Python: `.venv/bin/python` (always this one).

## HARD RULES
1. You may create/edit ONLY: `families/<your-family>.py` and `families/<your-family>.md`.
2. NEVER edit `core/*`, `tools/*`, other workers' families, or `qa/`.
   If a core function is broken or missing, DO NOT patch it — write the problem in
   your `<family>.md` under "CORE ISSUES" and work around it.
3. No network, no sample loading, no external audio files. Everything must be
   synthesised from the primitives below. The pack must be clearance-free.
4. Do not ask questions. Make the best engineering decision, write it down in the .md.
5. Run the acceptance test. Iterate until it passes. Then stop.

## FAMILY INTERFACE (copy the structure of families/laser.py — read it first)
```python
FAMILY = "impact"                 # module name
DESCRIPTORS = ["metal", "wood"]   # short UCS-safe words
def sample_params(rng) -> dict:   # design the spread, don't sample blindly
def generate(params, rng, fs=FS) -> Render
```
`Render(buf=np.ndarray, fs=fs, stereo=False, meta=dict(loop=False, descriptor="..."))`
Mono for one-shots. Stereo ONLY for beds/loops (`meta={"loop": True}`).
Layering: `from core.render import Layer, Render, mix` then `mix([...], fs=fs, stereo=...)`.

## CORE API (import as `from core import dsp, fx`)
dsp: FS, rng_for(seed), const, sweep(n,f0,f1,'exp'), saw, square(duty=), sine, fm(fc,ratio,index_env),
     karplus(freq,n,decay,bright,rng=), white, pink, brown, velvet, env_exp, env_adsr, dc_block, fade,
     lp, hp, bp(x,f,q=), moving_bp(x,f_curve,q=), MATERIALS, modal_bank(f0,material,tau0,size,rng=),
     modal(excitation,modes), contact_burst(n,fs,hardness,rng), grain, granulate, note_hz, midi_to_hz, NOTE_NAMES
fx: sat_adaa(x,drive), sat_os, soft_clip, bitcrush(x,bits,rate_div,mode='soft'|'harsh'),
    stutter(x,rate_hz,duty), tape(x,drive,wow,flutter,hiss_db,rng=), vinyl(x,crackle_hz,rng=),
    scratch(src,rate_curve), synth_ir(t60,size,rng=), conv, reverb(x,t60,size,mix,rng=), chain_clean, chain_dirty
render: Layer(buf,gain,offset_s,pan), Render, mix, render_family, master, save_wav, normalize_peak

## QUALITY RULES (learned the hard way — these are non-negotiable)
- MODAL RESONATORS ARE NOW NORMALISED by (1-r). Their output peak is ~1.0, not ~300.
  Never assume modal output is hot; normalise every layer explicitly to a peak target.
- Give EVERY layer an explicit target: `y_layer = layer / max(|layer|) * TARGET_PEAK`.
  Emergent mixes are why amateur packs sound thin.
- `moving_bp` is band-split and seam-free. Use it for anything whose centre frequency moves.
- Always end with `dsp.fade(x, fs, ms=3..5)` and a `dsp.hp(x, 25..120)` if the sound is one-shot.
- Transient first: the loudest moment must be within 0-15 ms. Sustained noise layers must sit
  UNDER the impact, not over it. A rolling/hissing layer at the same peak as the hit is a bug.
- Mono one-shots. Stereo only for texture/ambience beds.
- Vary at least one parameter per seed so 60 candidates are not 60 clones.

## ACCEPTANCE TEST (must pass before you stop)
```
.venv/bin/python tools/gen_candidates.py <family> --count 12
```
Pass = command exits 0, >=70% of candidates pass the audit, `dedup.json` reports few
near-duplicates, and you have listened to nothing yourself — the human does that.
Then render the real set:
```
.venv/bin/python tools/gen_candidates.py <family> --count 60
```

## RESEARCH TO USE (implement these, do not invent your own physics)
- impacts/collisions: modal synthesis, 2-pole resonator bank per mode; drive the bank with a
  contact-acceleration burst whose hardness sets bandwidth (O'Brien et al. 2002).
- booms/explosions: 3 layers — blast transient, pitch-dropping sub 120->30 Hz, debris crackle.
- drums: kick = pitch-enveloped sine (60->35 Hz, 40 ms decay); snare = noise + 180 Hz tone body;
  hat = highpassed noise with 20-40 ms decay + metallic FM partials; clap = 3-4 noise bursts 8-12 ms apart.
- risers/whooshes: bandpassed noise, moving centre, amplitude ~ velocity^k; 2-6 kHz air band.
- textures: per-band random amplitude modulation on pink noise; make it loopable (crossfade ends).
- tape: sat_adaa -> wow 0.5-2 Hz +-0.2% -> gap-loss LP at 12 kHz -> hiss -58 dBFS.
- vinyl: bandlimit 9 kHz -> Poisson crackle 0.5-20/s convolved with a 4 ms damped resonance.
- 808s/bass: sine with a fast pitch envelope (x2 -> root over 20-60 ms), then sat_adaa for drive,
  hp at 25 Hz. Render each of the 12 notes at its own root frequency (never pitch-shift one render).
- stutter: gate at a musically sensible rate (140 BPM -> 16ths = 9.33 Hz) with 1-2 ms ramps.
- glitch: bitcrush in both 'harsh' (unfiltered sample-hold) and 'soft' (pre-LPF) modes, plus grain scatter.

Read `briefs/RESEARCH.md` for the full citation list if you want the parameter ranges.
