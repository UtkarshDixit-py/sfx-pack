# impact — hard-surface collisions

Modal-synthesis collisions. The identity of an impact is the ring of the
object's normal modes; the contact burst only decides how hard and how bright it
is struck (O'Brien et al. 2002). No noise bursts pretending to be hits.

## Signal chain

| layer   | source | role |
|---------|--------|------|
| click   | highpassed white noise, 0.15 ms attack, ~1 ms tau | contact transient |
| body    | `dsp.modal_bank()` + `dsp.modal()` driven by `dsp.contact_burst()` | the material |
| sub     | 50-92 Hz sine, size-scaled, 1.2 Hz pitch settle | weight on small speakers |
| air     | highpassed noise, decays faster than the body | room/air tail |
| space   | level-matched synthesized IR | optional room (2-22 % wet) |

Every layer is explicitly peak-normalised before the mix
(`_norm(layer, TARGET)`); the mix is then satur(ated), optionally reverberated,
highpassed and peak-normalised to 0.9 before delivery.

## Materials

`dsp.MATERIALS` ratio sets, each with an f0 band and decay band:

| material | f0 @ size 1 | tau0 | descriptor |
|----------|-------------|------|------------|
| metal    | 300-1400 Hz | 0.18-0.50 s | metal |
| glass    | 500-1800 Hz | 0.12-0.40 s | glass |
| wood     | 110-460 Hz  | 0.06-0.18 s | wood |
| concrete | 65-220 Hz   | 0.05-0.16 s | stone |
| ceramic  | 280-1100 Hz | 0.08-0.26 s | stone |
| plastic  | 150-520 Hz  | 0.025-0.09 s | plastic |

Hardness > 0.86 on glass/ceramic/concrete is labelled **crack** instead of the
material word.

## Design decisions

- **Size** 0.55-1.9: `modal_bank` divides every mode frequency by size and
  multiplies every decay by size, so bigger genuinely sounds lower and longer.
  The sub oscillator follows the same trend down to 50 Hz.
- **Hardness** is bimodal (0.18-0.48 or 0.62-1.0) so the set contains both soft
  thuds and sharp cracks rather than a mushy middle.
- **Length** is derived, not sampled: `clip(3.6 * tau0 * size + 0.07, 0.25, 1.2)`
  so the file always contains the ring.
- **Excitation gate.** `contact_burst` smears a soft hit for ~20 ms, which
  pushed the modal peak past the 0-15 ms transient rule. The burst is multiplied
  by a short extra `env_exp` gate (tau 3-11 ms). Hardness still sets the burst
  bandwidth (the research-relevant parameter); only the tail of the smearing is
  trimmed. After this, the first 15 ms is within 2 dB of the file peak for 59/60
  candidates (worst -3.1 dB).
- **Dirt** 0-1 drives `sat_adaa`; medium dirt is a clean bright hit, heavy dirt
  is a harder, tape-ish knock. Space is a small room (t60 0.12-0.55 s).

## Sampling ranges

material (weighted), f0 (per material), size 0.55-1.9, tau0 (per material),
hardness bimodal 0.18-1.0, sub_f clipped 50-92 Hz, sub_gain 0.22-0.55,
air_gain 0.10-0.34, air_hz 3.2-7.2 kHz, bright 3.5-15 kHz, jitter 0-0.012,
dirt 0-1, space 0.02-0.22, space_t60 0.12-0.55 s.

## Acceptance

```
gen_candidates.py impact --count 12  -> 12 rendered, 12 pass, 1 near-dup
                                       diversity 11/12 unique  [OK]
gen_candidates.py impact --count 60  -> 60 rendered, 60 pass, 17 near-dup
                                       diversity 43/60 unique  [OK]
```
Durations 0.25-1.20 s, mono, peak -6 dBFS, alias score -99 dB, no NaNs,
bit-identical on re-render.

## CORE ISSUES

1. **`fx.reverb` / `fx.conv` do not gain-normalise the wet path.**
   `conv()` returns `(1-m)*dry + m*wet` with `wet = fftconvolve(x, ir)` and an
   IR whose *peak* is 1 but whose summed energy is large. The wet tail therefore
   set the file peak and the dry attack ended up 10-13 dB below it. Worked
   around with a local `_space()` that convolves, scales the wet peak to the dry
   peak, then mixes. This is the single biggest perceived-quality bug found.
2. **`render.master()` applies `soft_clip(0.995)` before peak-normalising.**
   A layer sum above 0.995 is nonlinearly compressed and can acquire a DC
   offset (the audit's `dc_offset` rule). Every family here now normalises its
   final buffer to 0.9 so the master's clipper is a no-op.
3. **`render.master()` applies an unconditional 4 ms fade-in.** A transient
   living in the first 4 ms is attenuated toward zero, so the earliest possible
   numerical peak is ~4 ms. This is why the click alone cannot be the loudest
   moment; the modal body has to carry it (see "Excitation gate").
4. **Dedup is MFCC mean+std over a mel filterbank starting at 50 Hz.** Impacts
   are broadband enough that this behaves, but it is blind to pure low-frequency
   pitch; see `sub.md`.
