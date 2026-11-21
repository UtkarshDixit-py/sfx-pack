# Worker E — drum, ui_tech, alarm

Read `briefs/CONTRACT.md` first, then `families/laser.py`. Build THREE families.

## drum
One-shot drums for trap / drill / phonk / jersey club / boom bap.
- kick: pitch-enveloped sine 60->35 Hz, 8 ms click, 120-400 ms decay. Optional distortion.
- snare: noise burst + 180 Hz tone body + short pitch drop; 120-350 ms.
- hat: highpassed noise 6-9 kHz, 20-80 ms, optional metallic FM partials for closed hats.
- clap: 3-4 noise bursts 8-12 ms apart, bandpassed 1-2 kHz, then a short tail.
- open hat / ride / 808 hat variants.
- Mono, 0.06-0.8 s. Descriptors: kick, snare, hat, clap, rim.

## ui_tech
Phones, notifications, buttons, toggles, scans, dial tones.
- Must be SHORT (30-300 ms) and bright (phone speakers have no bass).
- Types: tap, toggle, confirm, error, notify, message, unlock, scan, dial, charge.
- Descriptors: tap, toggle, notify, alert, scan, charge, unlock.

## alarm
Sirens, klaxons, alerts, warning tones.
- Two families of design: (a) electronic alarm pulses, (b) siren sweeps.
- Use urgency parameters deliberately: fundamental 400-1500 Hz, pulse rate 1-6 Hz,
  attack shape, harmonic spread. Tier them low/medium/high urgency.
- 0.5-2.5 s. Mono. Descriptors: alarm, siren, klaxon, warning, alert.

Run the acceptance test for each family. Write `families/<family>.md` with params, choices, CORE ISSUES.
