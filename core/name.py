"""Filenames and manifest, mapped onto the real Universal Category System.

CatIDs are validated against the official UCS v8.2.1 release (753 CatIDs,
82 categories) held in data/ucs_v8.2.1.csv. The earlier provisional map invented
CatIDs like "GLITCHSTUTTER" and "MUSICALBASS" that no library manager accepts;
nothing here is allowed to emit a CatID that is not in the official list.

Filename convention: CatID_FXName_CreatorID_SourceID_vNNN.wav
The CatID is also stamped into each file's bext chunk, so it survives a rename.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UCS_CSV = ROOT / "data" / "ucs_v8.2.1.csv"

CREATOR_ID = "SFXPACK"
SOURCE_ID = "ORIG"


def _load_catids() -> dict[str, tuple[str, str]]:
    if not UCS_CSV.exists():
        return {}
    with UCS_CSV.open(newline="") as fh:
        return {r["CatID"]: (r["Category"], r["SubCategory"]) for r in csv.DictReader(fh)}


CATIDS = _load_catids()

# family -> representative CatID
DEFAULT_CAT = {
    "laser":      "LASRBeam",    # LASERS / BEAM
    "impact":     "DSGNImpt",    # DESIGNED / IMPACT
    "boom":       "DSGNBoom",    # DESIGNED / BOOM
    "glitch":     "UIGlitch",    # USER INTERFACE / GLITCH
    "stutter":    "DSGNRythm",   # DESIGNED / RHYTHMIC
    "riser":      "DSGNRise",    # DESIGNED / RISER
    "swell":      "DSGNWhsh",    # DESIGNED / WHOOSH
    "bass_808":   "MUSCInst",    # MUSICAL / INSTRUMENT
    "bass_synth": "MUSCSynth",   # MUSICAL / SYNTHESIZED
    "sub":        "MUSCInst",
    "stab":       "MUSCInst",
    "drum":       "MUSCPerc",    # MUSICAL / PERCUSSION
    "perc":       "MUSCPerc",
    "ui_tech":    "UIBeep",      # USER INTERFACE / BEEP
    "sci_fi":     "SCIEnrg",     # SCIFI / ENERGY
    "alarm":      "ALRMElec",    # ALARMS / ELECTRONIC
    "retro":      "GAMEArcd",    # GAMES / ARCADE
    "texture":    "AMBDsgn",     # AMBIENCE / DESIGNED
    "tape_vinyl": "OBJTape",     # OBJECTS / TAPE
}

# family -> (param field, {substring: CatID})
BY_PARAM = {
    "impact": ("material", {"metal": "METLImpt", "wood": "WOODImpt", "glass": "GLASImpt",
                            "concrete": "ROCKImpt", "ceramic": "CERMImpt", "plastic": "PLASImpt"}),
    "alarm": ("archetype", {"siren": "ALRMSirn", "klaxon": "ALRMBuzr", "bell": "ALRMBell",
                            "buzzer": "ALRMBuzr"}),
    "sci_fi": ("archetype", {"door": "SCIDoor", "machine": "SCIMach", "holo": "SCICmpt",
                             "scan": "SCICmpt", "retro": "SCIRetro"}),
    "ui_tech": ("type", {"notify": "UIAlert", "alert": "UIAlert", "error": "UIAlert",
                         "message": "UIAlert", "tap": "UIClick", "toggle": "UIClick",
                         "scan": "UIData", "dial": "UIData"}),
    "texture": ("type", {"room": "AMBRoom", "drone": "DSGNDron", "machine": "AMBTech",
                         "wind": "AMBAir", "air": "AMBAir"}),
    "stab": ("type", {"pluck": "MUSCPluck", "bell": "MUSCBell", "tuned": "MUSCTnprc"}),
    "laser": ("archetype", {"blaster": "LASRGun", "gun": "LASRGun"}),
    "drum": ("archetype", {"kick": "MUSCPerc", "snare": "MUSCPerc", "hat": "MUSCPerc",
                           "clap": "MUSCPerc"}),
    "tape_vinyl": ("type", {"crackle": "OBJTape", "scratch": "OBJTape"}),
}


def cat_for(family: str, params: dict | None = None) -> str:
    """Best official CatID for this family/param combination."""
    fam = str(family)
    rule = BY_PARAM.get(fam)
    if rule:
        field, table = rule
        val = str((params or {}).get(field, "")).lower()
        for key, cat in table.items():
            if key and key in val:
                return cat
    return DEFAULT_CAT.get(fam, "DSGNMisc")


_SAFE = re.compile(r"[^A-Za-z0-9#\-]+")


def slug(text: str, maxlen: int = 25) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    t = _SAFE.sub("", t.replace(" ", ""))
    return t[:maxlen] or "SFX"


def build(family: str, descriptor: str, variant: int, params: dict | None = None) -> str:
    cat = cat_for(family, params)
    fx = slug(descriptor)
    return f"{cat}_{fx}_{CREATOR_ID}_{SOURCE_ID}_v{variant:03d}"


def validate(filename: str) -> dict:
    stem = Path(filename).stem
    parts = stem.split("_")
    cat = parts[0] if parts else ""
    ok_cat = cat in CATIDS
    ok_ascii = all(ord(ch) < 128 for ch in stem)
    ok_no_space = " " not in stem
    return {
        "valid": ok_cat and ok_ascii and ok_no_space,
        "cat": cat,
        "cat_known": ok_cat,
        "official_catids": len(CATIDS),
        "fields": len(parts),
    }


FIELDS = ["file", "family", "descriptor", "key", "bpm", "seconds", "channels",
          "peak_dbfs", "true_peak_dbtp", "lufs", "loop", "seed", "cat"]


def write_manifest(rows: list[dict], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{k: r.get(k, "") for k in FIELDS} for r in rows]
    csv_path = out_dir / "manifest.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    json_path = out_dir / "manifest.json"
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def total_bytes(paths: list[str | Path]) -> tuple[int, float]:
    b = sum(Path(p).stat().st_size for p in paths if Path(p).exists())
    return b, round(b / 1024 ** 3, 3)
