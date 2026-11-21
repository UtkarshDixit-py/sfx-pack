#!/usr/bin/env python3
"""Build the deliverable: cull to 1,000, UCS-name, stamp bext, manifest, zip.

    python3 tools/build_master.py            # dry run, prints the plan
    python3 tools/build_master.py --write     # writes master/ + dist/

Selection: one survivor per dedup cluster (the dedup report already does this),
allocated across families proportionally so the pack keeps its shape.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import bext, name as naming, qa  # noqa: E402
from core.render import peak_dbfs, true_peak_dbtp  # noqa: E402

TARGET = 1000
CREATOR = naming.CREATOR_ID


def survivors(family: str) -> list[dict]:
    d = ROOT / "out" / "_qa" / family
    audit_p, dedup_p = d / "audit.json", d / "dedup.json"
    if not audit_p.exists():
        return []
    rows = {r["id"]: r for r in json.loads(audit_p.read_text()) if r["audit"]["ok"]}
    if dedup_p.exists():
        keep = set(json.loads(dedup_p.read_text())["survivors"])
    else:
        keep = set(rows)
    return [rows[i] for i in keep if i in rows]


def allocate(pools: dict[str, list[dict]]) -> dict[str, int]:
    total = sum(len(v) for v in pools.values())
    if total == 0:
        return {}
    want = {f: TARGET * len(v) / total for f, v in pools.items()}
    out = {f: int(np.floor(w)) for f, w in want.items()}
    rem = TARGET - sum(out.values())
    for f, _ in sorted(want.items(), key=lambda kv: kv[1] - np.floor(kv[1]), reverse=True):
        if rem <= 0:
            break
        if out[f] < len(pools[f]):
            out[f] += 1
            rem -= 1
    return out


def descriptor_of(row: dict) -> str:
    meta = row.get("meta") or {}
    p = row.get("params") or {}
    d = meta.get("descriptor") or meta.get("archetype") or meta.get("type") or ""
    if not d:
        for k in ("archetype", "type", "variant", "material", "note", "kind"):
            if p.get(k):
                d = str(p[k])
                break
    note = p.get("note")
    if note and str(note) not in str(d):
        d = f"{d}{note}"
    return naming.slug(d or "sfx")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()

    fams = sorted(p.parent.name for p in (ROOT / "out" / "_qa").glob("*/audit.json"))
    pools = {f: survivors(f) for f in fams}
    pools = {f: v for f, v in pools.items() if v}
    alloc = allocate(pools)

    master = ROOT / "master"
    rows, written, chosen = [], [], 0
    for fam, n in sorted(alloc.items()):
        for k, row in enumerate(pools[fam][:n]):
            src = ROOT / row["wav"]
            desc = descriptor_of(row)
            cat = naming.cat_for(fam, row.get("params"))
            stem = naming.build(fam, desc, k + 1, row.get("params"))
            dst = master / cat / f"{stem}.wav"
            chosen += 1
            written.append(dst)
            if args.write:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                info = sf.info(str(dst))
                if info.subtype != "PCM_24" or info.samplerate != 48000:
                    raise RuntimeError(f"bad delivery format: {dst.name} {info.subtype} {info.samplerate}")
                y, sr = sf.read(str(dst), always_2d=False, dtype="float64")
                bext.write(
                    dst,
                    description=f"{desc} {fam}",
                    originator=f"{CREATOR} sample library",
                    originator_ref=stem,
                    date=date.today().isoformat(),
                    time=datetime.now().strftime("%H:%M:%S"),
                    coding_history=f"A=PCM,F=48000,W=24,M={'mono' if y.ndim == 1 else 'stereo'}",
                    lufs=row["audit"].get("lufs"),
                    true_peak_dbtp=row["audit"].get("true_peak_dbtp"),
                )
            rows.append(dict(
                file=str(dst.relative_to(ROOT)), family=fam, descriptor=desc,
                key=(row.get("params") or {}).get("note", ""),
                bpm="", seconds=row["audit"]["seconds"],
                channels=row["audit"]["channels"]
                if "channels" in row["audit"] else (2 if row["audit"].get("channels") == 2 else 1),
                peak_dbfs=row["audit"]["peak_dbfs"],
                true_peak_dbtp=row["audit"]["true_peak_dbtp"],
                lufs=row["audit"].get("lufs") or "",
                loop=bool((row.get("meta") or {}).get("loop", False)),
                seed=(row.get("meta") or {}).get("seed", ""),
                cat=cat,
            ))

    print(f"{'family':11s} {'avail':>6} {'take':>5}")
    for f in sorted(alloc):
        print(f"{f:11s} {len(pools[f]):6d} {alloc[f]:5d}")
    print(f"{'TOTAL':11s} {sum(len(v) for v in pools.values()):6d} {chosen:5d}")

    if not args.write:
        print("\ndry run — rerun with --write to build master/ and dist/")
        return 0

    csv_p, json_p = naming.write_manifest(rows, master)
    (master / "LICENSE.txt").write_text(LICENSE)
    sizes = written
    total = sum(p.stat().st_size for p in sizes if p.exists())
    zip_p = ROOT / "dist" / "sfx-pack-v1.zip"
    zip_p.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sizes):
            z.write(p, p.relative_to(master))
        z.write(csv_p, "manifest.csv")
        z.write(json_p, "manifest.json")
        z.write(master / "LICENSE.txt", "LICENSE.txt")
    print(f"\nmanifest -> {csv_p.relative_to(ROOT)}")
    print(f"audio    -> {total / 1024 ** 2:.0f} MB across {len(rows)} files")
    print(f"zip      -> {zip_p.relative_to(ROOT)} ({zip_p.stat().st_size / 1024 ** 2:.0f} MB)")
    return 0


LICENSE = """\
SFX BOX — ROYALTY-FREE SOUND EFFECTS LIBRARY
===========================================

WHAT YOU MAY DO
  Use every sound in this pack in unlimited commercial and non-commercial
  productions: games, apps, film, broadcast, YouTube, podcasts, music releases,
  live performance, installations. No further payment, no attribution required.

WHAT YOU MAY NOT DO
  Redistribute, resell, sublicense or share the sounds as sounds — on their own,
  in a bundle, as a sample pack, or as part of another sound library. You may not
  present them as your own library. One purchaser, unlimited seats within one
  organisation; separate teams or studios need their own licence.

HOW THESE WERE MADE
  Every file is synthesised from source code written for this pack — physical
  modelling, granular and statistical synthesis, and the processing chains
  documented with each family. No recording, no third-party sample, no neural
  generative model was used in the production of any audio in this library.
  There is nothing here to clear.

FORMAT
  24-bit / 48 kHz WAV, mono unless the filename says otherwise. Loops are
  seamless and marked in manifest.csv. Loudness: one-shots peak-normalised to
  about -6 dBFS; beds and loops integrated to roughly -24 LKFS; true peak
  ceiling -1 dBTP.

FILENAMES
  Universal Category System (UCS) convention: CatID_FXName_SFXPACK_ORIG_vNNN.wav
  CatID is also embedded in each file's bext chunk, so it survives a rename.

Version 1.0
"""


if __name__ == "__main__":
    raise SystemExit(main())
