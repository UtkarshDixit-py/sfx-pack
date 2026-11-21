#!/usr/bin/env python3
"""Render a family's candidate set, audit it, and write the QA queue.

    python3 tools/gen_candidates.py laser --count 60

Output per candidate:
    out/_qa/<family>/<id>.wav      staged, pre-approval
    out/_qa/<family>/<id>.png      spectrogram for the human + for visual review
    qa/queue.json                  what the QA server shows
    out/_qa/<family>/audit.json    every measurement + pass/fail
    out/_qa/<family>/dedup.json    near-duplicate clusters
Nothing is written to master/ here. Approval is a separate, human-gated step.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import qa  # noqa: E402
from core.dsp import rng_for  # noqa: E402
from core.render import render_family, save_wav  # noqa: E402


def load_family(name: str):
    return importlib.import_module(f"families.{name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    mod = load_family(args.family)
    fam = getattr(mod, "FAMILY", args.family)
    out_dir = ROOT / "out" / "_qa" / fam
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, queue, t0 = [], [], time.time()
    for i in range(args.count):
        seed = args.seed0 + i
        cid = f"{fam}_{seed:04d}"
        params = mod.sample_params(rng_for(seed ^ 0x5EED))
        try:
            buf, meta = render_family(mod.generate, params, seed)
        except Exception as exc:                      # a broken candidate must not stop the batch
            print(f"[skip] {cid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        wav = out_dir / f"{cid}.wav"
        save_wav(wav, buf, bits=24)
        png = ""
        if not args.no_png:
            try:
                png = str(qa.spectrogram_png(wav, out_dir / f"{cid}.png").relative_to(ROOT))
            except Exception as exc:
                print(f"[png] {cid}: {exc}", file=sys.stderr)
        a = qa.audit(wav, loop=bool(meta.get("loop", False)))
        row = dict(id=cid, family=fam, params=params, meta=meta,
                   wav=str(wav.relative_to(ROOT)), png=png, audit=a)
        rows.append(row)
        queue.append(dict(
            id=cid, family=fam,
            name=str(meta.get("descriptor", fam)),
            wav=str(wav.relative_to(ROOT)), png=png,
            params={k: v for k, v in params.items()},
            notes=", ".join(a["fails"]) if a["fails"] else "",
        ))

    ok = sum(1 for r in rows if r["audit"]["ok"])
    (out_dir / "audit.json").write_text(json.dumps(rows, indent=2, default=float))

    try:
        d = qa.dedup([dict(id=r["id"], wav=str(ROOT / r["wav"])) for r in rows if r["audit"]["ok"]])
        (out_dir / "dedup.json").write_text(json.dumps(d, indent=2, default=float))
        ndup = len(d["duplicates"])
        nuniq = len(d["survivors"])
    except Exception as exc:
        print(f"[dedup] {type(exc).__name__}: {exc}", file=sys.stderr)
        ndup, nuniq = -1, -1

    qpath = ROOT / "qa" / "queue.json"
    existing = []
    if qpath.exists():
        try:
            existing = [q for q in json.loads(qpath.read_text()) if q.get("family") != fam]
        except Exception:
            existing = []
    qpath.write_text(json.dumps(existing + queue, indent=2))

    dt = time.time() - t0
    target = max(1, int(0.45 * len(rows)))
    print(f"{fam}: {len(rows)} rendered, {ok} pass audit, {ndup} near-duplicates, "
          f"{dt:.1f}s ({dt / max(1, len(rows)):.2f}s each)")
    if nuniq >= 0:
        verdict = "OK" if nuniq >= target else "TOO NARROW - widen the parameter space"
        print(f"diversity: {nuniq} unique of {ok} passing  (target >= {target})  [{verdict}]")
    print(f"queue -> {qpath.relative_to(ROOT)}")
    if ok < len(rows):
        fails = {}
        for r in rows:
            for f in r["audit"]["fails"]:
                fails[f] = fails.get(f, 0) + 1
        print("failures:", fails)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
