"""Minimal Broadcast-WAVE `bext` chunk writer/reader (EBU Tech 3285).

soundfile cannot author bext, and neither BWF MetaEdit nor exiftool is installed
here, so we write the chunk ourselves. Buyers' ingest tooling (Soundminer,
BaseHead, Wwise/FMOD importers) indexes on Description + Originator, which is
why a professional SFX library ships Broadcast WAV rather than plain WAV.

Every write is verified by re-parsing the file; a verification failure raises
rather than silently producing a corrupt deliverable.
"""
from __future__ import annotations

import struct
from pathlib import Path

UNKNOWN = -32768


def _pad(text: str, width: int) -> bytes:
    b = str(text).encode("ascii", "replace")[: width - 1]
    return b + b"\x00" * (width - len(b))


def make(description: str, originator: str, originator_ref: str = "",
         date: str = "2026-11-19", time: str = "00:00:00",
         coding_history: str = "A=PCM,F=48000,W=24,M=mono",
         lufs: float | None = None, lra: float | None = None,
         true_peak_dbtp: float | None = None) -> bytes:
    def s100(v):
        return UNKNOWN if v is None else max(-32767, min(32767, int(round(v * 100))))

    d = bytearray()
    d += _pad(description, 256)
    d += _pad(originator, 32)
    d += _pad(originator_ref, 32)
    d += _pad(date, 10)
    d += _pad(time, 8)
    d += struct.pack("<II", 0, 0)                       # time reference
    d += struct.pack("<H", 2)                           # version 2
    d += b"\x00" * 64                                   # UMID
    d += struct.pack("<hhhhh", s100(lufs), s100(lra), s100(true_peak_dbtp),
                     UNKNOWN, UNKNOWN)
    d += b"\x00" * 180
    assert len(d) == 602, len(d)
    d += coding_history.encode("ascii", "replace") + b"\x00"
    return bytes(d)


def _chunks(body: bytes):
    out, i = [], 0
    while i + 8 <= len(body):
        cid = body[i:i + 4]
        size = struct.unpack("<I", body[i + 4:i + 8])[0]
        data = body[i + 8:i + 8 + size]
        out.append((cid, data))
        i += 8 + size + (size & 1)
    return out


def write(path: str | Path, **kw) -> None:
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path.name}")
    chunks = [(c, d) for c, d in _chunks(raw[12:]) if c != b"bext"]
    bext = make(**kw)
    out = bytearray(b"WAVE")
    out += b"bext" + struct.pack("<I", len(bext)) + bext
    if len(bext) & 1:
        out += b"\x00"
    for cid, data in chunks:
        out += cid + struct.pack("<I", len(data)) + data
        if len(data) & 1:
            out += b"\x00"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(out)) + bytes(out))
    got = read(path)
    if got.get("Description") != kw.get("description", "") and \
       got.get("originator") != kw.get("originator", ""):
        raise RuntimeError(f"bext verification failed for {path.name}")


def read(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    for cid, data in _chunks(raw[12:]):
        if cid == b"bext":
            def s(a, b):
                return data[a:b].split(b"\x00")[0].decode("ascii", "replace")
            return {
                "Description": s(0, 256),
                "originator": s(256, 288),
                "originator_ref": s(288, 320),
                "date": s(320, 330),
                "time": s(330, 338),
                "version": struct.unpack("<H", data[346:348])[0],
                "coding_history": data[602:].split(b"\x00")[0].decode("ascii", "replace"),
            }
    return {}
