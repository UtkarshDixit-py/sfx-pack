#!/usr/bin/env python3
"""qa_server.py -- single-file QA approval server for the sfx-pack sound pipeline.

    python3 qa_server.py --port 7788

Reads  <root>/qa/queue.json       (the pipeline writes it)
Writes <root>/qa/decisions.jsonl  (append-only; the last write per id wins)

GET  /            single page app        GET  /state      totals + cursor
GET  /items       queue items            GET  /decisions  history + latest per id
GET  /wav/<id>    WAV, Range supported   GET|POST /reload re-read queue + decisions
GET  /png/<id>    spectrogram PNG        POST /decide     {"id","decision"}
GET  /meta/<id>   duration / sample rate / peak dBFS, read from the WAV itself

Standard library only. One trusted human on the LAN, so no auth.
"""
from __future__ import annotations

import argparse, json, math, os, re, socket, sys, threading, wave
from array import array
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

DECISIONS = ("approve", "reject", "flag")


def log(msg):
    """Diagnostics to stderr; stdout carries only the startup banner."""
    print(f"[qa] {msg}", file=sys.stderr, flush=True)


def now_iso():
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return ts.replace("+00:00", "Z")


def peak_dbfs(raw, width):
    """Peak sample level in dBFS for raw PCM frames, or None if unsupported."""
    if not raw:
        return None
    try:
        if width == 1:  # 8-bit WAV is unsigned, centred on 128
            peak, full = max(abs(b - 128) for b in raw), 127.0
        elif width == 2:
            a = array("h"); a.frombytes(raw[: len(raw) // 2 * 2])
            if not len(a): return None
            if sys.byteorder == "big": a.byteswap()
            peak, full = max(max(a), -min(a)), 32768.0
        elif width == 3:  # no array typecode for 24-bit; walk the bytes
            peak, mv = 0, memoryview(raw)
            for i in range(0, len(raw) - 2, 3):
                v = int.from_bytes(mv[i:i + 3], "little", signed=True)
                peak = max(peak, -v, v)
            full = 8388608.0
        elif width == 4:
            a = array("i"); a.frombytes(raw[: len(raw) // 4 * 4])
            if a.itemsize != 4 or not len(a): return None
            if sys.byteorder == "big": a.byteswap()
            peak, full = max(max(a), -min(a)), 2147483648.0
        else:
            return None
    except Exception:
        return None
    return -999.0 if peak <= 0 else round(20 * math.log10(peak / full), 2)


def read_wav_meta(path):
    """Header facts + peak level. Peak covers the first 10 s only, which keeps
    very long files cheap and is plenty for a QA glance."""
    try:
        with wave.open(str(path), "rb") as w:
            nframes, rate = w.getnframes(), w.getframerate()
            width, chans = w.getsampwidth(), w.getnchannels()
            raw = w.readframes(min(nframes, rate * 10)) if nframes and rate else b""
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "sample_rate": rate, "channels": chans, "bits": width * 8,
            "duration_s": round(nframes / rate, 3) if rate else None,
            "peak_dbfs": peak_dbfs(raw, width)}


class Store:
    """Queue + decisions under one lock. Re-reading from disk is cheap."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.qdir = root / "qa"
        self.queue_path = self.qdir / "queue.json"
        self.dec_path = self.qdir / "decisions.jsonl"
        self.items, self.by_id = [], {}
        self.exists, self.error = False, None
        self.history, self.latest = [], {}
        self.meta_cache = {}

    def resolve(self, rel):
        """Absolute path for a queue entry, refusing anything outside the root."""
        if not isinstance(rel, str) or not rel.strip():
            return None
        try:
            p = (self.root / rel).resolve()
            p.relative_to(self.root)
        except (OSError, RuntimeError):
            return None
        except ValueError:
            log(f"refusing path outside root: {rel!r}")
            return None
        return p

    def _normalise(self, entry):
        try:
            if not isinstance(entry, dict): raise ValueError("not an object")
            iid = entry.get("id")
            if not isinstance(iid, str) or not iid.strip(): raise ValueError("missing/invalid id")
            wav = self.resolve(entry.get("wav"))
            if wav is None: raise ValueError("missing/invalid wav path")
        except Exception as exc:
            tag = entry.get("id") if isinstance(entry, dict) else repr(entry)[:60]
            log(f"skipping malformed queue item {tag!r}: {exc}")
            return None
        png = self.resolve(entry["png"]) if entry.get("png") else None
        params = entry.get("params")
        return {"id": iid, "family": str(entry.get("family") or ""),
                "name": str(entry.get("name") or iid),
                "params": params if isinstance(params, dict) else {},
                "notes": str(entry.get("notes") or ""),
                "wav": wav, "png": png, "wav_rel": entry.get("wav", ""),
                "has_png": bool(png and png.is_file()), "has_wav": wav.is_file()}

    def load_queue(self):
        exists, err, raw, bad = self.queue_path.is_file(), None, None, 0
        if exists:
            try:
                raw = json.loads(self.queue_path.read_text(encoding="utf-8"))
                if not isinstance(raw, list): raise ValueError("queue.json must be a JSON array")
            except Exception as exc:
                exists, err = False, f"queue.json unreadable: {exc}"
                log(err)
        items = []
        for entry in raw or []:
            it = self._normalise(entry)
            if it is None:
                bad += 1
            elif any(x["id"] == it["id"] for x in items):
                log(f"duplicate id {it['id']!r} in queue.json - keeping the first")
            else:
                items.append(it)
        if bad:
            log(f"skipped {bad} malformed queue item(s)")
        with self.lock:
            self.items, self.by_id = items, {it["id"]: it for it in items}
            self.exists, self.error = exists, err
        return len(items)

    def load_decisions(self):
        hist, latest = [], {}
        if self.dec_path.is_file():
            text = self.dec_path.read_text(encoding="utf-8", errors="replace")
            for ln, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    iid, dec = rec.get("id"), rec.get("decision")
                    if not isinstance(iid, str) or not iid or dec not in DECISIONS:
                        raise ValueError("bad record")
                except Exception as exc:
                    log(f"decisions.jsonl line {ln}: {exc} - skipped")
                    continue
                hist.append(rec)
                latest[iid] = dec  # last write wins
        with self.lock:
            self.history, self.latest = hist, latest
        return len(hist)

    def append_decision(self, iid, decision):
        rec = {"id": iid, "decision": decision, "ts": now_iso()}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self.lock:
            self.qdir.mkdir(parents=True, exist_ok=True)
            with open(self.dec_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())  # decisions gate delivery; don't lose them
            self.history.append(rec)
            self.latest[iid] = decision
        return rec

    def state(self):
        with self.lock:
            items, latest = self.items, dict(self.latest)
        n = {"approve": 0, "reject": 0, "flag": 0}
        cursor = None
        for i, it in enumerate(items):
            d = latest.get(it["id"])
            if d in n: n[d] += 1
            elif cursor is None: cursor = i
        return {"total": len(items), "approved": n["approve"], "rejected": n["reject"],
                "flagged": n["flag"], "decided": sum(n.values()),
                "cursor": len(items) if cursor is None else cursor,
                "exists": self.exists, "error": self.error}

    def meta_for(self, item):
        path = item["wav"]
        try:
            st = path.stat()
        except OSError:
            return {"ok": False, "error": "wav missing"}
        stamp, key = (st.st_mtime_ns, st.st_size), str(path)
        with self.lock:
            hit = self.meta_cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
        meta = read_wav_meta(path)
        with self.lock:
            self.meta_cache[key] = (stamp, meta)
        return meta


# ------------------------------------------------------------------------- page
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>SFX QA</title>
<style>
:root{--bg:#0b0d10;--panel:#141922;--fg:#e8ecf3;--dim:#8a94a6;--line:#222a35;--ap:#22c55e;--rj:#ef4444;--fl:#f59e0b}
*{box-sizing:border-box}html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--fg);min-height:100vh;min-height:100dvh;display:flex;flex-direction:column;font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-text-size-adjust:100%}
header{padding:10px 14px 8px;border-bottom:1px solid var(--line);background:var(--bg);position:sticky;top:0;z-index:9}
#bar{display:flex;height:10px;background:#1b2129;border-radius:6px;overflow:hidden}
#bar>div{height:100%;width:0;transition:width .12s linear}
#seg-ap{background:var(--ap)}#seg-rj{background:var(--rj)}#seg-fl{background:var(--fl)}
#stats{margin-top:6px;font-size:13px;color:var(--dim)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}main{flex:1;display:flex;justify-content:center;padding:14px}#wrap{width:100%;max-width:920px}
#empty{text-align:center;color:var(--dim);padding:22vh 20px;font-size:16px}#empty-msg{display:block;margin-bottom:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--fl);animation:pulse 1.4s infinite}@keyframes pulse{0%,100%{opacity:.25}50%{opacity:1}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
.head{display:flex;align-items:flex-start;gap:10px;justify-content:space-between;margin-bottom:10px}
.title{font-size:20px;font-weight:650;letter-spacing:.2px}.sub{color:var(--dim);font-size:12px}
.badge{font-size:11px;font-weight:700;letter-spacing:.8px;padding:4px 9px;border-radius:999px;border:1px solid var(--line);color:var(--dim);white-space:nowrap}
.badge.approve{background:rgba(34,197,94,.14);color:var(--ap);border-color:rgba(34,197,94,.4)}
.badge.reject{background:rgba(239,68,68,.14);color:var(--rj);border-color:rgba(239,68,68,.4)}
.badge.flag{background:rgba(245,158,11,.14);color:var(--fl);border-color:rgba(245,158,11,.4)}
.spectro{background:#070a0d;border:1px solid var(--line);border-radius:10px;display:flex;align-items:center;justify-content:center;min-height:180px;overflow:hidden}
.spectro img{width:100%;max-height:56vh;object-fit:contain;display:block}
#nospec{color:var(--dim);font-size:13px;padding:70px 0}
audio{width:100%;margin:12px 0 8px}.meta{font-size:13px;word-break:break-word}
.params{font-size:12px;color:var(--dim);margin-top:4px;word-break:break-all}
.notes{font-size:13px;color:var(--fl);margin-top:6px;white-space:pre-wrap}
.actions{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
.nav{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px}
button{font:inherit;color:var(--fg);background:#1c232d;border:1px solid var(--line);border-radius:10px;padding:14px 8px;min-height:56px;cursor:pointer;touch-action:manipulation;-webkit-tap-highlight-color:transparent}
button:active{transform:translateY(1px)}
.btn.approve{background:rgba(34,197,94,.16);border-color:rgba(34,197,94,.45);color:#b9f6cf;font-weight:650}
.btn.reject{background:rgba(239,68,68,.16);border-color:rgba(239,68,68,.45);color:#ffc9c9;font-weight:650}
.btn.flag{background:rgba(245,158,11,.16);border-color:rgba(245,158,11,.45);color:#ffe0a8;font-weight:650}
.nav button{min-height:48px;font-size:13px;color:var(--dim)}
kbd{font:inherit;font-size:11px;opacity:.6;border:1px solid currentColor;border-radius:4px;padding:0 4px;margin-left:5px}
footer{border-top:1px solid var(--line);padding:9px 14px;font-size:12px;color:var(--dim);text-align:center}
.hidden{display:none!important}
#toast{position:fixed;left:50%;bottom:66px;transform:translateX(-50%);background:#1f2937;color:var(--fg);border:1px solid var(--line);padding:10px 16px;border-radius:10px;font-size:13px;z-index:20;max-width:90vw}
@media (max-width:520px){.title{font-size:17px}.actions{gap:8px}button{padding:12px 6px}}
</style>
</head>
<body>
<header>
  <div id="bar"><div id="seg-ap"></div><div id="seg-rj"></div><div id="seg-fl"></div></div>
  <div id="stats" class="mono">loading&hellip;</div>
</header>
<main><div id="wrap">
  <div id="empty"><span id="empty-msg">waiting for qa/queue.json&hellip;</span><span class="dot"></span></div>
  <section id="card" class="card hidden">
    <div class="head">
      <div><div class="title" id="itemname"></div><div class="sub mono" id="itemid"></div></div>
      <div class="badge" id="badge">UNDECIDED</div>
    </div>
    <div class="spectro"><img id="spec" alt="spectrogram" class="hidden"><div id="nospec">no spectrogram</div></div>
    <audio id="audio" controls autoplay preload="auto"></audio>
    <div class="meta mono" id="metaline"></div>
    <div class="params mono" id="params"></div>
    <div class="notes hidden" id="notes"></div>
    <div class="actions">
      <button class="btn approve" data-act="approve">Approve<kbd>A</kbd></button>
      <button class="btn reject" data-act="reject">Reject<kbd>X</kbd></button>
      <button class="btn flag" data-act="flag">Flag<kbd>F</kbd></button>
    </div>
    <div class="nav">
      <button id="b-prev">&larr; Prev</button><button id="b-replay">Replay<kbd>&#9251;</kbd></button>
      <button id="b-next">Next &rarr;</button><button id="b-reload">Reload<kbd>R</kbd></button>
    </div>
  </section>
</div></main>
<footer class="mono">A approve &middot; X reject &middot; F flag &middot; space replay &middot; &larr; &rarr; move &middot; R reload</footer>
<div id="toast" class="hidden"></div>
<script>
"use strict";
const $ = s => document.querySelector(s), audio = $("#audio");
const S = { items: [], idx: 0, exists: true, latest: {}, meta: {} };
const LABEL = { approve: "APPROVED", reject: "REJECTED", flag: "FLAGGED" };
let toastTimer = 0;
const cur = () => S.items[S.idx] || null;
function counts() {
  let a = 0, r = 0, f = 0;
  for (const it of S.items) {
    const d = S.latest[it.id];
    if (d === "approve") a++; else if (d === "reject") r++; else if (d === "flag") f++;
  }
  return { a, r, f, total: S.items.length };
}
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 2400);
}
async function jget(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(url + " -> HTTP " + r.status);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                               body: JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.ok === false) throw new Error(d.error || ("HTTP " + r.status));
  return d;
}
function firstUndecided() {
  const i = S.items.findIndex(it => !(it.id in S.latest));
  return i >= 0 ? i : 0;
}
async function loadAll(keepPosition) {
  const keepId = keepPosition && cur() ? cur().id : null;
  try {
    const p = await jget("/items");
    S.items = Array.isArray(p.items) ? p.items : [];
    S.exists = !!p.exists;
  } catch (e) { S.items = []; S.exists = false; }
  try { S.latest = (await jget("/decisions")).latest || {}; } catch (e) { /* keep local */ }
  if (!S.items.length) { S.idx = 0; render(); return; }
  let i = keepId ? S.items.findIndex(x => x.id === keepId) : -1;
  if (i < 0) {
    let saved = null;
    try { saved = localStorage.getItem("sfxqa.last"); } catch (e) {}
    if (saved) i = S.items.findIndex(x => x.id === saved);
  }
  S.idx = i >= 0 ? i : firstUndecided();
  render();
}
function paintMeta(it) {
  const p = it.params || {}, m = S.meta[it.id] || null;
  const pick = (...keys) => {
    for (const k of keys) if (p[k] !== undefined && p[k] !== null && p[k] !== "") return p[k];
    return null;
  };
  const seed = pick("seed"), key = pick("key", "key_name", "note", "pitch_hz");
  const rawLen = pick("length_s", "length", "duration_s", "decay_s");
  const len = (m && m.duration_s != null) ? m.duration_s.toFixed(2) + "s"
            : (rawLen !== null ? rawLen + "s" : "\\u2014");
  const peak = (m && m.peak_dbfs != null) ? m.peak_dbfs.toFixed(1) + " dBFS" : "\\u2014";
  $("#metaline").textContent = [it.family || "\\u2014", it.name || it.id,
    "seed " + (seed !== null ? seed : "\\u2014"), "key " + (key !== null ? key : "\\u2014"),
    "len " + len, "peak " + peak].join("  \\u00b7  ");
}
async function loadMeta(it) {
  if (S.meta[it.id]) return;
  S.meta[it.id] = { pending: true };
  try { S.meta[it.id] = await jget("/meta/" + encodeURIComponent(it.id)); }
  catch (e) { S.meta[it.id] = { ok: false, error: String(e) }; }
  if (cur() && cur().id === it.id) paintMeta(it);
}
function render() {
  const c = counts();
  $("#stats").textContent = S.items.length
    ? (S.idx + 1) + " / " + c.total + " \\u00b7 " + c.a + " approved \\u00b7 " + c.r +
      " rejected \\u00b7 " + c.f + " flagged"
    : (S.exists ? "queue.json is empty" : "waiting for qa/queue.json\\u2026");
  const w = n => (c.total ? (100 * n / c.total).toFixed(2) + "%" : "0%");
  $("#seg-ap").style.width = w(c.a); $("#seg-rj").style.width = w(c.r); $("#seg-fl").style.width = w(c.f);
  const empty = $("#empty"), card = $("#card");
  if (!S.items.length) {
    empty.classList.remove("hidden"); card.classList.add("hidden");
    $("#empty-msg").textContent = S.exists
      ? "queue.json has no usable items \\u2014 press R to reload"
      : "waiting for qa/queue.json\\u2026";
    return;
  }
  empty.classList.add("hidden"); card.classList.remove("hidden");
  const it = cur(), d = S.latest[it.id], badge = $("#badge");
  badge.textContent = d ? LABEL[d] : "UNDECIDED";
  badge.className = "badge" + (d ? " " + d : "");
  $("#itemname").textContent = it.name || it.id;
  $("#itemid").textContent = it.id;
  $("#notes").textContent = it.notes || "";
  $("#notes").classList.toggle("hidden", !it.notes);
  const spec = $("#spec"), nospec = $("#nospec");
  if (it.has_png) {
    const u = "/png/" + encodeURIComponent(it.id);
    if (spec.dataset.src !== u) { spec.dataset.src = u; spec.src = u; }
    spec.classList.remove("hidden"); nospec.classList.add("hidden");
  } else {
    spec.removeAttribute("src"); spec.dataset.src = "";
    spec.classList.add("hidden"); nospec.classList.remove("hidden");
  }
  if (audio.dataset.id !== it.id) {          // only reload the audio when the item changes
    audio.dataset.id = it.id;
    audio.src = "/wav/" + encodeURIComponent(it.id);
    audio.load();
    audio.play().catch(() => {});            // blocked until the first user gesture
  }
  $("#params").textContent = JSON.stringify(it.params || {});
  paintMeta(it); loadMeta(it);
  try { localStorage.setItem("sfxqa.last", it.id); } catch (e) {}
}
function replay() {
  try { audio.currentTime = 0; audio.play().catch(() => toast("tap Replay once to enable audio")); }
  catch (e) {}
}
function go(i) {
  const n = S.items.length;
  if (!n) return;
  S.idx = ((i % n) + n) % n;
  render();
}
function nextUndecided(from) {
  const n = S.items.length;
  for (let k = 1; k <= n; k++) {
    const i = (from + k) % n;
    if (!(S.items[i].id in S.latest)) return i;
  }
  return -1;
}
function decide(kind) {
  const it = cur();
  if (!it) return;
  S.latest[it.id] = kind;                   // optimistic: advance before any round trip
  const target = nextUndecided(S.idx);
  if (target >= 0) go(target);
  else { go(S.idx + 1); toast("every item is decided"); }
  jpost("/decide", { id: it.id, decision: kind }).catch(err => {
    toast("save failed: " + err.message);
    jget("/decisions").then(d => { S.latest = d.latest || {}; render(); }).catch(() => {});
  });
}
async function reloadAll() {
  try { await jpost("/reload", {}); } catch (e) { toast("reload failed: " + e.message); }
  await loadAll(true);
  toast("queue reloaded");
}
document.querySelectorAll("[data-act]").forEach(b =>
  b.addEventListener("click", () => { b.blur(); decide(b.dataset.act); }));
[["#b-prev", () => go(S.idx - 1)], ["#b-next", () => go(S.idx + 1)],
 ["#b-replay", replay], ["#b-reload", reloadAll]].forEach(([sel, fn]) =>
  $(sel).addEventListener("click", function () { this.blur(); fn(); }));
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  const k = e.key;
  if (e.repeat && "aAxXfF".includes(k)) return;  // never approve 20 files by leaning on a key
  let handled = true;
  if (k === "a" || k === "A") decide("approve");
  else if (k === "x" || k === "X") decide("reject");
  else if (k === "f" || k === "F") decide("flag");
  else if (k === " ") replay();
  else if (k === "ArrowRight") go(S.idx + 1);
  else if (k === "ArrowLeft") go(S.idx - 1);
  else if (k === "r" || k === "R") reloadAll();
  else handled = false;
  if (handled) e.preventDefault();
});
// Keep polling while the pipeline has not produced a queue yet.
setInterval(() => { if (!S.exists || !S.items.length) loadAll(false); }, 2000);
loadAll(false);</script>
</body>
</html>
"""


# ------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sfx-qa/1.0"

    def log_message(self, fmt, *args):  # silence per-request access logs
        pass

    @property
    def store(self) -> Store:
        return self.server.store

    def _reply(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, status=200):
        self._reply(status, "application/json; charset=utf-8",
                    json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _send_file(self, path: Path, ctype: str):
        """Whole-file or single-range response (browsers need ranges to seek)."""
        try:
            size = path.stat().st_size
        except OSError:
            return self._json({"error": "not found"}, 404)
        start, end, status = 0, max(size - 1, 0), 200
        m = re.match(r"bytes=(\d*)-(\d*)$", (self.headers.get("Range") or "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2): end = min(int(m.group(2)), size - 1)
            else:  # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not length:
            return
        remaining = length
        try:
            with open(path, "rb") as f:
                f.seek(start)
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk: break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return b""
        if n <= 0: return b""
        if n > 1 << 20:
            self.close_connection = True
            return None
        return self.rfile.read(n)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._reply(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            if path == "/items":
                s = self.store
                keep = ("id", "family", "name", "params", "notes", "has_png", "has_wav", "wav_rel")
                with s.lock:
                    return self._json({"items": [{k: it[k] for k in keep} for it in s.items],
                                       "exists": s.exists, "error": s.error, "root": str(s.root)})
            if path == "/state":
                return self._json(self.store.state())
            if path == "/decisions":
                s = self.store
                with s.lock:
                    return self._json({"decisions": list(s.history), "latest": dict(s.latest),
                                       "count": len(s.history)})
            if path.rstrip("/") == "/reload":
                return self._json(self._reload())
            for prefix, kind in (("/wav/", "wav"), ("/png/", "png"), ("/meta/", "meta")):
                if path.startswith(prefix):
                    return self._item_route(kind, unquote(path[len(prefix):]))
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path.rstrip("/") == "/reload":
                return self._json(self._reload())
            if path == "/decide":
                return self._decide()
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _item_route(self, kind, iid):
        with self.store.lock:
            item = self.store.by_id.get(iid)
        if item is None:
            return self._json({"error": f"unknown id {iid!r}"}, 404)
        if kind == "meta":
            return self._json(self.store.meta_for(item))
        p = item["wav"] if kind == "wav" else item["png"]
        if p is None or not p.is_file():
            return self._json({"error": f"{kind} not found for {iid}"}, 404)
        self._send_file(p, "audio/wav" if kind == "wav" else "image/png")

    def _reload(self):
        n, d = self.store.load_queue(), self.store.load_decisions()
        log(f"reload: {n} queue item(s), {d} decision record(s)")
        return dict(self.store.state(), ok=True)

    def _decide(self):
        raw = self._body()
        if raw is None:
            return self._json({"ok": False, "error": "body too large"}, 413)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
        except Exception as exc:
            return self._json({"ok": False, "error": f"bad json: {exc}"}, 400)
        iid, decision = data.get("id"), data.get("decision")
        if not isinstance(iid, str) or not iid.strip():
            return self._json({"ok": False, "error": "id required"}, 400)
        if decision not in DECISIONS:
            return self._json({"ok": False, "error": "decision must be approve|reject|flag"}, 400)
        if iid not in self.store.by_id:
            # Record it anyway: a mid-session queue rewrite must not eat a decision.
            log(f"warning: decision for {iid!r}, which is not in the current queue")
        rec = self.store.append_decision(iid, decision)
        return self._json({"ok": True, "id": iid, "decision": decision, "ts": rec["ts"]})


def lan_ip():
    """Best-effort LAN address; the UDP connect sends no packets."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="QA approval server for the sfx-pack pipeline")
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    ap.add_argument("--root", default=None,
                    help="pipeline root holding qa/queue.json (default: script directory)")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent
    if not root.is_dir():
        log(f"warning: root {root} does not exist yet")
    store = Store(root)
    qn, dn = store.load_queue(), store.load_decisions()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.store = store

    log(f"root: {root}")
    log("queue: " + (f"{qn} item(s)" if store.exists
                     else "qa/queue.json missing - waiting for the pipeline"))
    log(f"decisions: {dn} record(s) in {store.dec_path}")
    print(f"\n  QA server listening on port {args.port}")
    print(f"    local:  http://127.0.0.1:{args.port}/")
    ip = lan_ip()
    if ip:
        print(f"    LAN:    http://{ip}:{args.port}/   (open this on your phone)")
    print("  Ctrl-C to stop\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
