#!/usr/bin/env python3
"""Local WAV player with a live waveform.

    .venv/bin/python tools/player.py                 # plays everything under out/_reel
    .venv/bin/python tools/player.py --dir out/_qa   # any folder
    .venv/bin/python tools/player.py --dir out/_reel/kit --port 7790

Zero dependencies (stdlib server + one self-contained HTML page). No CDN, no
network calls. Opens on 127.0.0.1 and also binds the LAN address so the same
page works on a phone for auditioning away from the desk.

Controls: Space play/pause, arrows seek 1 s, J/K prev/next file, click waveform
to seek, drag the list or use J/K.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXT = {".wav"}


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waveform player</title>
<style>
  :root { --bg:#12151a; --panel:#191d24; --line:#2a313c; --txt:#e6eaf0; --dim:#8892a0;
          --wave:#4dd2a4; --wave2:#2b7f66; --play:#ff8a4c; --accent:#5aa9ff; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--txt);
              font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }
  #app { display:flex; height:100%; }
  #side { width:280px; border-right:1px solid var(--line); display:flex; flex-direction:column; }
  #side header { padding:10px 12px; border-bottom:1px solid var(--line); color:var(--dim); }
  #filter { width:100%; margin-top:8px; background:#0e1116; color:var(--txt);
            border:1px solid var(--line); border-radius:5px; padding:7px 9px; font:inherit; }
  .grp { padding:8px 12px 4px; color:#5aa9ff; background:#161a20; font-size:11px;
         letter-spacing:.04em; text-transform:uppercase; position:sticky; top:0; }
  #list { overflow-y:auto; flex:1; }
  .row { padding:7px 12px; cursor:pointer; border-bottom:1px solid #1e2229;
         white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .row:hover { background:#1c2229; }
  .row.sel { background:#1f2a33; color:#fff; border-left:2px solid var(--accent); }
  .row .d { color:var(--dim); font-size:11px; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #top { padding:14px 18px 6px; }
  #name { font-size:16px; color:#fff; word-break:break-all; }
  #meta { color:var(--dim); margin-top:4px; }
  #wrap { flex:1; display:flex; flex-direction:column; padding:8px 18px 18px; min-height:0; }
  canvas { width:100%; background:var(--panel); border:1px solid var(--line); border-radius:6px; }
  #wave { flex:1; min-height:160px; cursor:crosshair; }
  #bar { display:flex; gap:10px; align-items:center; padding:12px 0 0; }
  button { background:#222a33; color:var(--txt); border:1px solid var(--line);
           border-radius:6px; padding:9px 14px; font:inherit; cursor:pointer; }
  button:hover { background:#2a3440; }
  button.primary { background:var(--accent); color:#08111c; border-color:var(--accent); font-weight:700; }
  #time { color:var(--dim); min-width:110px; }
  #hint { color:var(--dim); font-size:11px; margin-top:10px; }
  #empty { margin:auto; color:var(--dim); }
  @media (max-width:760px){ #app{flex-direction:column} #side{width:auto;height:38%;border-right:0;border-bottom:1px solid var(--line)} }
</style></head><body>
<div id="app">
  <div id="side"><header><div id="folder">loading…</div>
    <input id="filter" placeholder="filter: laser, boom, 808…" autocomplete="off"></header>
    <div id="list"></div></div>
  <div id="main">
    <div id="top"><div id="name">—</div><div id="meta"></div></div>
    <div id="wrap">
      <canvas id="wave"></canvas>
      <div id="bar">
        <button class="primary" id="play">▶ Play</button>
        <button id="prev">⏮</button><button id="next">⏭</button>
        <span id="time">0.00 / 0.00 s</span>
        <label style="color:var(--dim)"><input type="checkbox" id="loop"> loop</label>
        <label style="color:var(--dim)"><input type="checkbox" id="auto" checked> auto-advance</label>
      </div>
      <div id="hint">Space play/pause · ←/→ seek 1 s · J/K prev/next · click waveform to seek</div>
    </div>
  </div>
</div>
<script>
const S = { files:[], i:-1, buf:null, src:null, ctx:null, playing:false, t0:0, off:0, peaks:null, dur:0 };
const $ = id => document.getElementById(id);
const wave = $('wave'), ctx2d = wave.getContext('2d');

function fmt(n){ return (n||0).toFixed(2); }
function ac(){ return S.ctx || (S.ctx = new (window.AudioContext||window.webkitAudioContext)()); }

function markSel(){ document.querySelectorAll('.row').forEach(e=>e.classList.toggle('sel', +e.dataset.k===S.i)); }

function renderList(){
  const q=($('filter').value||'').toLowerCase();
  const list=$('list'); list.innerHTML='';
  const groups=new Map();
  S.files.forEach((f,k)=>{
    if(q && !(f.path.toLowerCase().includes(q))) return;
    if(!groups.has(f.folder)) groups.set(f.folder, []);
    groups.get(f.folder).push([f,k]);
  });
  for(const [folder,items] of groups){
    const h=document.createElement('div'); h.className='grp';
    h.textContent=folder+'  ·  '+items.length; list.appendChild(h);
    for(const [f,k] of items){
      const d=document.createElement('div'); d.className='row'; d.dataset.k=k;
      d.innerHTML=f.name+'<div class="d">'+f.size+' · '+(f.seconds?f.seconds.toFixed(2)+'s':'—')+'</div>';
      d.onclick=()=>select(k); list.appendChild(d);
    }
  }
  markSel();
}

async function load(){
  const r = await fetch('/list'); const j = await r.json();
  S.files = j.files; $('folder').textContent = j.dir + '  (' + j.files.length + ' files)';
  renderList();
  if (S.files.length && S.i<0) select(0);
}

async function select(k){
  if (k<0 || k>=S.files.length) return;
  stop(); S.i=k; S.peaks=null;
  markSel();
  const f=S.files[k];
  $('name').textContent=f.name;
  $('meta').textContent=f.path+' · '+(f.sr||'?')+' Hz · '+(f.ch||'?')+' ch · '+(f.bits||'?')+' bit · '+f.size;
  wave.width = wave.clientWidth*devicePixelRatio; wave.height = wave.clientHeight*devicePixelRatio;
  draw();
  const res = await fetch('/wav?'+new URLSearchParams({p:f.path}));
  const ab = await res.arrayBuffer();
  const buf = await ac().decodeAudioData(ab);
  if (S.i!==k) return;
  S.buf=buf; S.dur=buf.duration; S.peaks=computePeaks(buf);
  draw(); $('time').textContent='0.00 / '+fmt(S.dur)+' s';
}

function computePeaks(buf){
  const ch=buf.getChannelData(0), W=1200, block=Math.max(1,Math.floor(ch.length/W));
  const out=new Float32Array(W);
  for(let i=0;i<W;i++){
    let mn=0,mx=0; const s=i*block, e=Math.min(ch.length,(i+1)*block);
    for(let j=s;j<e;j++){ const v=ch[j]; if(v<mn)mn=v; if(v>mx)mx=v; }
    out[i]=Math.max(Math.abs(mn),Math.abs(mx));
  }
  return out;
}

function draw(){
  const w=wave.width,h=wave.height,dpr=devicePixelRatio;
  ctx2d.setTransform(1,0,0,1,0,0);
  ctx2d.clearRect(0,0,w,h);
  ctx2d.fillStyle='#191d24'; ctx2d.fillRect(0,0,w,h);
  const mid=h/2;
  ctx2d.strokeStyle='#2a313c'; ctx2d.lineWidth=1*dpr;
  ctx2d.beginPath(); ctx2d.moveTo(0,mid); ctx2d.lineTo(w,mid); ctx2d.stroke();
  if(S.peaks){
    const g=ctx2d.createLinearGradient(0,0,0,h);
    g.addColorStop(0,'#4dd2a4'); g.addColorStop(.5,'#2b7f66'); g.addColorStop(1,'#2b7f66');
    ctx2d.fillStyle=g;
    const n=S.peaks.length, bw=w/n;
    for(let i=0;i<n;i++){
      const a=S.peaks[i]*(h/2)*0.92;
      ctx2d.fillRect(i*bw, mid-a, Math.max(bw*0.8,1), a*2 || 1);
    }
    // second channel in dimmer overlay if stereo
    if(S.buf && S.buf.numberOfChannels>1){
      ctx2d.globalAlpha=.25; ctx2d.fillStyle='#5aa9ff';
      const ch=S.buf.getChannelData(1), block=Math.max(1,Math.floor(ch.length/n));
      for(let i=0;i<n;i++){ let m=0; const s=i*block,e=Math.min(ch.length,(i+1)*block);
        for(let j=s;j<e;j++){ const v=Math.abs(ch[j]); if(v>m)m=v; }
        const a=m*(h/2)*0.92; ctx2d.fillRect(i*bw, mid-a, Math.max(bw*0.8,1), a*2||1); }
      ctx2d.globalAlpha=1;
    }
  }
  if(S.dur>0){
    const x=(pos()/S.dur)*w;
    ctx2d.strokeStyle='#ff8a4c'; ctx2d.lineWidth=2*dpr;
    ctx2d.beginPath(); ctx2d.moveTo(x,0); ctx2d.lineTo(x,h); ctx2d.stroke();
  }
}

function pos(){ return S.playing ? Math.min(S.dur, ac().currentTime - S.t0 + S.off) : S.off; }

function play(){
  if(!S.buf) return;
  const a=ac(); if(a.state==='suspended') a.resume();
  const s=a.createBufferSource(); s.buffer=S.buf;
  if($('loop').checked) s.loop=true;
  s.connect(a.destination);
  S.t0=a.currentTime; s.start(0, S.off % S.buf.duration);
  S.src=s; S.playing=true; $('play').textContent='❚❚ Pause';
  s.onended=()=>{ if(S.src!==s) return; S.playing=false; $('play').textContent='▶ Play'; S.off=0;
                  if($('auto').checked) select(S.i+1); };
  requestAnimationFrame(tick);
}
function stop(){ if(S.src){ try{S.src.stop()}catch(e){} S.src=null; } S.playing=false; S.off=0; $('play').textContent='▶ Play'; draw(); }
function toggle(){ S.playing ? (S.off=pos(), stop()) : play(); }
function seek(t){ S.off=Math.max(0,Math.min(S.dur, t)); if(S.playing){ try{S.src.stop()}catch(e){} S.playing=false; play(); } else draw(); }

function tick(){
  if(!S.playing) return;
  $('time').textContent=fmt(pos())+' / '+fmt(S.dur)+' s';
  draw();
  requestAnimationFrame(tick);
}

$('play').onclick=toggle; $('prev').onclick=()=>select(S.i-1); $('next').onclick=()=>select(S.i+1);
$('filter').addEventListener('input', renderList);
wave.onclick=e=>{ const r=wave.getBoundingClientRect(); seek(((e.clientX-r.left)/r.width)*S.dur); };
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.code==='Space'){ e.preventDefault(); toggle(); }
  else if(e.key==='ArrowRight'){ e.preventDefault(); seek(pos()+1); }
  else if(e.key==='ArrowLeft'){ e.preventDefault(); seek(pos()-1); }
  else if(e.key==='j'||e.key==='J'){ select(S.i-1); }
  else if(e.key==='k'||e.key==='K'){ select(S.i+1); }
});
addEventListener('resize',()=>{ wave.width=wave.clientWidth*devicePixelRatio; wave.height=wave.clientHeight*devicePixelRatio; draw(); });
load();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    directory: Path = ROOT / "out" / "_reel"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/list":
            return self._send(200, json.dumps(self._listing()).encode(), "application/json")
        if u.path == "/wav":
            p = self._resolve(q.get("p", [""])[0])
            if not p:
                return self._send(404, b"not found", "text/plain")
            data = p.read_bytes()
            return self._send(200, data, mimetypes.guess_type(p.name)[0] or "audio/wav")
        return self._send(404, b"not found", "text/plain")

    def _resolve(self, rel: str) -> Path | None:
        if not rel:
            return None
        try:
            p = (ROOT / rel).resolve()
            p.relative_to(self.directory.resolve())
        except Exception:
            return None
        return p if p.is_file() and p.suffix.lower() in AUDIO_EXT else None

    def _listing(self) -> dict:
        files = []
        for p in sorted(self.directory.rglob("*")):
            if p.suffix.lower() not in AUDIO_EXT or not p.is_file():
                continue
            info = {"name": p.name, "path": str(p.relative_to(ROOT)),
                    "folder": str(p.parent.relative_to(ROOT)),
                    "size": f"{p.stat().st_size / 1024:.0f} KB"}
            try:
                import wave
                with wave.open(str(p), "rb") as w:
                    info.update(sr=w.getframerate(), ch=w.getnchannels(),
                                bits=w.getsampwidth() * 8,
                                seconds=w.getnframes() / w.getframerate())
            except Exception:
                info.update(seconds=None)
            files.append(info)
        return {"dir": str(self.directory.relative_to(ROOT)), "files": files}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out")
    ap.add_argument("--port", type=int, default=7790)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    d = (ROOT / a.dir).resolve()
    if not d.is_dir():
        print(f"no such directory: {d}")
        return 1
    Handler.directory = d
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"player: http://127.0.0.1:{a.port}/")
    print(f"phone:  http://{lan_ip()}:{a.port}/")
    print(f"folder: {d}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
