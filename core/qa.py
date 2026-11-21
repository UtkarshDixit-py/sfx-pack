"""Automated QA: the cull that happens before a human listens.

References (briefs/RESEARCH.md):
  defect rules (clip runs, DC, silence, crest)   ClipDaT / Hansen et al. 2021
  true peak + target loudness                    BS.1770-5; ASWG-R001
  MFCC self-similarity + dedup                   Foote 1999
  roughness                                     Daniel & Weber 1997 (MoSQITo)
No neural model is used here. Everything is deterministic and cheap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .dsp import FS

# --------------------------------------------------------------------------
# defect detection
# --------------------------------------------------------------------------
DEFAULTS = dict(
    max_clipped_pct=0.01,      # % of samples inside a run of >=2 at >=99.5% FS
    max_dc=0.001,              # absolute DC offset
    min_peak_dbfs=-60.0,
    min_lufs=-50.0,
    max_edge=0.02,             # |first/last sample| after fades
    hole_db=15.0,              # LTAS band gap below both neighbours
)


def dc_offset(x: np.ndarray) -> float:
    return float(np.mean(x))


def clipped_pct(x: np.ndarray) -> float:
    """ClipDaT-style: runs of >=2 samples at >=99.5% full scale."""
    a = np.abs(np.asarray(x))
    if a.size == 0:
        return 0.0
    hot = a >= 0.995
    if not hot.any():
        return 0.0
    # allow up to 3 sub-threshold samples inside a run
    idx = np.flatnonzero(hot)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 4) + 1)
    bad = sum(r.size for r in runs if r.size >= 2)
    return 100.0 * bad / a.size


def ltas_db(x: np.ndarray, fs: int = FS, bands: int = 24) -> np.ndarray:
    f, p = signal.welch(np.asarray(x), fs=fs, nperseg=min(8192, len(x)))
    edges = np.geomspace(50.0, min(fs / 2 * 0.98, 20000.0), bands + 1)
    out = np.full(bands, -120.0)
    for i in range(bands):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            out[i] = 10.0 * np.log10(np.mean(p[m]) + 1e-20)
    return out


def spectral_flatness(x: np.ndarray, fs: int = FS) -> float:
    """0 = pure tone, 1 = white noise. Used to decide whether a gap is a defect."""
    f, p = signal.welch(np.asarray(x), fs=fs, nperseg=min(8192, max(256, len(x))))
    m = (f > 200) & (f < fs / 2 * 0.9)
    p = p[m] + 1e-20
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def spectral_holes(x: np.ndarray, fs: int = FS, gap_db: float = 15.0) -> int:
    """Only meaningful for noise-like material; harmonics of a tone are not holes."""
    # 0.05 was too permissive: bitcrushed harmonic combs sit at 0.05-0.12 and
    # their partial valleys were being reported as spectral holes.
    if spectral_flatness(x, fs) < 0.15:
        return 0
    b = ltas_db(x, fs)
    holes = 0
    for i in range(1, len(b) - 1):
        if b[i] < b[i - 1] - gap_db and b[i] < b[i + 1] - gap_db:
            holes += 1
    return holes


def alias_score(x: np.ndarray, fs: int = FS) -> float:
    """Worst mirrored-peak ratio in dB.

    A naive oscillator's harmonic at f gains a mirror image at fs-f. Broadband
    noise has no such peaks, so this does not false-positive on hiss or air.
    Returns the strongest mirror level relative to its source peak (more negative
    is cleaner); -99 means no candidate peaks were found.
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    n = int(min(len(y), 1 << 15))
    if n < 1024:
        return -99.0
    seg = y[:n] * np.hanning(n)
    S = np.abs(np.fft.rfft(seg)) ** 2 + 1e-20
    f = np.fft.rfftfreq(n, 1.0 / fs)
    db = 10.0 * np.log10(S)
    peaks, _ = signal.find_peaks(db, prominence=12.0)
    worst = -99.0
    for pi in peaks:
        fp = f[pi]
        if fp < 3000.0 or fp > fs / 2 - 1500.0:
            continue
        mirror = fs - fp
        j = int(round(mirror / (fs / n)))
        if j <= 0 or j >= len(db):
            continue
        local = db[max(0, j - 2):min(len(db), j + 3)].max()
        worst = max(worst, float(local - db[pi]))
    return round(worst, 1)


def lf_share(x: np.ndarray, fs: int = FS, fc: float = 45.0) -> float:
    """Fraction of energy below fc. A sub-cycle LF partial reads as DC in a
    short file, which used to fail one-shots that were perfectly correct."""
    if len(x) < 8:
        return 0.0
    p = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(p[f < fc].sum() / (p.sum() + 1e-20))


def crest_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if rms <= 0 or peak <= 0:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


def audit(path: str | Path, fs: int | None = None,
          thresh: dict | None = None, loop: bool = False) -> dict:
    t = {**DEFAULTS, **(thresh or {})}
    x, sr = sf.read(str(path), always_2d=False, dtype="float64")
    fs = fs or sr
    mono = x if x.ndim == 1 else x.mean(axis=1)
    from . import render as R

    res = dict(
        file=str(path),
        seconds=round(len(mono) / fs, 4),
        channels=1 if x.ndim == 1 else 2,
        peak_dbfs=round(R.peak_dbfs(mono), 2),
        true_peak_dbtp=round(R.true_peak_dbtp(mono), 2),
        lufs=R.integrated_lufs(mono, fs),
        dc=round(dc_offset(mono), 6),
        lf_share=round(lf_share(mono, fs), 3),
        clipped_pct=round(clipped_pct(mono), 4),
        crest_db=round(crest_db(mono), 2),
        edge=max(abs(float(mono[0])), abs(float(mono[-1]))),
        holes=spectral_holes(mono, fs, t["hole_db"]),
        flatness=round(spectral_flatness(mono, fs), 4),
        alias_db=alias_score(mono, fs),
    )
    fails = []
    if res["clipped_pct"] > t["max_clipped_pct"]:
        fails.append("clipped")
    if abs(res["dc"]) > t["max_dc"]:
        # In a file shorter than ~3 cycles of its lowest partial the mean is not
        # a DC offset, it is the waveform. Only fail when there is no such content.
        if not (res["seconds"] < 0.35 and res["lf_share"] > 0.05):
            fails.append("dc_offset")
    if res["peak_dbfs"] < t["min_peak_dbfs"]:
        fails.append("silent")
    if res["lufs"] is not None and res["lufs"] < t["min_lufs"]:
        fails.append("too_quiet")
    if res["edge"] > t["max_edge"] and not loop:
        # A loop must NOT be faded, so nonzero edges are correct there.
        fails.append("edge_click")
    if res["holes"] >= 2:
        fails.append("spectral_hole")
    res["fails"] = fails
    res["ok"] = not fails
    return res


# --------------------------------------------------------------------------
# spectrogram for the QA sheet and for visual review
# --------------------------------------------------------------------------
def spectrogram_png(wav: str | Path, out_png: str | Path,
                    fs: int | None = None, dpi: int = 110) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, sr = sf.read(str(wav), always_2d=False, dtype="float64")
    fs = fs or sr
    mono = x if x.ndim == 1 else x.mean(axis=1)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=dpi)
    ax.specgram(mono, NFFT=1024, Fs=fs, noverlap=768, cmap="magma",
                vmin=-140, vmax=-20)
    ax.set_ylim(0, min(24000, fs / 2))
    ax.set_xlabel("s")
    ax.set_ylabel("Hz")
    ax.set_title(out_png.parent.name, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------
# perceptual features + dedup
# --------------------------------------------------------------------------
def _mel_filterbank(fs: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    # 20 Hz floor, not 50: below 50 the old bank made a 30 Hz and a 55 Hz sub
    # indistinguishable, so an entire 808/sub family collapsed into one cluster.
    mels = np.linspace(hz2mel(20.0), hz2mel(fs / 2 * 0.95), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if not (l < c < r):
            continue
        fb[i, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        fb[i, c:r] = (r - np.arange(c, r)) / max(1, r - c)
    return fb


def mfcc_features(x: np.ndarray, fs: int = FS, n_mfcc: int = 13) -> np.ndarray:
    """Mean+std MFCC vector. Deterministic, no external MIR dependency."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n_fft, hop = 1024, 512
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    f, t, Z = signal.stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    P = np.abs(Z) ** 2
    fb = _mel_filterbank(fs, n_fft)
    mel = np.log(fb @ P + 1e-10)
    from scipy.fftpack import dct
    cc = dct(mel, type=2, axis=0, norm="ortho")[1:n_mfcc + 1]
    return np.concatenate([cc.mean(axis=1), cc.std(axis=1)])


def _envelope(x: np.ndarray, points: int = 24) -> np.ndarray:
    """Coarse amplitude envelope. Separates a kick from a snare that share MFCCs."""
    n = len(x)
    idx = np.linspace(0, n, points + 1).astype(int)
    env = np.array([np.sqrt(np.mean(np.square(x[idx[i]:max(idx[i] + 1, idx[i + 1])])))
                    for i in range(points)])
    env = 20.0 * np.log10(env + 1e-9)
    return env - env.mean()


def _timbre(x: np.ndarray, fs: int = FS) -> np.ndarray:
    f, p = signal.welch(np.asarray(x), fs=fs, nperseg=min(2048, max(256, len(x))))
    p = p + 1e-20
    tot = p.sum()
    centroid = float((f * p).sum() / tot)
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * p).sum() / tot))
    cum = np.cumsum(p) / tot
    rolloff = float(f[min(len(f) - 1, np.searchsorted(cum, 0.95))])
    flatness = float(np.exp(np.mean(np.log(p))) / np.mean(p))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    # Dominant partial. This is the pitch axis the MFCC bank was blind to,
    # and it is why a 30 Hz and a 55 Hz sine used to score 0.99 similarity.
    peak_hz = float(f[int(np.argmax(p))]) if len(p) else 0.0
    return np.array([np.log10(centroid + 1), np.log10(bandwidth + 1),
                     np.log10(rolloff + 1), flatness, zcr,
                     np.log10(peak_hz + 1)])


def perceptual_blocks(x: np.ndarray, fs: int = FS, n_mfcc: int = 13) -> list[np.ndarray]:
    """Blocked feature vector for near-duplicate detection.

    Plain mean+std MFCC cannot tell two kicks apart (they score 0.97-0.996
    against each other) because it discards time. Adding the amplitude envelope,
    the timbre scalars and a log-mel snapshot, each block L2-normalised so
    cosine distance weights them equally, restores the discriminative axes.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)

    n_fft, hop = 1024, 512
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    f, _, Z = signal.stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    P = np.abs(Z) ** 2
    fb = _mel_filterbank(fs, n_fft)
    mel = np.log(fb @ P + 1e-10)
    from scipy.fftpack import dct
    cc = dct(mel, type=2, axis=0, norm="ortho")[1:n_mfcc + 1]

    blocks = [
        cc.mean(axis=1),
        cc.std(axis=1),
        mel.mean(axis=1) - mel.mean(),
        _envelope(x),
        _timbre(x[:min(len(x), 48000 * 2)], fs),
    ]
    out = []
    for b in blocks:
        b = np.asarray(b, dtype=np.float64)
        b = b - b.mean()
        out.append(b / (np.linalg.norm(b) + 1e-12))
    return out


def perceptual_features(x: np.ndarray, fs: int = FS, n_mfcc: int = 13) -> np.ndarray:
    return np.concatenate(perceptual_blocks(x, fs, n_mfcc))


def similarity_matrix_blocks(blocks_list: list[list[np.ndarray]]) -> np.ndarray:
    """Geometric mean of per-block similarities.

    Arithmetic mean is wrong here: four blocks agreeing drags the score to 0.97
    even when the timbre block flatly disagrees. Two sounds are duplicates only
    if they are similar in every respect, so the combination must be multiplicative.
    """
    n, k = len(blocks_list), len(blocks_list[0])
    S = np.ones((n, n))
    for b in range(k):
        M = np.array([bl[b] for bl in blocks_list], dtype=np.float64)
        S = np.minimum(S, M @ M.T)
    return S


def similarity_matrix(feats: np.ndarray) -> np.ndarray:
    F = np.asarray(feats, dtype=np.float64)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    return F @ F.T


def dedup(candidates: list[dict], near: float = 0.93, dup: float = 0.97) -> dict:
    """candidates: [{"id":..., "wav":...}] -> survivors + clusters (Foote SSM)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    feats, ids = [], []
    for c in candidates:
        x, sr = sf.read(c["wav"], always_2d=False, dtype="float64")
        feats.append(perceptual_blocks(x, sr))
        ids.append(c["id"])
    if not feats:
        return {"survivors": [], "clusters": [], "duplicates": []}
    S = similarity_matrix_blocks(feats)
    D = np.clip(1.0 - S, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - near, criterion="distance")
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    survivors, duplicates = [], []
    for lab, members in clusters.items():
        members.sort()
        survivors.append(ids[members[0]])
        for m in members[1:]:
            duplicates.append({"id": ids[m], "duplicate_of": ids[members[0]],
                               "sim": round(float(S[members[0], m]), 4),
                               "exact": bool(S[members[0], m] >= dup)})
    return {"survivors": survivors, "clusters": list(clusters.values()),
            "duplicates": duplicates}
