"""One-pass low-res video analysis, cached on disk.

Produces per-sample shot-change scores, changed-pixel fractions and changed-region
boxes. From these: scene (shot) boundaries, change events (great for screen
recordings), and 'stable' representative frames.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import video
from .config import settings

VERSION = 3
MAX_SAMPLES = 2400
FULL_DECODE_MAX_FRAMES = 36_000  # longer videos get a keyframes-only pass
WIDTH = 384
DIFF_THRESHOLD = 22
Progress = Callable[[float, str], None]


@dataclass
class Analysis:
    path: str
    duration: float
    mode: str  # "full" | "keyframes"
    every: float
    times: np.ndarray  # (N,) seconds
    cut: np.ndarray  # (N,) shot-change score vs previous sample, 0..1
    change: np.ndarray  # (N,) fraction of pixels changed vs previous sample
    boxes: np.ndarray  # (N,4) changed-region bbox in ORIGINAL px, zeros if none
    cuts: list[float]

    # ------------------------------------------------------------ queries
    def window(self, start: float, end: float) -> np.ndarray:
        return np.nonzero((self.times >= start - 1e-6) & (self.times <= end + 1e-6))[0]

    def scenes(self, start: float = 0.0, end: float | None = None) -> list[tuple[float, float]]:
        end = self.duration if end is None else end
        bounds = [0.0, *self.cuts, self.duration]
        out = []
        for a, b in zip(bounds, bounds[1:]):
            a2, b2 = max(a, start), min(b, end)
            if b2 - a2 > 1e-3:
                out.append((a2, b2))
        return out

    def stable_time(self, a: float, b: float) -> float:
        """Least-changing sample in the middle 60% of [a, b] (avoids blur/transitions)."""
        lo, hi = a + 0.2 * (b - a), b - 0.2 * (b - a)
        idx = self.window(lo, hi)
        if len(idx) == 0:
            return (a + b) / 2
        mid = (a + b) / 2
        score = self.change[idx] + 1e-4 * np.abs(self.times[idx] - mid)
        return float(self.times[idx[int(np.argmin(score))]])

    def events(self, start: float, end: float, min_frac: float, merge_gap: float) -> list[dict]:
        idx = [i for i in self.window(start, end) if i > 0 and self.change[i] >= min_frac]
        groups: list[list[int]] = []
        for i in idx:
            if groups and self.times[i] - self.times[groups[-1][-1]] <= merge_gap:
                groups[-1].append(i)
            else:
                groups.append([i])
        out = []
        n = len(self.times)
        for g in groups:
            first, last = g[0], g[-1]
            bx = self.boxes[g]
            bx = bx[(bx[:, 2] > bx[:, 0]) & (bx[:, 3] > bx[:, 1])]
            box = ([int(bx[:, 0].min()), int(bx[:, 1].min()), int(bx[:, 2].max()), int(bx[:, 3].max())]
                   if len(bx) else None)
            peak = float(self.change[g].max())
            is_cut = any(abs(self.times[i] - c) < 1e-6 for i in g for c in self.cuts)
            out.append({
                "start": round(float(self.times[first - 1]), 3),  # last frame before the change
                "end": round(float(self.times[last]), 3),
                "settled": round(float(self.times[min(last + 1, n - 1)]), 3),  # first frame after it
                "changed_pct": round(100 * peak, 2),
                "box": box,
                "kind": "cut" if is_cut else ("large" if peak > 0.25 else "local"),
            })
        return out


_mem: dict[str, Analysis] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|v{VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def analyze(path: str | Path, progress: Progress | None = None) -> Analysis:
    p = Path(path)
    key = _key(p)
    if key in _mem:
        return _mem[key]
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if key in _mem:
            return _mem[key]
        an = _load(key) or _compute(p, progress)
        _save(key, an)
        _mem[key] = an
        return an


def _cache_file(key: str) -> Path:
    return settings().cache_dir / "analysis" / f"{key}.npz"


def _load(key: str) -> Analysis | None:
    f = _cache_file(key)
    if not f.exists():
        return None
    try:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return Analysis(times=z["times"], cut=z["cut"], change=z["change"], boxes=z["boxes"], **meta)
    except Exception:
        return None


def _save(key: str, an: Analysis) -> None:
    f = _cache_file(key)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        meta = {"path": an.path, "duration": an.duration, "mode": an.mode, "every": an.every, "cuts": an.cuts}
        np.savez_compressed(f, times=an.times, cut=an.cut, change=an.change, boxes=an.boxes,
                            meta=np.array(json.dumps(meta)))
    except OSError:
        pass


def _hist(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    feats = [x / max(1.0, x.sum()) for x in (h, s, v)]
    return np.concatenate(feats)


def measure(p: Path, *, start: float = 0.0, end: float | None = None, every: float | None = None,
            keyframes_only: bool = False, region: tuple[int, int, int, int] | None = None,
            progress: Progress | None = None):
    """Per-sample (t, cut_score, changed_fraction, changed_box_in_original_px) vs the previous sample.
    `region` (original px) restricts the change mask to that area."""
    vi = video.info(p)
    dur = max((end if end is not None else vi.duration) - start, 1e-3)
    sx = vi.width / WIDTH
    kernel = np.ones((2, 2), np.uint8)
    prev_g = prev_h = None
    rs = None
    last_report = start
    for t, rgb in video.iter_small(p, width=WIDTH, start=start, end=end, every=every,
                                   keyframes_only=keyframes_only, gray=False):
        g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
        sy = vi.height / g.shape[0]
        if region is not None and rs is None:
            rs = (int(region[0] / sx), int(region[1] / sy), max(int(region[0] / sx) + 1, int(np.ceil(region[2] / sx))),
                  max(int(region[1] / sy) + 1, int(np.ceil(region[3] / sy))))
        hist = _hist(rgb)
        if prev_g is None:
            c, frac, box = 0.0, 0.0, (0, 0, 0, 0)
        else:
            diff = cv2.absdiff(g, prev_g)
            hist_d = float(np.abs(hist - prev_h).sum() / 6.0)
            pix_d = float(diff.mean() / 255.0)
            c = 0.5 * hist_d + 0.5 * min(1.0, pix_d * 3.0)
            ox = oy = 0
            if rs is not None:
                diff = diff[rs[1]:rs[3], rs[0]:rs[2]]
                ox, oy = rs[0], rs[1]
            mask = cv2.morphologyEx((diff > DIFF_THRESHOLD).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            frac = float(mask.mean()) if mask.size else 0.0
            pts = cv2.findNonZero(mask)
            if pts is not None:
                x, y, w, h = cv2.boundingRect(pts)
                x, y = x + ox, y + oy
                box = (int(x * sx), int(y * sy), int(min(vi.width, (x + w) * sx)), int(min(vi.height, (y + h) * sy)))
            else:
                box = (0, 0, 0, 0)
        yield t, c, frac, box
        prev_g, prev_h = g, hist
        if progress and t - last_report >= dur / 20:
            last_report = t
            progress(min(0.99, (t - start) / dur), f"analyzing {t:.0f}/{start + dur:.0f}s")


def _compute(p: Path, progress: Progress | None) -> Analysis:
    vi = video.info(p)
    dur = max(vi.duration, 1e-3)
    total = vi.frames or int(dur * (vi.fps or 30))
    keyframes_only = total > FULL_DECODE_MAX_FRAMES
    every = max(1.0 / (vi.fps or 30), dur / MAX_SAMPLES)
    rows = list(measure(p, every=every, keyframes_only=keyframes_only, progress=progress))
    ta = np.array([r[0] for r in rows], dtype=np.float64)
    ca = np.array([r[1] for r in rows], dtype=np.float32)
    cuts = _detect_cuts(ta, ca, keyframes_only)
    return Analysis(path=p.as_posix(), duration=vi.duration, mode="keyframes" if keyframes_only else "full",
                    every=round(every, 4), times=ta, cut=ca, change=np.array([r[2] for r in rows], dtype=np.float32),
                    boxes=np.array([r[3] for r in rows], dtype=np.int32).reshape(-1, 4), cuts=cuts)


def analyze_region(path: str | Path, start: float, end: float, region: tuple[int, int, int, int],
                   progress: Progress | None = None) -> Analysis:
    """Uncached, region-restricted analysis of a time window (for watching one UI panel)."""
    p = Path(path)
    vi = video.info(p)
    every = max(1.0 / (vi.fps or 30), (end - start) / MAX_SAMPLES)
    rows = list(measure(p, start=start, end=end, every=every, region=region, progress=progress))
    ta = np.array([r[0] for r in rows], dtype=np.float64)
    ca = np.array([r[1] for r in rows], dtype=np.float32)
    return Analysis(path=p.as_posix(), duration=vi.duration, mode="region", every=round(every, 4), times=ta,
                    cut=ca, change=np.array([r[2] for r in rows], dtype=np.float32),
                    boxes=np.array([r[3] for r in rows], dtype=np.int32).reshape(-1, 4),
                    cuts=_detect_cuts(ta, ca, False))


def _detect_cuts(t: np.ndarray, c: np.ndarray, sparse: bool) -> list[float]:
    """Adaptive threshold: score must be high in absolute terms AND vs its neighbourhood."""
    n = len(c)
    abs_thr = 0.38 if sparse else 0.22
    min_len = 1.0 if sparse else 0.5
    cuts: list[float] = []
    for i in range(1, n):
        if c[i] < abs_thr:
            continue
        lo, hi = max(1, i - 8), min(n, i + 9)
        nb = np.concatenate([c[lo:i], c[i + 1:hi]])
        base = float(np.median(nb)) if len(nb) else 0.0
        if c[i] < 3.0 * base + 0.05:
            continue
        if cuts and t[i] - cuts[-1] < min_len:
            if c[i] > c[np.searchsorted(t, cuts[-1])]:
                cuts[-1] = float(t[i])
            continue
        cuts.append(float(t[i]))
    return cuts
