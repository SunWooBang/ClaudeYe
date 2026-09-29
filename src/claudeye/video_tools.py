"""Video -> images + text: metadata, storyboards, dense frame windows, change timelines."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from PIL import ImageDraw

from . import analyze, render, video
from .config import settings
from .encode import to_mcp
from .refs import Ref, fmt_time, parse_ref, parse_time

MAX_SHEET_FRAMES = 36


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def _video_ref(path: str) -> Ref:
    r = parse_ref(path)
    return Ref(r.path, None)


def _t(v: float | str | None, default: float) -> float:
    if v is None or v == "":
        return default
    return parse_time(v) if isinstance(v, str) else float(v)


def _window(vi: video.VideoInfo, start, end) -> tuple[float, float]:
    s = min(max(0.0, _t(start, 0.0)), vi.duration)
    e = min(max(s, _t(end, vi.duration)), vi.duration)
    if e - s < 1e-3:
        e = min(vi.duration, s + 1.0)
    return s, e


def _sheet_layout(vi: video.VideoInfo, n: int, cols: int | None, long_edge: int) -> tuple[int, int]:
    portrait = vi.height > vi.width
    cols = cols or (min(n, 6) if portrait else min(n, 4 if n > 4 else n))
    cols = max(1, min(cols, n))
    gap, label_h = 4, 20
    cell_w = (long_edge - gap * (cols + 1)) // cols
    rows = math.ceil(n / cols)
    ar = vi.height / max(1, vi.width)
    while cell_w > 96 and 24 + gap + rows * (label_h + round(cell_w * ar) + gap) > long_edge:
        cell_w -= 16
    return cols, cell_w


def sheet_image(path: str, vi: video.VideoInfo, times: list[float], labels: list[str], *, cols: int | None,
                header: str, max_edge: int | None, boxes: list[list[int] | None] | None = None):
    """Contact sheet as a PIL image + the actual decoded frame times."""
    s = settings()
    long_edge = max(512, min(max_edge or s.max_edge, s.hard_max_edge))
    cols, cell_w = _sheet_layout(vi, len(times), cols, long_edge)
    frames = video.sample(path, times, long_edge=max(cell_w, 256) * 2)
    cells = []
    for (t, im), label, box in zip(frames, labels, boxes or [None] * len(frames)):
        if box:
            sx, sy = im.width / vi.width, im.height / vi.height
            d = ImageDraw.Draw(im)
            d.rectangle([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], outline=(255, 40, 120), width=3)
        cells.append((im, label))
    return render.contact_sheet(cells, cols=cols, cell_w=cell_w, header=header), [t for t, _ in frames]


def _sheet(path: str, vi: video.VideoInfo, times: list[float], labels: list[str], *, cols: int | None,
           header: str, max_edge: int | None, boxes: list[list[int] | None] | None = None):
    sheet, actual = sheet_image(path, vi, times, labels, cols=cols, header=header, max_edge=max_edge, boxes=boxes)
    return to_mcp(sheet), actual


def _summary(vi: video.VideoInfo) -> dict:
    d = {"duration": fmt_time(vi.duration), "seconds": vi.duration, "size": f"{vi.width}x{vi.height}",
         "fps": vi.fps, "audio": bool(vi.audio)}
    if vi.rotation:
        d["rotation"] = vi.rotation
    return d


# ---------------------------------------------------------------- info

def video_info(path: str) -> list:
    r = _video_ref(path)
    vi = video.info(r.path)
    d = vi.as_dict()
    d["duration_hms"] = fmt_time(vi.duration)
    tips = ("Next: video_storyboard(path) for a visual overview; video_changes(path) for screen recordings; "
            "transcribe(path) for speech. Any frame can be used as an image ref: 'path@SECONDS'.")
    return [_j(d) + "\n" + tips]


# ---------------------------------------------------------------- storyboard

def _pick_scenes(an: analyze.Analysis, s: float, e: float, n: int) -> list[tuple[float, str]]:
    scenes = an.scenes(s, e)
    while len(scenes) > n:  # merge the shortest scene into its shorter neighbour
        i = min(range(len(scenes)), key=lambda k: scenes[k][1] - scenes[k][0])
        if i == 0:
            j = 1
        elif i == len(scenes) - 1:
            j = i - 1
        else:
            j = i - 1 if (scenes[i - 1][1] - scenes[i - 1][0]) < (scenes[i + 1][1] - scenes[i + 1][0]) else i + 1
        a, b = sorted((i, j))
        scenes[a:b + 1] = [(scenes[a][0], scenes[b][1])]
    alloc = [1] * len(scenes)
    for _ in range(n - len(scenes)):  # give spare frames to the scene with the most time per frame
        k = max(range(len(scenes)), key=lambda q: (scenes[q][1] - scenes[q][0]) / alloc[q])
        alloc[k] += 1
    out = []
    for si, ((a, b), m) in enumerate(zip(scenes, alloc)):
        if m == 1:
            out.append((an.stable_time(a, b), f"S{si + 1}"))
        else:
            step = (b - a) / m
            out += [(an.stable_time(a + k * step, a + (k + 1) * step), f"S{si + 1}") for k in range(m)]
    return out


def _pick_hybrid(an: analyze.Analysis, s: float, e: float, n: int, min_frac: float) -> list[tuple[float, str]]:
    """One stable frame per shot, then the biggest change events, then fill the largest gaps."""
    scenes = an.scenes(s, e)
    if len(scenes) >= n:
        return _pick_scenes(an, s, e, n)
    picks = [(an.stable_time(a, b), "") for a, b in scenes]
    min_sep = (e - s) / (4 * n)
    for ev in sorted(an.events(s, e, min_frac=min_frac, merge_gap=0.6), key=lambda v: -v["changed_pct"]):
        if len(picks) >= n:
            break
        t = ev["settled"]
        if s <= t <= e and all(abs(t - p) > min_sep for p, _ in picks):
            picks.append((t, "Δ"))
    while len(picks) < n:
        edges = sorted([s, *[p for p, _ in picks], e])
        k = max(range(len(edges) - 1), key=lambda q: edges[q + 1] - edges[q])
        a, b = edges[k], edges[k + 1]
        if b - a < 2 * min_sep / 4:
            break
        picks.append((an.stable_time(a, b), ""))
    return sorted(picks)


def _pick_changes(an: analyze.Analysis, s: float, e: float, n: int, min_frac: float) -> list[tuple[float, str]]:
    evs = an.events(s, e, min_frac=min_frac, merge_gap=0.6)
    if len(evs) > n - 1:
        keep = sorted(sorted(evs, key=lambda v: -v["changed_pct"])[: n - 1], key=lambda v: v["start"])
    else:
        keep = evs
    out = [(s, "start")] + [(v["settled"], f"after {fmt_time(v['start'])}") for v in keep]
    return out


def storyboard_picks(path: str, start: float | str | None = None, end: float | str | None = None,
                     frames: int = 16, mode: str = "auto", progress=None):
    """Choose storyboard times. Returns (ref, info, window, mode, scenes, picks[(t, tag)], analysis)."""
    r = _video_ref(path)
    vi = video.info(r.path)
    s, e = _window(vi, start, end)
    n = max(1, min(int(frames), MAX_SHEET_FRAMES))
    an = analyze.analyze(r.path, progress)
    mode = (mode or "auto").lower()
    scenes = an.scenes(s, e)
    if mode == "auto":
        mode = "hybrid"
    if mode == "hybrid":
        picks = _pick_hybrid(an, s, e, n, 0.002)
    elif mode == "scenes":
        picks = _pick_scenes(an, s, e, n)
    elif mode == "changes":
        picks = _pick_changes(an, s, e, n, 0.002)
    elif mode == "uniform":
        step = (e - s) / n
        picks = [(s + (k + 0.5) * step, "") for k in range(n)]
    else:
        raise ValueError("mode must be auto | hybrid | scenes | changes | uniform")
    picks.sort(key=lambda p: p[0])
    return r, vi, (s, e), mode, scenes, picks, an


def video_storyboard(path: str, start: float | str | None = None, end: float | str | None = None,
                     frames: int = 16, mode: str = "auto", cols: int | None = None,
                     max_edge: int | None = None, progress=None, as_pil: bool = False):
    r, vi, (s, e), mode, scenes, picks, an = storyboard_picks(path, start, end, frames, mode, progress)
    times = [t for t, _ in picks]
    bounds = [a for a, _ in scenes]

    def scene_of(t: float) -> int:
        return max(0, sum(1 for a in bounds if a <= t + 1e-6) - 1) + 1 if bounds else 1

    labels = [f"#{i + 1} {fmt_time(t)} S{scene_of(t)}" + (" Δ" if tag == "Δ" else "")
              for i, (t, tag) in enumerate(picks)]
    header = f"{r.path.name} | {fmt_time(s)}-{fmt_time(e)} of {fmt_time(vi.duration)} | {len(times)} frames ({mode})"
    sheet, actual = sheet_image(str(r.path), vi, times, labels, cols=cols, header=header, max_edge=max_edge)
    frames_out = [{"i": i + 1, "t": round(t, 3), "at": fmt_time(t), "ref": f"{r.path.as_posix()}@{t:.3f}"}
                  for i, t in enumerate(actual)]
    if as_pil:
        return sheet, frames_out
    mimg, meta = to_mcp(sheet)
    info = {"video": _summary(vi), "window": [round(s, 3), round(e, 3)], "mode": mode,
            "analysis": an.mode, "scenes_in_window": len(scenes),
            "scenes": [[round(a, 2), round(b, 2)] for a, b in scenes[:60]], "frames": frames_out, "image": meta}
    tip = ("Labels: #index time S<shot> (Δ = frame right after a visual change). Zoom into a frame: "
           "zoom('<ref>', region). Denser look at a span: video_frames(path, start=..., end=..., count=...).")
    return [_j(info) + "\n" + tip, mimg]


# ---------------------------------------------------------------- frames

def video_frames(path: str, times: list[float | str] | None = None, start: float | str | None = None,
                 end: float | str | None = None, count: int | None = None, fps: float | None = None,
                 layout: str = "auto", cols: int | None = None, max_edge: int | None = None) -> list:
    r = _video_ref(path)
    vi = video.info(r.path)
    if times:
        ts = [min(max(0.0, _t(t, 0.0)), vi.duration) for t in times][:MAX_SHEET_FRAMES]
    else:
        s, e = _window(vi, start, end)
        if fps:
            n = max(1, min(MAX_SHEET_FRAMES, int((e - s) * fps) + 1))
            ts = [s + k / fps for k in range(n) if s + k / fps <= e + 1e-6]
        else:
            n = max(1, min(int(count or 8), MAX_SHEET_FRAMES))
            ts = [s + (e - s) * (k / (n - 1) if n > 1 else 0.5) for k in range(n)]
    layout = (layout or "auto").lower()
    if layout == "auto":
        layout = "separate" if len(ts) <= 3 else "sheet"
    s_ = settings()
    if layout == "separate":
        long_edge = max(512, min(max_edge or s_.max_edge, s_.hard_max_edge))
        got = video.sample(r.path, ts, long_edge=long_edge)
        out: list = []
        listing = []
        for t, im in got:
            mimg, meta = to_mcp(im)
            listing.append({"t": round(t, 3), "at": fmt_time(t), "ref": f"{r.path.as_posix()}@{t:.3f}", "image": meta})
            out.append(mimg)
        return [_j({"video": _summary(vi), "frames": listing}) + "\nImages follow the order of 'frames'.", *out]
    labels = [f"#{i + 1} {fmt_time(t)}" for i, t in enumerate(ts)]
    header = f"{r.path.name} | {len(ts)} frames {fmt_time(ts[0])}-{fmt_time(ts[-1])}"
    (mimg, meta), actual = _sheet(str(r.path), vi, ts, labels, cols=cols, header=header, max_edge=max_edge)
    listing = [{"i": i + 1, "t": round(t, 3), "at": fmt_time(t), "ref": f"{r.path.as_posix()}@{t:.3f}"}
               for i, t in enumerate(actual)]
    return [_j({"video": _summary(vi), "frames": listing, "image": meta}), mimg]


# ---------------------------------------------------------------- changes

# Fraction of changed pixels (in the analysed area) that counts as an event.
# high ~ single typed characters; medium ~ cursor-sized changes and up; low ~ dialogs/page changes.
SENSITIVITY = {"high": 0.00005, "medium": 0.0003, "low": 0.003}


def video_changes(path: str, start: float | str | None = None, end: float | str | None = None,
                  sensitivity: str = "medium", region: str | None = None, max_events: int = 40,
                  sheet: bool = True, max_edge: int | None = None, progress=None) -> list:
    r = _video_ref(path)
    vi = video.info(r.path)
    s, e = _window(vi, start, end)
    if region:
        rbox = render.parse_region(region, vi.width, vi.height)
        an = analyze.analyze_region(r.path, s, e, rbox, progress)
    else:
        rbox = None
        an = analyze.analyze(r.path, progress)
    if sensitivity not in SENSITIVITY:
        raise ValueError(f"sensitivity must be one of {', '.join(SENSITIVITY)}")
    min_frac = SENSITIVITY[sensitivity]
    evs = an.events(s, e, min_frac=min_frac, merge_gap=0.6)
    shown = evs[: max(1, max_events)]
    for i, v in enumerate(shown):
        v["id"] = i + 1
        v["at"] = fmt_time(v["start"])
        v["after_ref"] = f"{r.path.as_posix()}@{v['settled']:.3f}"
        v["before_ref"] = f"{r.path.as_posix()}@{v['start']:.3f}"
    idx = an.window(s, e)
    activity = float(np.mean(an.change[idx] >= min_frac)) if len(idx) else 0.0
    info = {"video": _summary(vi), "window": [round(s, 3), round(e, 3)], "sensitivity": sensitivity,
            "region": list(rbox) if rbox else "full frame",
            "events_total": len(evs), "active_time_pct": round(100 * activity, 1),
            "cuts": [round(c, 2) for c in an.cuts if s <= c <= e][:100], "events": shown}
    out: list = []
    if sheet and shown:
        pick = shown if len(shown) <= 16 else sorted(sorted(shown, key=lambda v: -v["changed_pct"])[:16],
                                                      key=lambda v: v["start"])
        times = [v["settled"] for v in pick]
        labels = [f"E{v['id']} {fmt_time(v['start'])} {v['kind']}" for v in pick]
        header = f"{r.path.name} | change events (state AFTER each change, pink box = changed area)"
        (mimg, meta), _ = _sheet(str(r.path), vi, times, labels, cols=None, header=header,
                                 max_edge=max_edge, boxes=[v["box"] for v in pick])
        info["image"] = meta
        out.append(mimg)
    tip = ("start=last frame before the change, settled=first frame after it. Compare states with "
           "compare('<before_ref>', '<after_ref>'), or zoom into after_ref.")
    return [_j(info) + "\n" + tip, *out]
