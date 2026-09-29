"""Image references.

Every image tool accepts a *ref*:
  - an image file:            D:/shots/screen.png
  - a frame of a video:       D:/clips/bug.mp4@12.5   (seconds, MM:SS(.ms) or H:MM:SS)
Relative paths resolve against the server's working directory (the Claude Code project).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 250_000_000

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".ico", ".jfif"}
VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".3gp", ".ogv",
}


def parse_time(s: str) -> float:
    """'12.5' -> 12.5, '01:02' -> 62.0, '1:02:03.5' -> 3723.5, '90s' -> 90.0"""
    s = s.strip().lower().removesuffix("s")
    parts = s.split(":")
    if not 1 <= len(parts) <= 3 or any(p == "" for p in parts):
        raise ValueError(f"bad time: {s!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    if secs < 0:
        raise ValueError(f"negative time: {s!r}")
    return secs


def fmt_time(t: float) -> str:
    """62.5 -> '01:02.50', 3723.5 -> '1:02:03.50'"""
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}" if h else f"{int(m):02d}:{s:05.2f}"


@dataclass(frozen=True)
class Ref:
    path: Path
    time: float | None = None

    @property
    def is_video(self) -> bool:
        return self.path.suffix.lower() in VIDEO_EXTS

    def __str__(self) -> str:
        p = self.path.as_posix()
        return f"{p}@{self.time:.3f}" if self.time is not None else p


def _resolve(s: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(s)))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def parse_ref(ref: str) -> Ref:
    raw = ref.strip().strip('"').strip("'")
    if raw.lower().startswith("file:///"):
        raw = raw[8:]
    elif raw.lower().startswith("file://"):
        raw = raw[7:]
    if "@" in raw:
        head, _, tail = raw.rpartition("@")
        try:
            t = parse_time(tail)
        except ValueError:
            t = None
        if t is not None:
            p = _resolve(head)
            if p.is_file():
                return Ref(p, t)
    p = _resolve(raw)
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {p}")
    return Ref(p, None)


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.seek(0)
        im = ImageOps.exif_transpose(im)
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        import numpy as np

        a = np.asarray(im, dtype=np.float32)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (a - lo) / (hi - lo or 1.0) * 255.0
        return Image.fromarray(a.astype(np.uint8)).convert("RGB")
    if im.mode == "P":
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        return bg
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


_cache: OrderedDict[tuple, Image.Image] = OrderedDict()
_lock = threading.Lock()
_CACHE_MAX = 8


def load(ref: str | Ref) -> tuple[Image.Image, Ref]:
    """Load a ref as an RGB PIL image (cached; callers must not mutate it)."""
    r = parse_ref(ref) if isinstance(ref, str) else ref
    st = r.path.stat()
    key = (str(r.path), st.st_mtime_ns, st.st_size, r.time)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key], r
    if r.time is not None or r.is_video:
        from . import video

        img = video.frame_at(r.path, r.time or 0.0)
    else:
        img = _open_image(r.path)
    with _lock:
        _cache[key] = img
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return img, r
