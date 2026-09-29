"""OCR with line boxes: RapidOCR (PP-OCRv5/v6 detection + Korean PP-OCRv5 recognition) on ONNX Runtime.

CPU is fast enough (a 4K screenshot takes a few seconds), so no GPU dependency is needed.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import MissingBackend

_engines: dict[tuple, object] = {}
_lock = threading.Lock()
MAX_SIDE = 4096
LANGS = {"korean": "KOREAN", "ko": "KOREAN", "en": "EN", "english": "EN", "ch": "CH", "chinese": "CH",
         "japan": "JAPAN", "ja": "JAPAN", "latin": "LATIN"}
_JUNK = re.compile(r"^[\W_•·]{1,2}$")


@dataclass
class Line:
    text: str
    score: float
    box: tuple[int, int, int, int]
    quad: list[list[int]]
    row: int = 0


def _engine(lang: str, rotated: bool):
    try:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    except ImportError as e:
        raise MissingBackend("RapidOCR", "ocr") from e
    key = (lang, rotated)
    with _lock:
        if key not in _engines:
            code = LANGS.get(lang.lower(), "KOREAN")
            params = {"Rec.lang_type": getattr(LangRec, code), "Global.log_level": "error",
                      "Global.max_side_len": MAX_SIDE, "Global.use_cls": rotated}
            if code in ("KOREAN", "JAPAN", "LATIN"):
                params |= {"Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE}
            _engines[key] = RapidOCR(params=params)
        return _engines[key]


def _run(eng, img: Image.Image, scale: float, ox: int, oy: int) -> list[Line]:
    arr = np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR
    with _lock:
        r = eng(arr)
    out = []
    if r.txts is None:
        return out
    for quad, text, score in zip(r.boxes, r.txts, r.scores):
        text = text.strip()
        if not text or _JUNK.match(text):
            continue
        q = [[int(round(x / scale + ox)), int(round(y / scale + oy))] for x, y in quad]
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        out.append(Line(text, float(score), (min(xs), min(ys), max(xs), max(ys)), q))
    return out


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def read(img: Image.Image, box: tuple[int, int, int, int], *, lang: str = "korean", rotated: bool = False,
         upscale: float | None = None) -> tuple[list[Line], float]:
    """OCR `box` of `img`. Small regions are upscaled (tiny glyphs read far better at 2x).
    Regions longer than MAX_SIDE are processed in overlapping strips. Returns (lines, scale used)."""
    eng = _engine(lang, rotated)
    x0, y0, x1, y1 = box
    crop = img.crop(box)
    w, h = crop.size
    if upscale is None:
        upscale = 2.0 if w * h <= 1_500_000 and max(w, h) * 2 <= MAX_SIDE else 1.0
    if upscale != 1.0:
        crop = crop.resize((max(1, round(w * upscale)), max(1, round(h * upscale))), Image.LANCZOS)
    W, H = crop.size
    lines: list[Line] = []
    if max(W, H) <= MAX_SIDE:
        lines = _run(eng, crop, upscale, x0, y0)
    else:  # strips along the long side with 10% overlap
        step = int(MAX_SIDE * 0.9)
        vertical = H >= W
        L = H if vertical else W
        for a in range(0, L, step):
            b = min(L, a + MAX_SIDE)
            part = crop.crop((0, a, W, b) if vertical else (a, 0, b, H))
            sub = _run(eng, part, upscale, x0 + (0 if vertical else a / upscale), y0 + (a / upscale if vertical else 0))
            for ln in sub:
                if all(_iou(ln.box, o.box) < 0.5 for o in lines):
                    lines.append(ln)
            if b == L:
                break
    return reading_order(lines), upscale


def reading_order(lines: list[Line]) -> list[Line]:
    """Group into rows by vertical overlap, then left-to-right."""
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        cy = (ln.box[1] + ln.box[3]) / 2
        if rows:
            last = rows[-1]
            top = min(l.box[1] for l in last)
            bot = max(l.box[3] for l in last)
            if top <= cy <= bot:
                last.append(ln)
                continue
        rows.append([ln])
    out = []
    for i, row in enumerate(rows):
        for ln in sorted(row, key=lambda l: l.box[0]):
            ln.row = i
            out.append(ln)
    return out
