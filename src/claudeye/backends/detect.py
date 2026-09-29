"""Open-vocabulary detection (text -> boxes) with LLMDet / MM-Grounding-DINO (Apache-2.0) via transformers.

Large images are sliced into overlapping tiles (SAHI-style) plus one full-image pass, then merged with
per-label NMS, so small objects in 4K screenshots or photos are not lost to the model's ~800px input.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image

from . import device as dev

TILE = 800
MODEL_INPUT_LONG = 1333


@dataclass
class Det:
    label: str
    score: float
    box: tuple[int, int, int, int]


def model_id() -> str:
    env = os.environ.get("CLAUDEYE_DETECT_MODEL")
    if env:
        return env
    return "iSEE-Laboratory/llmdet_large" if dev.device() == "cuda" else "iSEE-Laboratory/llmdet_tiny"


def _load():
    t = dev.torch()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    mid = model_id()
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev.device()).eval()
    _ = t
    return proc, model


def _run(proc, model, img: Image.Image, labels: list[str], box_thr: float, text_thr: float) -> list[Det]:
    t = dev.torch()
    inputs = proc(images=img, text=[labels], return_tensors="pt").to(dev.device())
    with t.inference_mode():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, input_ids=inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[(img.height, img.width)])[0]
    names = res.get("text_labels", res.get("labels"))
    dets = []
    for b, s, name in zip(res["boxes"].tolist(), res["scores"].tolist(), names):
        name = name if isinstance(name, str) else labels[int(name)]
        label = _canon(name, labels)
        if label is not None:
            dets.append(Det(label, float(s), tuple(int(round(v)) for v in b)))
    return dets


def _tiles(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    step = int(size * (1 - overlap))
    xs = list(range(0, max(1, w - size) + 1, step)) or [0]
    ys = list(range(0, max(1, h - size) + 1, step)) or [0]
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    return [(x, y, min(w, x + size), min(h, y + size)) for y in ys for x in xs]


def _canon(name: str, labels: list[str]) -> str | None:
    """Map a predicted phrase ('red circle blue square', 'dot [SEP]') onto exactly one requested label."""
    words = set(name.replace("[SEP]", " ").replace("[CLS]", " ").split())
    if not words:
        return None
    scored = sorted(((len(words & set(lb.split())) / len(words | set(lb.split())), lb) for lb in labels), reverse=True)
    if scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
        return None
    return scored[0][1]


def _nms(dets: list[Det], iou: float = 0.5, contain: float = 0.8) -> list[Det]:
    """Per-label NMS, then drop boxes mostly inside a better box (tile-boundary fragments)."""
    t = dev.torch()
    from torchvision.ops import batched_nms

    if not dets:
        return dets
    boxes = t.tensor([d.box for d in dets], dtype=t.float32)
    scores = t.tensor([d.score for d in dets], dtype=t.float32)
    names = sorted({d.label for d in dets})
    idx = t.tensor([names.index(d.label) for d in dets])
    kept = [dets[i] for i in batched_nms(boxes, scores, idx, iou).tolist()]
    kept.sort(key=lambda d: -d.score)
    out: list[Det] = []
    for d in kept:
        area = max(1, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        inside = False
        for o in out:
            ix = max(0, min(d.box[2], o.box[2]) - max(d.box[0], o.box[0]))
            iy = max(0, min(d.box[3], o.box[3]) - max(d.box[1], o.box[1]))
            if ix * iy / area >= contain:
                inside = True
                break
        if not inside:
            out.append(d)
    return out


def detect(img: Image.Image, box: tuple[int, int, int, int], labels: list[str], *, box_thr: float = 0.3,
           text_thr: float = 0.25, tile: str = "auto") -> tuple[list[Det], dict]:
    proc, model = dev.get(f"detect:{model_id()}", _load)
    x0, y0, x1, y1 = box
    crop = img.crop(box)
    w, h = crop.size
    use_tiles = tile == "on" or (tile == "auto" and max(w, h) > 2 * MODEL_INPUT_LONG)
    with dev.gpu_lock:
        dets = _run(proc, model, crop, labels, box_thr, text_thr)
        n_tiles = 1
        if use_tiles:
            tiles = _tiles(w, h, TILE, 0.2)
            n_tiles += len(tiles)
            for tx0, ty0, tx1, ty1 in tiles:
                for d in _run(proc, model, crop.crop((tx0, ty0, tx1, ty1)), labels, box_thr, text_thr):
                    bx0, by0, bx1, by1 = d.box
                    dets.append(Det(d.label, d.score, (bx0 + tx0, by0 + ty0, bx1 + tx0, by1 + ty0)))
    dets = _nms(dets)
    dets = [Det(d.label, d.score, (d.box[0] + x0, d.box[1] + y0, d.box[2] + x0, d.box[3] + y0)) for d in dets]
    dets.sort(key=lambda d: (d.box[1] // 20, d.box[0]))
    return dets, {"model": model_id(), "device": dev.device(), "passes": n_tiles}
