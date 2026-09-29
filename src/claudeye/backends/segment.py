"""Segmentation masks.

- points / boxes -> SAM 2.1 (Apache-2.0; hiera-tiny on CPU, hiera-large on GPU)
- text prompt    -> SAM 3 when configured (CLAUDEYE_SEGMENT_MODEL=facebook/sam3; gated: accept the licence on
                    Hugging Face and `hf auth login`), otherwise detect() boxes -> SAM 2.1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import device as dev


@dataclass
class Mask:
    label: str
    score: float
    box: tuple[int, int, int, int]
    area: int
    polygons: list[list[list[int]]] = field(default_factory=list)
    mask: np.ndarray | None = None  # full-image boolean mask (not serialised)


def sam2_id() -> str:
    return os.environ.get("CLAUDEYE_SAM2_MODEL") or (
        "facebook/sam2.1-hiera-large" if dev.device() == "cuda" else "facebook/sam2.1-hiera-tiny")


def sam3_id() -> str | None:
    v = os.environ.get("CLAUDEYE_SEGMENT_MODEL", "")
    return v if "sam3" in v.lower() else None


def _load_sam2():
    from transformers import Sam2Model, Sam2Processor

    mid = sam2_id()
    return Sam2Processor.from_pretrained(mid), Sam2Model.from_pretrained(mid).to(dev.device()).eval()


def _load_sam3():
    from transformers import Sam3Model, Sam3Processor

    mid = sam3_id()
    model = Sam3Model.from_pretrained(mid, dtype=dev.dtype()).to(dev.device()).eval()
    return Sam3Processor.from_pretrained(mid), model


def polygons(mask: np.ndarray, ox: int, oy: int, max_pts: int = 60) -> list[list[list[int]]]:
    import cv2

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(cs, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 4:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        while len(approx) > max_pts:
            eps *= 1.5
            approx = cv2.approxPolyDP(c, eps, True)
        out.append([[int(p[0][0]) + ox, int(p[0][1]) + oy] for p in approx])
    return out


def _to_mask(m: np.ndarray, label: str, score: float, ox: int, oy: int, full: tuple[int, int]) -> Mask | None:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    fm = np.zeros((full[1], full[0]), dtype=bool)
    fm[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m
    return Mask(label, float(score), (int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + 1 + ox,
                                      int(ys.max()) + 1 + oy), int(m.sum()), polygons(m, ox, oy), fm)


def sam2(img: Image.Image, region: tuple[int, int, int, int], *, points: list[list[float]] | None = None,
         boxes: list[list[float]] | None = None, labels: list[str] | None = None) -> list[Mask]:
    """points: [[x, y, 1|0], ...] (1=foreground, 0=background) for ONE object; boxes: one object per box.
    Coordinates are ORIGINAL image px."""
    t = dev.torch()
    proc, model = dev.get(f"sam2:{sam2_id()}", _load_sam2)
    x0, y0, x1, y1 = region
    crop = img.crop(region)
    kw = {}
    if boxes:
        kw["input_boxes"] = [[[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0] for b in boxes]]
    elif points:
        kw["input_points"] = [[[[p[0] - x0, p[1] - y0] for p in points]]]
        kw["input_labels"] = [[[int(p[2]) if len(p) > 2 else 1 for p in points]]]
    else:
        raise ValueError("sam2 needs points or boxes")
    with dev.gpu_lock:
        inputs = proc(images=crop, return_tensors="pt", **kw).to(dev.device())
        with t.inference_mode():
            out = model(**inputs, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
    scores = out.iou_scores.float().cpu().numpy().reshape(-1)
    res = []
    for i in range(masks.shape[0]):
        m = masks[i, 0].numpy().astype(bool)
        lab = (labels[i] if labels and i < len(labels) else f"object {i + 1}")
        mk = _to_mask(m, lab, scores[i] if i < len(scores) else 0.0, x0, y0, img.size)
        if mk:
            res.append(mk)
    return res


def sam3(img: Image.Image, region: tuple[int, int, int, int], prompts: list[str], threshold: float) -> list[Mask]:
    t = dev.torch()
    proc, model = dev.get(f"sam3:{sam3_id()}", _load_sam3)
    x0, y0, _, _ = region
    crop = img.crop(region)
    res = []
    with dev.gpu_lock:
        for p in prompts:
            inputs = proc(images=crop, text=p, return_tensors="pt").to(dev.device())
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dev.dtype())
            with t.inference_mode():
                out = model(**inputs)
            r = proc.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=0.5,
                                                        target_sizes=[(crop.height, crop.width)])[0]
            for m, s in zip(r["masks"].cpu().numpy(), r["scores"].float().cpu().numpy()):
                mk = _to_mask(m.astype(bool), p, float(s), x0, y0, img.size)
                if mk:
                    res.append(mk)
    return res
