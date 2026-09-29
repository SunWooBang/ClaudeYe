"""Monocular relative depth with Depth Anything V2 Small (Apache-2.0). Larger value = closer."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from . import device as dev


def model_id() -> str:
    return os.environ.get("CLAUDEYE_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")


def _load():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    mid = model_id()
    return AutoImageProcessor.from_pretrained(mid), AutoModelForDepthEstimation.from_pretrained(mid).to(dev.device()).eval()


def estimate(img: Image.Image) -> np.ndarray:
    """Relative inverse depth, normalised 0 (far) .. 1 (near), same size as img."""
    t = dev.torch()
    proc, model = dev.get(f"depth:{model_id()}", _load)
    with dev.gpu_lock:
        inputs = proc(images=img, return_tensors="pt").to(dev.device())
        with t.inference_mode():
            pred = model(**inputs).predicted_depth
        pred = t.nn.functional.interpolate(pred.unsqueeze(1).float(), size=(img.height, img.width),
                                           mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
    lo, hi = np.percentile(pred, 1), np.percentile(pred, 99)
    return np.clip((pred - lo) / (hi - lo or 1.0), 0, 1)


def colorize(d: np.ndarray) -> Image.Image:
    import cv2

    c = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
