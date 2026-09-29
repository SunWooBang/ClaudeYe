"""Text -> frame retrieval for long videos with SigLIP 2 (Apache-2.0, multilingual text tower).
Frame embeddings are computed once per video and cached."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .. import video
from ..config import settings
from . import device as dev

MAX_FRAMES = 1800
BATCH = 16


def model_id() -> str:
    return os.environ.get("CLAUDEYE_EMBED_MODEL", "google/siglip2-base-patch16-224")


def _load():
    from transformers import AutoModel, AutoProcessor

    mid = model_id()
    model = AutoModel.from_pretrained(mid, dtype=dev.dtype()).to(dev.device()).eval()
    return AutoProcessor.from_pretrained(mid), model


def _feats(out):
    t = dev.torch()
    x = out if isinstance(out, t.Tensor) else getattr(out, "pooler_output", None)
    if x is None:
        x = out[0]
    return t.nn.functional.normalize(x.float(), dim=-1)


def index(path: Path, every: float | None = None, progress=None) -> tuple[np.ndarray, np.ndarray]:
    vi = video.info(path)
    every = every or max(1.0, vi.duration / MAX_FRAMES)
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{every:.3f}|{model_id()}".encode()).hexdigest()[:20]
    f = settings().cache_dir / "embeddings" / f"{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["times"], z["emb"]
    t = dev.torch()
    proc, model = dev.get(f"embed:{model_id()}", _load)
    times, embs, batch, bt = [], [], [], []
    frames = video.iter_small(path, width=384, every=every, gray=False,
                              keyframes_only=(vi.frames or 0) > 36_000)

    def flush():
        with dev.gpu_lock, t.inference_mode():
            inp = proc(images=batch, return_tensors="pt").to(dev.device())
            if "pixel_values" in inp:
                inp["pixel_values"] = inp["pixel_values"].to(dev.dtype())
            embs.append(_feats(model.get_image_features(**inp)).cpu().numpy().astype(np.float16))
        times.extend(bt)
        batch.clear()
        bt.clear()

    from PIL import Image

    for ts, arr in frames:
        batch.append(Image.fromarray(arr))
        bt.append(ts)
        if len(batch) == BATCH:
            flush()
            if progress:
                progress(min(0.99, ts / max(vi.duration, 1e-3)), f"indexing {ts:.0f}/{vi.duration:.0f}s")
    if batch:
        flush()
    ta, ea = np.array(times, dtype=np.float64), np.concatenate(embs, 0)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, times=ta, emb=ea)
    except OSError:
        pass
    return ta, ea


def query(texts: list[str]) -> tuple[np.ndarray, float, float]:
    """Normalised text embeddings plus the model's logit scale/bias (for sigmoid scores)."""
    t = dev.torch()
    proc, model = dev.get(f"embed:{model_id()}", _load)
    with dev.gpu_lock, t.inference_mode():
        inp = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(dev.device())
        e = _feats(model.get_text_features(**inp)).cpu().numpy()
    scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 1.0
    bias = float(model.logit_bias.item()) if hasattr(model, "logit_bias") else 0.0
    return e, scale, bias
