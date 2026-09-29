"""Torch device selection and a small model manager sized for one consumer GPU.

- one lock serialises GPU work (12 GB VRAM is shared by every backend and Ollama)
- models are loaded lazily and evicted least-recently-used (MAX_LOADED)
"""

from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from ..config import settings
from . import MissingBackend

MAX_LOADED = 3
gpu_lock = threading.RLock()
_models: OrderedDict[str, Any] = OrderedDict()
_meta: dict[str, dict] = {}


def torch():
    try:
        import torch as t
    except ImportError as e:
        raise MissingBackend("PyTorch/transformers", "ml") from e
    return t


def device() -> str:
    t = torch()
    if settings().device == "cpu":
        return "cpu"
    return "cuda" if t.cuda.is_available() else "cpu"


def dtype():
    t = torch()
    if device() == "cuda":
        return t.bfloat16 if t.cuda.is_bf16_supported() else t.float16
    return t.float32


def get(key: str, loader: Callable[[], Any]) -> Any:
    """Return a cached model bundle, loading it (and evicting the LRU one) if needed."""
    with gpu_lock:
        if key in _models:
            _models.move_to_end(key)
            return _models[key]
        while len(_models) >= MAX_LOADED:
            old, _ = _models.popitem(last=False)
            _meta.pop(old, None)
            free()
        if device() == "cuda":
            from .vlm import ollama_unload

            ollama_unload()  # give torch the VRAM Ollama may be holding
        t0 = time.perf_counter()
        _models[key] = loader()
        _meta[key] = {"load_s": round(time.perf_counter() - t0, 1), "device": device()}
        return _models[key]


def free() -> None:
    gc.collect()
    try:
        import torch as t

        if t.cuda.is_available():
            t.cuda.empty_cache()
    except ImportError:
        pass


def loaded() -> dict[str, dict]:
    return dict(_meta)


def vram() -> dict | None:
    try:
        import torch as t
    except ImportError:
        return None
    if not t.cuda.is_available():
        return None
    free_b, total_b = t.cuda.mem_get_info()
    return {"gpu": t.cuda.get_device_name(0), "free_gb": round(free_b / 2**30, 2), "total_gb": round(total_b / 2**30, 2)}
