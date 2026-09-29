"""Runtime settings, read once from CLAUDEYE_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Long edge (px) of images sent back to Claude. Current Claude models accept up to
    # 2576px, but an overview does not need it: details are fetched with zoom().
    max_edge: int
    # Hard ceiling for any returned image. 2000px: once a conversation holds >20 images the
    # API requires every image to be <=2000px, and long vision sessions get there quickly.
    hard_max_edge: int
    # Small crops are upscaled to at least this long edge so glyphs span several patches.
    zoom_min_edge: int
    jpeg_quality: int
    cache_dir: Path
    device: str  # auto | cuda | cpu
    ollama_host: str
    ollama_model: str
    gemini_model: str
    whisper_model: str


@lru_cache(maxsize=1)
def settings() -> Settings:
    cache = os.environ.get("CLAUDEYE_CACHE_DIR") or str(Path.home() / ".claudeye" / "cache")
    return Settings(
        max_edge=_int("CLAUDEYE_MAX_EDGE", 1568),
        hard_max_edge=_int("CLAUDEYE_HARD_MAX_EDGE", 2000),
        zoom_min_edge=_int("CLAUDEYE_ZOOM_MIN_EDGE", 1024),
        jpeg_quality=_int("CLAUDEYE_JPEG_QUALITY", 90),
        cache_dir=Path(cache),
        device=os.environ.get("CLAUDEYE_DEVICE", "auto").lower(),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        ollama_model=os.environ.get("CLAUDEYE_OLLAMA_MODEL", "qwen3.5:9b"),
        gemini_model=os.environ.get("CLAUDEYE_GEMINI_MODEL", "gemini-flash-latest"),
        whisper_model=os.environ.get("CLAUDEYE_WHISPER_MODEL", "auto"),
    )
