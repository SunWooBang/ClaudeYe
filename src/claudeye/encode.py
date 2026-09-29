"""PIL image -> MCP image content, with a size/token budget."""

from __future__ import annotations

import io
import math

from mcp.server.mcpserver import Image as McpImage
from PIL import Image

from .config import settings

PNG_MAX_BYTES = 900_000


def est_tokens(w: int, h: int) -> int:
    """Visual tokens on current Claude models: one per 28x28 patch."""
    return math.ceil(w / 28) * math.ceil(h / 28)


def fit(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= long_edge:
        return img
    s = long_edge / max(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def to_mcp(img: Image.Image, *, fmt: str = "auto") -> tuple[McpImage, dict]:
    """Encode for Claude. PNG keeps text crisp; falls back to JPEG when PNG is large."""
    img = fit(img, settings().hard_max_edge)
    if img.mode != "RGB":
        img = img.convert("RGB")
    data, kind = b"", fmt
    if fmt in ("auto", "png"):
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=6)
        data, kind = buf.getvalue(), "png"
    if fmt == "jpeg" or (fmt == "auto" and len(data) > PNG_MAX_BYTES):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=settings().jpeg_quality, subsampling=0)
        data, kind = buf.getvalue(), "jpeg"
    meta = {"px": f"{img.width}x{img.height}", "tokens": est_tokens(img.width, img.height),
            "kb": round(len(data) / 1024)}
    return McpImage(data=data, format=kind), meta
