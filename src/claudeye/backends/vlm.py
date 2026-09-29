"""Second-opinion vision models.

- Ollama (local, private): Qwen3.5 / Qwen3-VL / Gemma ... images only, so video is sent as sampled frames.
- Gemini (cloud, opt-in via CLAUDEYE_ALLOW_CLOUD=1): native video + audio understanding.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from ..config import settings
from . import MissingBackend

_down_until = 0.0  # skip Ollama probes for a while after it was unreachable


# ---------------------------------------------------------------- Ollama

def _ollama(path: str, payload: dict | None = None, timeout: float = 2.0) -> dict:
    url = settings().ollama_host.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def ollama_models() -> list[str] | None:
    global _down_until
    if time.time() < _down_until:
        return None
    try:
        return [m["name"] for m in _ollama("/api/tags", timeout=1.5).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        _down_until = time.time() + 60
        return None


def ollama_unload() -> None:
    """Ask Ollama to release VRAM before a torch model loads (best effort)."""
    if ollama_models() is None:
        return
    try:
        for m in _ollama("/api/ps", timeout=1.5).get("models", []):
            _ollama("/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=5)
    except (urllib.error.URLError, OSError, ValueError):
        pass


_PREFERRED = ("qwen3.5", "qwen3-vl", "gemma4", "gemma3", "minicpm-v", "qwen2.5vl", "llava", "llama3.2-vision")


def ollama_vision_models() -> list[str]:
    """Installed models that report the 'vision' capability, best-known families first.
    (llama3.2-vision / mllama last: recent Ollama engines no longer load that architecture.)"""
    out = []
    for name in ollama_models() or []:
        try:
            caps = _ollama("/api/show", {"model": name}, timeout=3).get("capabilities") or []
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if "vision" in caps:
            out.append(name)

    def rank(n: str) -> int:
        return next((i for i, p in enumerate(_PREFERRED) if p in n), len(_PREFERRED) - 1)

    return sorted(out, key=rank)


def pick_ollama_model(requested: str | None) -> str:
    names = ollama_models()
    if names is None:
        raise MissingBackend("Ollama (not reachable at %s)" % settings().ollama_host, "ollama")
    want = requested or settings().ollama_model
    if want in names or f"{want}:latest" in names:
        return want
    vis = ollama_vision_models()
    if requested:
        raise ValueError(f"Ollama model {requested!r} is not pulled. Vision models available: {vis or 'none'}")
    if not vis:
        raise MissingBackend(f"an Ollama vision model ({want})", "ollama")
    return vis[0]


def _png(img: Image.Image, long_edge: int = 1280) -> bytes:
    if max(img.size) > long_edge:
        s = long_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def ollama_chat(prompt: str, images: list[Image.Image], *, model: str | None = None, json_mode: bool = False,
                long_edge: int = 1280, timeout: float = 600) -> tuple[str, str]:
    model = pick_ollama_model(model)
    payload = {"model": model, "stream": False, "keep_alive": "5m", "think": False,
               "options": {"temperature": 0, "num_ctx": 16384},
               "messages": [{"role": "user", "content": prompt,
                             "images": [base64.b64encode(_png(im, long_edge)).decode() for im in images]}]}
    if json_mode:
        payload["format"] = "json"
    try:
        res = _ollama("/api/chat", payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if "think" in body.lower():  # model without a thinking switch
            payload.pop("think")
            res = _ollama("/api/chat", payload, timeout=timeout)
        else:
            raise RuntimeError(f"Ollama error {e.code}: {body[:300]}") from e
    return res.get("message", {}).get("content", "").strip(), model


# ---------------------------------------------------------------- Gemini

def cloud_allowed() -> bool:
    return os.environ.get("CLAUDEYE_ALLOW_CLOUD", "").lower() in ("1", "true", "yes")


def _gemini():
    if not cloud_allowed():
        raise PermissionError("Cloud backends are disabled. The user must opt in by setting CLAUDEYE_ALLOW_CLOUD=1 "
                              "(and GEMINI_API_KEY) in the MCP server env; media will be sent to Google.")
    try:
        from google import genai
    except ImportError as e:
        raise MissingBackend("google-genai", "gemini") from e
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise PermissionError("GEMINI_API_KEY is not set in the MCP server environment.")
    return genai.Client()


def gemini_images(prompt: str, images: list[Image.Image], *, model: str | None = None,
                  json_mode: bool = False) -> tuple[str, str]:
    from google.genai import types

    client = _gemini()
    model = model or settings().gemini_model
    parts = [types.Part.from_bytes(data=_png(im, 2048), mime_type="image/png") for im in images]
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json" if json_mode else None)
    resp = client.models.generate_content(model=model, contents=[*parts, prompt], config=cfg)
    return (resp.text or "").strip(), model


def _uploaded(client, path: Path):
    """Upload once per file version; reuse for 46h (Gemini keeps files 48h)."""
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
    index_file = settings().cache_dir / "gemini_files.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else {}
    ent = index.get(key)
    if ent and time.time() - ent["t"] < 46 * 3600:
        try:
            f = client.files.get(name=ent["name"])
            if f.state and f.state.name == "ACTIVE":
                return f
        except Exception:
            pass
    f = client.files.upload(file=str(path))
    t0 = time.time()
    while not f.state or f.state.name != "ACTIVE":
        if f.state and f.state.name == "FAILED":
            raise RuntimeError(f"Gemini failed to process the video: {f.error}")
        if time.time() - t0 > 600:
            raise TimeoutError("Gemini video processing timed out")
        time.sleep(3)
        f = client.files.get(name=f.name)
    index[key] = {"name": f.name, "t": time.time()}
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(json.dumps(index))
    return f


def gemini_video(path: Path, prompt: str, *, start: float | None = None, end: float | None = None,
                 fps: float | None = None, resolution: str = "default", model: str | None = None) -> tuple[str, str]:
    from google.genai import types

    client = _gemini()
    model = model or settings().gemini_model
    f = _uploaded(client, path)
    meta = {}
    if fps:
        meta["fps"] = fps
    if start is not None:
        meta["start_offset"] = f"{start:.2f}s"
    if end is not None:
        meta["end_offset"] = f"{end:.2f}s"
    part = types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                      video_metadata=types.VideoMetadata(**meta) if meta else None)
    res = {"low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
           "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH}.get(resolution)
    cfg = types.GenerateContentConfig(temperature=0, media_resolution=res)
    resp = client.models.generate_content(model=model, contents=[part, prompt], config=cfg)
    return (resp.text or "").strip(), model


# ---------------------------------------------------------------- grounding output parsing

def parse_boxes(text: str, w: int, h: int, convention: str) -> list[dict]:
    """Parse JSON boxes from a VLM answer into pixel boxes.
    convention 'gemini': box_2d=[ymin,xmin,ymax,xmax] in 0-1000; 'qwen': bbox_2d=[x1,y1,x2,y2] in 0-1000."""
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    out = []
    for i, d in enumerate(data):
        if not isinstance(d, dict):
            continue
        b = d.get("box_2d") or d.get("bbox_2d") or d.get("bbox") or d.get("box")
        if not b or len(b) != 4:
            continue
        b = [float(v) for v in b]
        if convention == "gemini" or "box_2d" in d:
            y0, x0, y1, x1 = b
        else:
            x0, y0, x1, y1 = b
        scale = 1000.0 if max(b) > 1.5 else 1.0
        out.append({"id": i + 1, "label": str(d.get("label", d.get("name", ""))),
                    "box": [round(x0 / scale * w), round(y0 / scale * h), round(x1 / scale * w), round(y1 / scale * h)]})
    return out
