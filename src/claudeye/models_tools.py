"""Model-backed tools: OCR, speech-to-text, open-vocabulary detection, segmentation, depth,
text->frame video search and second-opinion VLMs. Every backend is optional and lazily loaded."""

from __future__ import annotations

import difflib
import json
import os
from typing import Annotated, Any

from mcp.server.mcpserver import Context
from pydantic import Field

from .params import MaxEdge, OptRegion, OptTime, Ref

__all__ = ["ocr", "transcribe", "detect", "segment", "depth", "ask_vision", "ask_video", "video_search",
           "vision_status"]


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def _norm(s: str) -> str:
    return "".join(s.casefold().split())


def _cloud_ok() -> bool:
    from .backends.vlm import cloud_allowed

    return cloud_allowed()


# ---------------------------------------------------------------- OCR

def ocr(ref: str, region: str | None = None, lang: str = "korean", query: str | None = None,
        min_score: float = 0.5, rotated: bool = False, upscale: float | None = None, mark: bool = True,
        max_lines: int = 300, max_edge: int | None = None) -> list:
    from . import image_tools, refs, render
    from .backends import ocr as ocr_be
    from .encode import to_mcp

    img, r = refs.load(ref)
    W, H = img.size
    box = render.parse_region(region, W, H, image_tools._grid_for(r, W, H, None))
    lines, used = ocr_be.read(img, box, lang=lang, rotated=rotated, upscale=upscale)
    lines = [ln for ln in lines if ln.score >= min_score]
    if query:
        q = _norm(query)

        def sim(t: str) -> float:
            n = _norm(t)
            return 1.0 if q in n else difflib.SequenceMatcher(None, q, n).ratio()

        lines = [ln for ln in lines if sim(ln.text) >= 0.6]
        lines.sort(key=lambda ln: -sim(ln.text))
    total = len(lines)
    lines = lines[:max_lines]
    items = []
    for i, ln in enumerate(lines):
        it = {"id": i + 1, "text": ln.text, "score": round(ln.score, 3), "box": list(ln.box)}
        if ln.score < 0.9:
            it["verify"] = True
        items.append(it)
    rows: dict[int, list[str]] = {}
    for ln in lines:
        rows.setdefault(ln.row, []).append(ln.text)
    head = {"ref": str(r), "region": list(box), "lang": lang, "ocr_scale": used, "lines": total}
    if total > len(lines):
        head["truncated_to"] = len(lines)
    parts = [_j(head)]
    if not query:
        parts.append("TEXT (reading order, same-row items separated by ' | '):\n" +
                     "\n".join(" | ".join(v) for v in rows.values()))
    parts.append("LINES:\n" + "\n".join(_j(it) for it in items))
    low = sum(1 for it in items if it.get("verify"))
    parts.append(f"{low} low-confidence line(s) marked verify:true -> check them with zoom(ref, box). "
                 "OCR can drop/merge spaces and confuse similar glyphs; the image is the ground truth.")
    out: list = ["\n".join(parts)]
    if mark and items:
        from . import image_tools as it_

        view = render.render_view(img, box, long_edge=it_._edge(max_edge), min_edge=768)
        render.draw_marks(view, [{"id": it["id"], "box": it["box"]} for it in items], width=1, label_size=11)
        mimg, _ = to_mcp(view.image)
        out.append(mimg)
    return out


# ---------------------------------------------------------------- speech

def transcribe(path: str, start: float | str | None = None, end: float | str | None = None,
               language: str = "auto", words: bool = False, model: str | None = None,
               prompt: str | None = None, progress=None) -> list:
    from . import video
    from .backends import asr
    from .refs import fmt_time, parse_ref, parse_time

    r = parse_ref(path)
    if r.path.suffix.lower() not in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"):
        vi = video.info(r.path)
        if not vi.audio:
            raise ValueError("this video has no audio stream")
    s = parse_time(start) if isinstance(start, str) else float(start or 0.0)
    e = parse_time(end) if isinstance(end, str) else (float(end) if end is not None else None)
    res = asr.transcribe(r.path, start=s, end=e, language=language, words=words, model=model,
                         prompt=prompt, progress=progress)
    head = {k: v for k, v in res.items() if k != "segments"} | {"file": r.path.as_posix(),
                                                                 "segments": len(res["segments"])}
    lines = []
    for seg in res["segments"]:
        lines.append(f"[{fmt_time(seg['start'])}-{fmt_time(seg['end'])}] {seg['text']}")
        if words and seg.get("words"):
            lines.append("   words: " + _j(seg["words"]))
    tip = ("Timestamps are absolute. To see what is on screen while something is said: "
           "video_frames(path, times=[...]) or zoom('<path>@SECONDS', region).")
    body = "\n".join(lines) if lines else "(no speech detected)"
    return [_j(head) + "\n" + body + "\n" + tip]


# ---------------------------------------------------------------- detection

def _labels(prompt: str) -> list[str]:
    parts = [p.strip().strip(".").lower() for p in prompt.replace(";", ",").replace(".", ",").split(",")]
    return [p for p in parts if p]


def detect(ref: str, prompt: str, region: str | None = None, threshold: float = 0.3, text_threshold: float = 0.25,
           tile: str = "auto", max_results: int = 200, mark: bool = True, max_edge: int | None = None) -> list:
    from . import image_tools, refs, render
    from .backends import detect as det_be
    from .encode import to_mcp

    img, r = refs.load(ref)
    W, H = img.size
    box = render.parse_region(region, W, H, image_tools._grid_for(r, W, H, None))
    labels = _labels(prompt)
    if not labels:
        raise ValueError("prompt must name at least one object, e.g. 'red circle, blue square'")
    dets, meta = det_be.detect(img, box, labels, box_thr=threshold, text_thr=text_threshold, tile=tile)
    dets = dets[:max_results]
    counts: dict[str, int] = {}
    for d in dets:
        counts[d.label] = counts.get(d.label, 0) + 1
    items = [{"id": i + 1, "label": d.label, "score": round(d.score, 3), "box": list(d.box)} for i, d in enumerate(dets)]
    head = {"ref": str(r), "region": list(box), "prompt": labels, "counts": counts, "total": len(items)} | meta
    text = (_j(head) + "\nDETECTIONS:\n" + "\n".join(_j(it) for it in items) +
            "\nVerify on the overlay: missed objects -> lower threshold or tile='on'; duplicates/false hits -> raise "
            "threshold. Prompts work best as short English noun phrases.")
    out: list = [text]
    if mark:
        view = render.render_view(img, box, long_edge=image_tools._edge(max_edge), min_edge=768)
        render.draw_marks(view, [{"id": it["id"], "box": it["box"]} for it in items], width=2, label_size=11)
        out.append(to_mcp(view.image)[0])
    return out


# ---------------------------------------------------------------- segmentation

def _parse_rows(spec: str | None, n: int) -> list[list[float]]:
    rows = []
    for chunk in (spec or "").replace("|", ";").split(";"):
        nums = [float(v) for v in chunk.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
                .replace(",", " ").split()]
        if len(nums) >= n:
            rows.append(nums)
    return rows


def segment(ref: str, prompt: str | None = None, points: str | None = None, boxes: str | None = None,
            region: str | None = None, threshold: float = 0.35, max_results: int = 50,
            max_edge: int | None = None) -> list:
    from . import image_tools, refs, render
    from .backends import detect as det_be
    from .backends import segment as seg
    from .encode import to_mcp

    img, r = refs.load(ref)
    W, H = img.size
    box = render.parse_region(region, W, H, image_tools._grid_for(r, W, H, None))
    backend = ""
    if prompt:
        labels = _labels(prompt)
        if seg.sam3_id():
            masks, backend = seg.sam3(img, box, labels, threshold), seg.sam3_id()
        else:
            dets, meta = det_be.detect(img, box, labels, box_thr=threshold)
            masks = seg.sam2(img, box, boxes=[list(d.box) for d in dets], labels=[d.label for d in dets]) if dets else []
            backend = f"{meta['model']} boxes -> {seg.sam2_id()}"
    elif points:
        pts = _parse_rows(points, 2)
        masks, backend = seg.sam2(img, box, points=pts, labels=["object"]), seg.sam2_id()
    elif boxes:
        bxs = [b[:4] for b in _parse_rows(boxes, 4)]
        masks, backend = seg.sam2(img, box, boxes=bxs), seg.sam2_id()
    else:
        raise ValueError("give prompt='what to segment', or points='x,y; x,y,0', or boxes='x0,y0,x1,y1; ...'")
    masks = sorted(masks, key=lambda m: -m.score)[:max_results]
    items = []
    for i, m in enumerate(masks):
        poly = m.polygons[0] if m.polygons else []
        items.append({"id": i + 1, "label": m.label, "score": round(m.score, 3), "box": list(m.box),
                      "area_px": m.area, "area_pct": round(100 * m.area / (W * H), 3),
                      "polygon": poly if len(poly) <= 40 else poly[:: max(1, len(poly) // 40)]})
    view = render.render_view(img, box, long_edge=image_tools._edge(max_edge), min_edge=768)
    render.draw_marks(view, [{"id": it["id"], "polygon": m.polygons[0]} for it, m in zip(items, masks) if m.polygons],
                      width=2, label_size=11)
    head = {"ref": str(r), "region": list(box), "backend": backend, "masks": len(items)}
    return [_j(head) + "\nMASKS:\n" + "\n".join(_j(it) for it in items) +
            "\nPolygons are in ORIGINAL px. Check the overlay; refine with points='x,y' (fg) / 'x,y,0' (bg).",
            to_mcp(view.image)[0]]


# ---------------------------------------------------------------- depth

def depth(ref: str, region: str | None = None, points: str | None = None, max_edge: int | None = None) -> list:
    from . import image_tools, refs, render
    from .backends import depth as depth_be
    from .encode import to_mcp

    img, r = refs.load(ref)
    W, H = img.size
    box = render.parse_region(region, W, H, image_tools._grid_for(r, W, H, None))
    x0, y0, x1, y1 = box
    d = depth_be.estimate(img.crop(box))
    g = render.Grid.auto(x1 - x0, y1 - y0)
    xs, ys = g.xs(x1 - x0), g.ys(y1 - y0)
    table = [" ".join(f"{d[ys[rr]:ys[rr + 1], xs[c]:xs[c + 1]].mean():.2f}" for c in range(g.cols))
             for rr in range(g.rows)]
    info: dict[str, Any] = {"ref": str(r), "region": list(box), "model": depth_be.model_id(),
                            "scale": "relative: 0=far .. 1=near (not metres)"}
    if points:
        info["points"] = [{"xy": [int(p[0]), int(p[1])],
                           "depth": round(float(d[min(max(int(p[1]) - y0, 0), d.shape[0] - 1),
                                                  min(max(int(p[0]) - x0, 0), d.shape[1] - 1)]), 3)}
                          for p in _parse_rows(points, 2)]
    view = render.render_view(img, box, long_edge=image_tools._edge(max_edge), min_edge=768,
                              crop=depth_be.colorize(d))
    return [_j(info) + f"\nMean depth per cell ({g.cols}x{g.rows}, rows top->bottom):\n" + "\n".join(table) +
            "\nColour map: red/yellow = near, blue = far.", to_mcp(view.image)[0]]


# ---------------------------------------------------------------- second opinion (VLMs)

GROUND_PROMPT = {
    "qwen": ('\nReturn ONLY a JSON list, one entry per instance: [{"label": "...", "bbox_2d": [x1, y1, x2, y2]}] '
             "with coordinates normalised to 0-1000 of the image width/height."),
    "gemini": ('\nReturn ONLY a JSON list: [{"label": "...", "box_2d": [ymin, xmin, ymax, xmax]}] '
               "with coordinates normalised to 0-1000."),
}


def _backend(backend: str) -> str:
    from .backends import vlm

    b = (backend or "auto").lower()
    if b != "auto":
        return b
    if vlm.ollama_models() is not None:
        return "ollama"
    if vlm.cloud_allowed():
        return "gemini"
    raise RuntimeError("No second-opinion model available: start Ollama with a vision model (e.g. "
                       "`ollama pull qwen3.5:9b`), or opt in to Gemini with CLAUDEYE_ALLOW_CLOUD=1 + GEMINI_API_KEY.")


def ask_vision(question: str, refs_: list[str], region: str | None = None, backend: str = "auto",
               model: str | None = None, ground: str | None = None, max_edge: int | None = None) -> list:
    from . import image_tools, refs, render
    from .backends import vlm
    from .encode import to_mcp

    if not refs_:
        raise ValueError("give at least one image ref")
    imgs, boxes, names = [], [], []
    for rf in refs_[:8]:
        img, r = refs.load(rf)
        W, H = img.size
        box = render.parse_region(region, W, H, image_tools._grid_for(r, W, H, None))
        imgs.append(img.crop(box))
        boxes.append(box)
        names.append(str(r))
    b = _backend(backend)
    prompt = question + (f"\nTarget to localise: {ground}." + GROUND_PROMPT["gemini" if b == "gemini" else "qwen"]
                         if ground else "")
    if b == "ollama":
        answer, used = vlm.ollama_chat(prompt, imgs, model=model, json_mode=False)
    elif b == "gemini":
        answer, used = vlm.gemini_images(prompt, imgs, model=model)
    else:
        raise ValueError("backend must be auto | ollama | gemini")
    head = {"backend": b, "model": used, "images": names, "region": list(boxes[0]) if region else None}
    out: list = [_j(head) + "\nANSWER (another model's opinion - verify before relying on it):\n" + answer]
    if ground:
        x0, y0, x1, y1 = boxes[0]
        found = vlm.parse_boxes(answer, x1 - x0, y1 - y0, "gemini" if b == "gemini" else "qwen")
        for f in found:
            f["box"] = [f["box"][0] + x0, f["box"][1] + y0, f["box"][2] + x0, f["box"][3] + y0]
        img0, _ = refs.load(refs_[0])
        view = render.render_view(img0, boxes[0], long_edge=image_tools._edge(max_edge), min_edge=768)
        render.draw_marks(view, [{"id": f["id"], "box": f["box"], "label": f"{f['id']} {f['label']}"[:24]}
                                 for f in found])
        out[0] += "\nPARSED BOXES (original px):\n" + "\n".join(_j(f) for f in found)
        out.append(to_mcp(view.image)[0])
    return out


def ask_video(path: str, question: str, start: float | str | None = None, end: float | str | None = None,
              backend: str = "auto", model: str | None = None, frames: int = 12, fps: float | None = None,
              resolution: str = "default", transcript: bool = True, progress=None) -> list:
    from . import video, video_tools
    from .backends import vlm
    from .refs import fmt_time, parse_ref

    r = parse_ref(path)
    vi = video.info(r.path)
    s, e = video_tools._window(vi, start, end)
    b = _backend(backend)
    head = {"backend": b, "video": r.path.as_posix(), "window": [round(s, 2), round(e, 2)]}
    if b == "gemini":
        q = (question + "\nRefer to moments with absolute MM:SS timestamps of the full video."
             + (f" (The clip analysed starts at {fmt_time(s)}.)" if s > 0 else ""))
        answer, used = vlm.gemini_video(r.path, q, start=s if s > 0 else None, end=e if e < vi.duration else None,
                                        fps=fps, resolution=resolution, model=model)
        head["model"] = used
    elif b == "ollama":
        # one timestamped storyboard sheet (works even for single-image models) + optional transcript
        sheet, frames_out = video_tools.video_storyboard(str(r.path), start=s, end=e, frames=frames,
                                                         progress=progress, as_pil=True)
        ctx = ""
        if transcript and vi.audio:
            try:
                segs = transcribe(str(r.path), start=s, end=e)[0]
                ctx = "\nSpeech transcript (timestamps MM:SS):\n" + "\n".join(
                    ln for ln in segs.split("\n") if ln.startswith("["))
            except Exception:
                ctx = ""
        stamps = ", ".join(f"#{f['i']}={f['at']}" for f in frames_out)
        q = (f"The image is a grid of {len(frames_out)} frames from a video, in time order left-to-right, "
             f"top-to-bottom; each cell is labelled '#index MM:SS'. Frame times: {stamps}.{ctx}\n"
             f"Question: {question}\nCite frame numbers / timestamps as evidence.")
        answer, used = vlm.ollama_chat(q, [sheet], model=model, long_edge=1568)
        head["model"] = used
        head["frames_sent"] = len(frames_out)
    else:
        raise ValueError("backend must be auto | ollama | gemini")
    return [_j(head) + "\nANSWER (another model's opinion - verify cited moments with video_frames/zoom):\n" + answer]


# ---------------------------------------------------------------- video search

def video_search(path: str, query: str, top_k: int = 8, min_gap: float | None = None, every: float | None = None,
                 max_edge: int | None = None, progress=None) -> list:
    """Text -> moments: rank sampled frames by SigLIP 2 similarity, keep temporally separated peaks."""
    import numpy as np

    from . import video, video_tools
    from .backends import embed
    from .refs import fmt_time, parse_ref

    r = parse_ref(path)
    vi = video.info(r.path)
    times, emb = embed.index(r.path, every=every, progress=progress)
    queries = [q.strip() for q in query.split("|") if q.strip()]
    q, scale, bias = embed.query(queries)
    sims = emb.astype(np.float32) @ q.T  # (N, Q)
    sim = sims.max(axis=1)
    gap = min_gap if min_gap is not None else max(2.0, vi.duration / 60)
    order = np.argsort(-sim)
    picks: list[int] = []
    for i in order:
        if all(abs(times[i] - times[j]) >= gap for j in picks):
            picks.append(int(i))
        if len(picks) >= top_k:
            break
    hits = [{"rank": k + 1, "t": round(float(times[i]), 2), "at": fmt_time(float(times[i])),
             "score": round(float(1 / (1 + np.exp(-(sim[i] * scale + bias)))), 3), "cos": round(float(sim[i]), 3),
             "ref": f"{r.path.as_posix()}@{float(times[i]):.3f}"} for k, i in enumerate(picks)]
    head = {"video": r.path.as_posix(), "query": queries, "indexed_frames": len(times),
            "every_s": round(float(times[1] - times[0]), 2) if len(times) > 1 else None, "model": embed.model_id()}
    out: list = [_j(head) + "\nHITS (best first; similarity is relative - check the frames):\n" +
                 "\n".join(_j(h) for h in hits) +
                 "\nNarrow down with video_frames(path, start=hit-2, end=hit+2, count=8). English queries work best."]
    if hits:
        chrono = sorted(hits, key=lambda h: h["t"])
        labels = [f"#{h['rank']} {h['at']} s={h['score']:.2f}" for h in chrono]
        (mimg, _), _ = video_tools._sheet(str(r.path), vi, [h["t"] for h in chrono], labels, cols=None,
                                          header=f"{r.path.name} | search: {query}", max_edge=max_edge)
        out.append(mimg)
    return out


# ---------------------------------------------------------------- status

def vision_status() -> list:
    import importlib.util
    import platform
    import sys

    from . import __version__
    from .backends import INSTALL_HINTS
    from .config import settings

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    s = settings()
    st: dict[str, Any] = {"claudeye": __version__, "python": sys.version.split()[0], "os": platform.platform(),
                          "cache_dir": str(s.cache_dir), "max_edge": s.max_edge, "hard_max_edge": s.hard_max_edge}
    comps: dict[str, Any] = {
        "core (look/zoom/scan/pixels/blobs/annotate/compare/locate/video_*)": "ok",
        "ocr (RapidOCR)": "ok" if has("rapidocr") and has("onnxruntime") else f"missing: {INSTALL_HINTS['ocr']}",
        "transcribe (faster-whisper)": "ok" if has("faster_whisper") else f"missing: {INSTALL_HINTS['asr']}",
        "detect/segment/depth/video_search (torch+transformers)":
            "ok" if has("torch") and has("transformers") else f"missing: {INSTALL_HINTS['ml']}",
        "gemini": ("enabled" if _cloud_ok() else "disabled (set CLAUDEYE_ALLOW_CLOUD=1)")
        + ("" if has("google.genai") else f"; missing: {INSTALL_HINTS['gemini']}")
        + ("" if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") else "; GEMINI_API_KEY not set"),
    }
    if has("torch"):
        from .backends import device as dev

        comps["torch_device"] = dev.device()
        comps["vram"] = dev.vram()
        comps["models_loaded"] = dev.loaded()
        from .backends import detect as d_, embed as e_, segment as s_

        comps["models"] = {"detect": d_.model_id(), "sam2": s_.sam2_id(), "sam3": s_.sam3_id(),
                           "embed": e_.model_id()}
    if has("faster_whisper"):
        from .backends import asr

        try:
            comps["whisper"] = dict(zip(("device", "compute", "model"), asr.device()))
        except Exception as e:  # noqa: BLE001
            comps["whisper"] = f"error: {e}"
    from .backends import vlm

    names = vlm.ollama_models()
    comps["ollama"] = ("not reachable at " + s.ollama_host) if names is None else {
        "vision_models": vlm.ollama_vision_models(), "configured": s.ollama_model}
    st["components"] = comps
    return [_j(st)]


# ---------------------------------------------------------------- registration

def register(mcp, RO, progress) -> None:
    @mcp.tool(name="ocr", annotations=RO, structured_output=False)
    def ocr_tool(ref: Ref, region: OptRegion = None,
                 lang: Annotated[str, Field(description="korean (also reads English/digits), en, ch, japan, "
                                                        "latin.")] = "korean",
                 query: Annotated[str | None, Field(description="Find text: return only lines matching this "
                                                                "(fuzzy), best first, with exact boxes.")] = None,
                 min_score: Annotated[float, Field(description="Drop lines below this confidence.")] = 0.5,
                 rotated: Annotated[bool, Field(description="Enable 180-degree text detection (photos of "
                                                            "documents). Keep False for screenshots.")] = False,
                 upscale: Annotated[float | None, Field(description="Pre-scale factor; default auto (2x for "
                                                                    "small regions).")] = None,
                 mark: Annotated[bool, Field(description="Return the image with numbered line boxes.")] = True,
                 max_lines: int = 300, max_edge: MaxEdge = None) -> list:
        """Read text with exact line boxes (Korean + English). Returns reading-order text, per-line
        {id,text,score,box} and a numbered overlay to cross-check. Use `query` to find where a label/button/
        message is. For tiny text pass a region; low-score lines should be confirmed with zoom."""
        return ocr(ref, region=region, lang=lang, query=query, min_score=min_score, rotated=rotated,
                   upscale=upscale, mark=mark, max_lines=max_lines, max_edge=max_edge)

    @mcp.tool(name="transcribe", annotations=RO, structured_output=False)
    def transcribe_tool(path: Annotated[str, Field(description="Video or audio file.")],
                        start: OptTime = None, end: OptTime = None,
                        language: Annotated[str, Field(description="auto, ko, en, ja, ...")] = "auto",
                        words: Annotated[bool, Field(description="Add word-level timestamps.")] = False,
                        model: Annotated[str | None, Field(description="Whisper model override (tiny/base/small/"
                                                                       "medium/large-v3/large-v3-turbo).")] = None,
                        prompt: Annotated[str | None, Field(description="Vocabulary hint: names, jargon, "
                                                                        "product terms.")] = None,
                        ctx: Context | None = None) -> list:
        """Speech-to-text with timestamps (Whisper; Korean/English/99 languages; VAD skips silence). Pair with
        video_storyboard/video_frames: what is said + what is shown gives far better video understanding."""
        return transcribe(path, start=start, end=end, language=language, words=words, model=model,
                          prompt=prompt, progress=progress(ctx))

    @mcp.tool(name="detect", annotations=RO, structured_output=False)
    def detect_tool(ref: Ref,
                    prompt: Annotated[str, Field(description="What to find, comma-separated short English noun "
                                                             "phrases: 'person, car, traffic light'.")],
                    region: OptRegion = None,
                    threshold: Annotated[float, Field(description="Box confidence (0.15 finds more, 0.4 "
                                                                  "stricter).")] = 0.3,
                    text_threshold: float = 0.25,
                    tile: Annotated[str, Field(description="auto | on (slice into tiles: small objects in big "
                                                           "images) | off.")] = "auto",
                    max_results: int = 200, mark: bool = True, max_edge: MaxEdge = None) -> list:
        """Open-vocabulary object detection (text -> boxes) with counts per label and a numbered overlay
        (LLMDet / Grounding-DINO). Use for counting and locating natural objects; for flat-coloured marks,
        blobs() is exact. Always sanity-check the overlay."""
        return detect(ref, prompt, region=region, threshold=threshold, text_threshold=text_threshold, tile=tile,
                      max_results=max_results, mark=mark, max_edge=max_edge)

    @mcp.tool(name="segment", annotations=RO, structured_output=False)
    def segment_tool(ref: Ref,
                     prompt: Annotated[str | None, Field(description="Segment every instance of these "
                                                                     "(comma-separated English phrases).")] = None,
                     points: Annotated[str | None, Field(description="Or click points for ONE object: "
                                                                     "'x,y; x,y' (add ',0' for background).")] = None,
                     boxes: Annotated[str | None, Field(description="Or boxes, one object each: "
                                                                    "'x0,y0,x1,y1; ...'.")] = None,
                     region: OptRegion = None,
                     threshold: Annotated[float, Field(description="Instance confidence for prompt mode.")] = 0.35,
                     max_results: int = 50, max_edge: MaxEdge = None) -> list:
        """Pixel-accurate masks (SAM 2.1 / SAM 3): polygons, areas (px and % of image), boxes, overlay. Use for
        exact shapes, areas/proportions, separating overlapping objects, or precise outlines."""
        return segment(ref, prompt=prompt, points=points, boxes=boxes, region=region, threshold=threshold,
                       max_results=max_results, max_edge=max_edge)

    @mcp.tool(name="depth", annotations=RO, structured_output=False)
    def depth_tool(ref: Ref, region: OptRegion = None,
                   points: Annotated[str | None, Field(description="Report depth at 'x,y; x,y'.")] = None,
                   max_edge: MaxEdge = None) -> list:
        """Relative depth map (Depth Anything V2): which things are nearer/farther, per-cell depth table, depth
        at points. For spatial/3D reasoning about photos (not metric distances)."""
        return depth(ref, region=region, points=points, max_edge=max_edge)

    @mcp.tool(name="ask_vision", annotations=RO, structured_output=False)
    def ask_vision_tool(question: Annotated[str, Field(description="What to ask the other vision model.")],
                        refs: Annotated[list[str], Field(description="Image refs (paths or 'video.mp4@SECONDS'), "
                                                                     "max 8.")],
                        region: OptRegion = None,
                        backend: Annotated[str, Field(description="auto (local Ollama first) | ollama | gemini "
                                                                  "(cloud, needs user opt-in).")] = "auto",
                        model: Annotated[str | None, Field(description="Override model name.")] = None,
                        ground: Annotated[str | None, Field(description="Also ask it to box this target; boxes "
                                                                        "are drawn for you to verify.")] = None,
                        max_edge: MaxEdge = None) -> list:
        """Second opinion from a different vision model (local Qwen/Llama-vision via Ollama, or Gemini).
        Useful when unsure, for cross-checking a reading, or for grounding a described UI element. Its answer
        can be wrong too - treat agreement as evidence, disagreement as a cue to zoom/measure."""
        return ask_vision(question, refs, region=region, backend=backend, model=model, ground=ground,
                          max_edge=max_edge)

    @mcp.tool(name="ask_video", annotations=RO, structured_output=False)
    def ask_video_tool(path: Annotated[str, Field(description="Video file path.")],
                       question: Annotated[str, Field(description="Question about the video.")],
                       start: OptTime = None, end: OptTime = None,
                       backend: Annotated[str, Field(description="auto | ollama (local: timestamped storyboard + "
                                                                 "transcript) | gemini (cloud: native video+audio, "
                                                                 "needs user opt-in).")] = "auto",
                       model: str | None = None,
                       frames: Annotated[int, Field(description="Frames sent to a local model.")] = 12,
                       fps: Annotated[float | None, Field(description="Gemini sampling rate (default 1).")] = None,
                       resolution: Annotated[str, Field(description="Gemini: default | low | high (read "
                                                                    "on-screen text).")] = "default",
                       transcript: bool = True, ctx: Context | None = None) -> list:
        """Ask another model about a whole video or a time window. Gemini watches the actual video with audio;
        the local path sends a timestamped frame grid plus the speech transcript. Verify cited timestamps."""
        return ask_video(path, question, start=start, end=end, backend=backend, model=model, frames=frames, fps=fps,
                         resolution=resolution, transcript=transcript, progress=progress(ctx))

    @mcp.tool(name="video_search", annotations=RO, structured_output=False)
    def video_search_tool(path: Annotated[str, Field(description="Video file path.")],
                          query: Annotated[str, Field(description="What to find, in English; alternatives "
                                                                  "separated by '|'.")],
                          top_k: Annotated[int, Field(description="Moments to return.")] = 8,
                          min_gap: Annotated[float | None, Field(description="Min seconds between hits.")] = None,
                          every: Annotated[float | None, Field(description="Index sampling interval (s).")] = None,
                          max_edge: MaxEdge = None, ctx: Context | None = None) -> list:
        """Find WHEN something appears in a long video by text description (SigLIP 2 frame embeddings, cached
        per video) - returns ranked timestamps + a sheet of the hits. Then drill in with video_frames/zoom."""
        return video_search(path, query, top_k=top_k, min_gap=min_gap, every=every, max_edge=max_edge,
                            progress=progress(ctx))

    @mcp.tool(name="vision_status", annotations=RO, structured_output=False)
    def vision_status_tool() -> list:
        """Which ClaudeYe backends are installed/enabled (OCR, speech, detection, Ollama, Gemini), device, VRAM,
        loaded models, and install hints for missing parts."""
        return vision_status()
