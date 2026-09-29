"""Speech-to-text with faster-whisper (CTranslate2). GPU: large-v3-turbo fp16; CPU: small int8.
Transcripts are cached per file/window/options."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

from ..config import settings
from . import MissingBackend

SR = 16000
_models: dict[tuple, object] = {}
_lock = threading.Lock()


def _add_cuda_dll_dirs() -> None:
    """CTranslate2 needs CUDA 12 cuBLAS/cuDNN DLLs; pip wheels (nvidia-*-cu12) put them in site-packages."""
    if sys.platform != "win32":
        return
    base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        d = base / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def device() -> tuple[str, str, str]:
    """(device, compute_type, default model)"""
    try:
        import ctranslate2
    except ImportError as e:
        raise MissingBackend("faster-whisper", "asr") from e
    _add_cuda_dll_dirs()
    want = settings().device
    if want != "cpu":
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "int8_float16", "large-v3-turbo"
        except Exception:
            pass
    return "cpu", "int8", "small"


def _model(name: str | None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise MissingBackend("faster-whisper", "asr") from e
    dev, ctype, default = device()
    cfg = settings().whisper_model
    name = name or (default if cfg == "auto" else cfg)
    key = (name, dev, ctype)
    with _lock:
        if key not in _models:
            try:
                _models[key] = WhisperModel(name, device=dev, compute_type=ctype)
            except Exception:
                if dev != "cuda":
                    raise
                # CUDA libraries missing/mismatched: fall back to CPU rather than failing the call
                key = (name, "cpu", "int8")
                _models[key] = WhisperModel(name, device="cpu", compute_type="int8")
        return _models[key], key


def transcribe(path: Path, *, start: float = 0.0, end: float | None = None, language: str | None = None,
               words: bool = False, model: str | None = None, prompt: str | None = None,
               progress=None) -> dict:
    st = path.stat()
    raw = json.dumps([str(path.resolve()), st.st_size, st.st_mtime_ns, start, end, language, model, prompt, 2])
    f = settings().cache_dir / "transcripts" / (hashlib.sha1(raw.encode()).hexdigest()[:20] + ".json")
    if f.exists():
        res = json.loads(f.read_text(encoding="utf-8"))
        return res if words else _strip_words(res)

    from faster_whisper import decode_audio

    audio = decode_audio(str(path), sampling_rate=SR)
    a = int(start * SR)
    b = len(audio) if end is None else min(len(audio), int(end * SR))
    audio = audio[a:b]
    dur = len(audio) / SR
    m, key = _model(model)
    lang = None if language in (None, "", "auto") else language

    def run(mdl):
        # word_timestamps always on: segment bounds from Whisper alone can swallow long silences;
        # the word-level (DTW) alignment gives accurate start/end. The generator is consumed here so
        # CUDA/DLL problems surface now (and can fall back to CPU).
        segs, info = mdl.transcribe(audio, language=lang, vad_filter=True, word_timestamps=True, beam_size=5,
                                    initial_prompt=prompt, condition_on_previous_text=False)
        return list(segs), info

    try:
        segs, info = run(m)
    except (RuntimeError, OSError) as e:
        if key[1] != "cuda":
            raise
        from faster_whisper import WhisperModel

        key = (key[0], "cpu", "int8")
        with _lock:
            _models[key] = WhisperModel(key[0], device="cpu", compute_type="int8")
        segs, info = run(_models[key])
        info_note = f"CUDA failed ({str(e)[:120]}); ran on CPU"
    else:
        info_note = None
    out_segs = []
    for s in segs:
        ws = [[round(w.start + start, 2), round(w.end + start, 2), w.word.strip()] for w in (s.words or [])]
        seg = {"start": ws[0][0] if ws else round(s.start + start, 2),
               "end": ws[-1][1] if ws else round(s.end + start, 2), "text": s.text.strip(), "words": ws}
        out_segs.append(seg)
        if progress and dur:
            progress(min(0.99, s.end / dur), f"transcribing {s.end:.0f}/{dur:.0f}s")
    res = {"model": key[0], "device": key[1], "language": info.language,
           "language_prob": round(info.language_probability, 3), "audio_seconds": round(dur, 2),
           "segments": out_segs}
    if info_note:
        res["note"] = info_note
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return res if words else _strip_words(res)


def _strip_words(res: dict) -> dict:
    return {**res, "segments": [{k: v for k, v in s.items() if k != "words"} for s in res["segments"]]}
