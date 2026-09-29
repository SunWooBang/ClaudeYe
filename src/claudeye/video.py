"""Video decoding on PyAV (bundled FFmpeg; no system ffmpeg needed).

All times are seconds relative to the first video frame (stream start offset removed).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image

SEEK_GAP = 2.5  # seconds: larger forward jumps seek instead of decoding through


@dataclass(frozen=True)
class VideoInfo:
    path: str
    duration: float
    fps: float
    width: int  # display size (rotation applied)
    height: int
    frames: int | None
    codec: str
    rotation: int
    bitrate_kbps: int | None
    container: str
    size_mb: float
    audio: dict | None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _open(path: str | Path):
    c = av.open(str(path), metadata_errors="ignore")
    if not c.streams.video:
        c.close()
        raise ValueError(f"no video stream in {path}")
    vs = c.streams.video[0]
    vs.thread_type = "AUTO"
    return c, vs


def _start_sec(vs) -> float:
    return float(vs.start_time * vs.time_base) if vs.start_time is not None else 0.0


def _frame_time(fr, vs, start: float) -> float | None:
    if fr.pts is None:
        return None
    return float(fr.pts * vs.time_base) - start


def info(path: str | Path) -> VideoInfo:
    p = Path(path)
    st = p.stat()
    return _info_cached(str(p), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=64)
def _info_cached(path: str, mtime_ns: int, size: int) -> VideoInfo:
    c, vs = _open(path)
    try:
        start = _start_sec(vs)
        if vs.duration:
            dur = float(vs.duration * vs.time_base)
        elif c.duration:
            dur = c.duration / 1_000_000
        else:
            dur = 0.0
        fps = float(vs.average_rate or vs.guessed_rate or 0) or 0.0
        rotation = 0
        for fr in c.decode(vs):
            rotation = int(getattr(fr, "rotation", 0) or 0)
            break
        if dur <= 0:  # e.g. some webm/streams: demux (no decode) to find the last timestamp
            c.seek(0)
            last = 0
            for pkt in c.demux(vs):
                if pkt.pts is not None:
                    last = max(last, pkt.pts + (pkt.duration or 0))
            dur = max(0.0, float(last * vs.time_base) - start)
        w, h = vs.codec_context.width, vs.codec_context.height
        if rotation % 180:
            w, h = h, w
        audio = None
        if c.streams.audio:
            a = c.streams.audio[0]
            audio = {"codec": a.codec_context.name, "sample_rate": a.codec_context.sample_rate,
                     "channels": a.codec_context.channels}
        frames = vs.frames or (round(dur * fps) if dur and fps else None)
        br = c.bit_rate or vs.bit_rate
        return VideoInfo(
            path=Path(path).as_posix(), duration=round(dur, 3), fps=round(fps, 3), width=w, height=h,
            frames=frames, codec=vs.codec_context.name, rotation=rotation,
            bitrate_kbps=round(br / 1000) if br else None, container=c.format.name,
            size_mb=round(size / 1_048_576, 2), audio=audio,
        )
    finally:
        c.close()


def _to_pil(fr, long_edge: int | None) -> Image.Image:
    rot = int(getattr(fr, "rotation", 0) or 0)
    w, h = fr.width, fr.height
    if long_edge and max(w, h) > long_edge:
        s = long_edge / max(w, h)
        img = fr.to_image(width=max(2, round(w * s)), height=max(2, round(h * s)), interpolation="AREA")
    else:
        img = fr.to_image()
    if rot % 360:
        img = img.rotate(rot, expand=True)
    return img


def sample(path: str | Path, times: list[float], *, long_edge: int | None = None) -> list[tuple[float, Image.Image]]:
    """Decode the first frame at/after each requested time. Returns (actual_time, image) in input order."""
    if not times:
        return []
    order = sorted(range(len(times)), key=lambda i: times[i])
    out: list[tuple[float, Image.Image] | None] = [None] * len(times)
    c, vs = _open(path)
    try:
        start = _start_sec(vs)
        fps = float(vs.average_rate or vs.guessed_rate or 30)
        half = 0.5 / fps
        it = None
        held = None  # (t, frame): most recent decoded frame
        for i in order:
            target = max(0.0, float(times[i]))
            if held is not None and held[0] >= target - half:
                out[i] = (held[0], _to_pil(held[1], long_edge))
                continue
            if it is None or held is None or target - held[0] > SEEK_GAP:
                c.seek(int((target + start) / vs.time_base), stream=vs, backward=True, any_frame=False)
                it = c.decode(vs)
            got = None
            for fr in it:
                t = _frame_time(fr, vs, start)
                if t is None:
                    continue
                held = (t, fr)
                if t >= target - half:
                    got = held
                    break
            if got is None:  # past the end: use the last frame
                it = None
                got = held
            if got is None:
                raise ValueError(f"could not decode a frame near {target:.3f}s")
            out[i] = (got[0], _to_pil(got[1], long_edge))
        return out  # type: ignore[return-value]
    finally:
        c.close()


def frame_at(path: str | Path, t: float, long_edge: int | None = None) -> Image.Image:
    return sample(path, [t], long_edge=long_edge)[0][1]


def iter_small(path: str | Path, *, width: int = 160, start: float = 0.0, end: float | None = None,
               every: float | None = None, keyframes_only: bool = False,
               gray: bool = True) -> Iterator[tuple[float, np.ndarray]]:
    """Fast low-res frame stream for analysis: yields (t, HxW[x3] uint8), rotation applied."""
    c, vs = _open(path)
    try:
        if keyframes_only:
            vs.codec_context.skip_frame = "NONKEY"
        st = _start_sec(vs)
        cw, ch = vs.codec_context.width, vs.codec_context.height
        height = max(2, round(width * ch / cw / 2) * 2)
        if start > 0:
            c.seek(int((start + st) / vs.time_base), stream=vs, backward=True, any_frame=False)
        next_t = start
        fmt = "gray" if gray else "rgb24"
        for fr in c.decode(vs):
            t = _frame_time(fr, vs, st)
            if t is None or t < start - 1e-6:
                continue
            if end is not None and t > end:
                break
            if every and t < next_t - 1e-6:
                continue
            arr = fr.reformat(width=width, height=height, format=fmt, interpolation="AREA").to_ndarray()
            rot = int(getattr(fr, "rotation", 0) or 0)
            if rot % 360:
                arr = np.ascontiguousarray(np.rot90(arr, k=(rot // 90) % 4))
            yield t, arr
            if every:
                next_t = t + every
    finally:
        c.close()
