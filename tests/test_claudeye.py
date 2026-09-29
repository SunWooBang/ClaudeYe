"""Regression tests against the synthetic fixtures (python scripts/make_samples.py first).

Fast by default; model-backed tests run only when their backend is installed
(set CLAUDEYE_TEST_HEAVY=1 to include the slow torch ones)."""

from __future__ import annotations

import importlib.util
import json
import os
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "samples"
DASH, VID, SPEECH = str(S / "dashboard_4k.png"), str(S / "demo.mp4"), str(S / "demo_speech.mp4")
HEAVY = os.environ.get("CLAUDEYE_TEST_HEAVY") == "1"

pytestmark = pytest.mark.skipif(not (S / "dashboard_4k.png").exists(), reason="run scripts/make_samples.py")


def head(res) -> dict:
    return json.loads(res[0].split("\n", 1)[0])


# ---------------------------------------------------------------- parsing

def test_regions_and_times():
    from claudeye.refs import fmt_time, parse_ref, parse_time
    from claudeye.render import Grid, parse_region

    g = Grid.auto(3840, 2160)
    assert (g.cols, g.rows) == (8, 5)
    assert parse_region("A1", 3840, 2160, g) == (0, 0, 480, 432)
    assert parse_region("B2:C3", 3840, 2160, g) == (480, 432, 1440, 1296)
    assert parse_region("0.5,0.5,1,1", 100, 100) == (50, 50, 100, 100)
    assert parse_region("10,20,30,40", 100, 100) == (10, 20, 30, 40)
    assert parse_region("500,500", 1000, 1000, point_radius=100) == (400, 400, 600, 600)
    assert parse_region("50,50", 1000, 1000, point_radius=100) == (0, 0, 150, 150)  # clamped at the edge
    with pytest.raises(ValueError):
        parse_region("Z9", 3840, 2160, g)
    assert parse_time("1:02:03.5") == 3723.5 and parse_time("90s") == 90 and parse_time("01:02") == 62
    assert fmt_time(62.5) == "01:02.50"
    r = parse_ref(f"{VID}@00:12.5")
    assert r.time == 12.5 and r.is_video


# ---------------------------------------------------------------- core image tools

def test_look_zoom_budget():
    from claudeye import image_tools as it

    h = head(it.look(DASH))
    assert h["size"] == "3840x2160" and h["image"]["tokens"] < 2500
    z = head(it.zoom(DASH, "60,180,560,420"))
    assert z["region"] == [60, 180, 560, 420] and z["magnification"] > 1.5


def test_blobs_exact_counts():
    from claudeye import image_tools as it

    truth = json.loads((S / "truth.json").read_text(encoding="utf-8"))["counts"]
    panel = ",".join(map(str, truth["panel"]))
    assert head(it.blobs(DASH, color="red", region=panel))["count"] == truth["red_dots"]
    assert head(it.blobs(DASH, color="#2850dc", region=panel))["count"] == truth["blue_squares"]


def test_locate_and_compare():
    from claudeye import image_tools as it

    loc = json.loads(it.locate(DASH, str(S / "star_icon.png"))[0].split("\n", 1)[0])
    assert loc["found"] >= 1 and loc["matches"][0]["box"][:2] == [3312, 1733]
    cmp_ = head(it.compare(DASH, str(S / "dashboard_4k_v2.png")))
    assert not cmp_["identical"] and len(cmp_["regions"]) == 2


# ---------------------------------------------------------------- video

def test_video_scenes_and_changes():
    from claudeye import video_tools as vt

    sb = head(vt.video_storyboard(VID, frames=12))
    assert [round(b) for a, b in sb["scenes"][:-1]] == [5, 10, 18]
    assert len(sb["frames"]) == 12
    ch = head(vt.video_changes(VID, start=10, end=18))
    starts = [e["settled"] for e in ch["events"] if e["kind"] != "cut"]
    assert any(10.9 < t < 11.2 for t in starts) and any(15.9 < t < 16.2 for t in starts)


def test_frame_refs_are_exact():
    from claudeye import video

    (t, im), = video.sample(VID, [16.5])
    assert abs(t - 16.5) < 0.04 and im.size == (1280, 720)


# ---------------------------------------------------------------- model backends

@pytest.mark.skipif(not importlib.util.find_spec("rapidocr"), reason="ocr extra not installed")
def test_ocr_find_text():
    from claudeye import models_tools as m

    res = m.ocr(f"{VID}@16.5", query="오류")[0]
    line = json.loads(res.split("LINES:\n", 1)[1].split("\n", 1)[0])
    assert "E-503" in line["text"] and line["box"][0] > 800


@pytest.mark.skipif(not importlib.util.find_spec("faster_whisper") or not (S / "demo_speech.mp4").exists(),
                    reason="asr extra / speech sample missing")
def test_transcribe_korean_timing():
    from claudeye import models_tools as m

    text = m.transcribe(SPEECH, language="ko")[0]
    assert "서버 주소" in text and "[00:11." in text


@pytest.mark.skipif(not (HEAVY and importlib.util.find_spec("transformers")), reason="set CLAUDEYE_TEST_HEAVY=1")
def test_segment_button_click():
    from claudeye import models_tools as m

    res = m.segment(f"{VID}@16.5", points="1025,500")[0]
    box = json.loads(res.split("MASKS:\n", 1)[1].split("\n", 1)[0])["box"]
    assert abs(box[0] - 900) < 6 and abs(box[2] - 1150) < 6


# ---------------------------------------------------------------- cloud path, offline (no key, no network)

class _FakeFiles:
    def __init__(self):
        self.uploaded = []

    def upload(self, file):
        self.uploaded.append(file)
        return types.SimpleNamespace(name="files/x", uri="https://fake/x", mime_type="video/mp4",
                                     state=types.SimpleNamespace(name="ACTIVE"), error=None)

    def get(self, name):
        return self.upload(name)


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        return types.SimpleNamespace(text='[{"label": "star", "box_2d": [100, 200, 300, 400]}]')


@pytest.mark.skipif(not importlib.util.find_spec("google.genai"), reason="gemini extra not installed")
def test_gemini_requests_offline(monkeypatch, tmp_path):
    from google import genai
    from google.genai import types as gt

    from claudeye.backends import vlm
    from claudeye.config import settings

    fake = types.SimpleNamespace(files=_FakeFiles(), models=_FakeModels())
    monkeypatch.setattr(genai, "Client", lambda *a, **k: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("CLAUDEYE_ALLOW_CLOUD", "1")
    monkeypatch.setattr(type(settings()), "cache_dir", property(lambda self: tmp_path), raising=False)
    from PIL import Image

    txt, model = vlm.gemini_images("find the star", [Image.new("RGB", (64, 64))])
    parts = fake.models.calls[-1][1]
    assert isinstance(parts[0], gt.Part) and parts[0].inline_data.mime_type == "image/png" and parts[-1] == "find the star"
    boxes = vlm.parse_boxes(txt, 1000, 500, "gemini")
    assert boxes[0]["box"] == [200, 50, 400, 150]  # [ymin,xmin,ymax,xmax]/1000 -> px

    vlm.gemini_video(Path(VID), "what happens?", start=10, end=18, fps=2, resolution="high")
    part = fake.models.calls[-1][1][0]
    assert part.file_data.file_uri == "https://fake/x"
    assert part.video_metadata.fps == 2 and part.video_metadata.start_offset == "10.00s"
    assert fake.models.calls[-1][2].media_resolution == gt.MediaResolution.MEDIA_RESOLUTION_HIGH


def test_cloud_is_opt_in(monkeypatch):
    from claudeye.backends import vlm

    monkeypatch.delenv("CLAUDEYE_ALLOW_CLOUD", raising=False)
    with pytest.raises(PermissionError):
        vlm._gemini()
