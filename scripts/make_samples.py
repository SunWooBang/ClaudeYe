"""Generate synthetic test fixtures with known ground truth.

samples/dashboard_4k.png   3840x2160, tiny KO/EN text, counting targets, a hidden icon
samples/star_icon.png      the icon (template for locate())
samples/demo.mp4           24s, 4 shots + a screen-recording-like segment with UI changes
samples/truth.json         ground truth for all of the above
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "samples"


def font(size: int, bold: bool = False):
    for n in (["malgunbd.ttf"] if bold else []) + ["malgun.ttf", "arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def star(size: int, color=(20, 170, 60)) -> Image.Image:
    im = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(im)
    c, r1, r2 = size / 2, size * 0.48, size * 0.2
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        r = r1 if k % 2 == 0 else r2
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    d.polygon(pts, fill=color)
    return im


def make_image(truth: dict) -> None:
    rng = random.Random(7)
    W, H = 3840, 2160
    im = Image.new("RGB", (W, H), (248, 248, 250))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 120], fill=(30, 40, 60))
    d.text((40, 60), "ClaudeYe Test Dashboard", font=font(56, True), anchor="lm", fill=(255, 255, 255))

    texts = [
        (80, 200, 9, "서버 상태: 정상 (응답시간 123ms)"),
        (80, 240, 10, "Error code: E-4071-ZX / build 2026.09.29"),
        (80, 280, 12, "사용자 수: 1,284명 · 활성 세션 77"),
        (80, 330, 16, "마지막 배포: 2026-09-28 17:42 KST"),
        (80, 390, 24, "Throughput 98.6 req/s"),
        (2900, 2080, 8, "tiny footer: license key ABCD-1234-XYZ"),
    ]
    truth["texts"] = []
    for x, y, size, t in texts:
        d.text((x, y), t, font=font(size), fill=(20, 20, 20))
        truth["texts"].append({"text": t, "size_px": size, "xy": [x, y]})

    # table with small numbers
    tx, ty, cw, rh = 1400, 200, 190, 30
    headers = ["지역", "요청", "오류", "지연(ms)"]
    rows = [["서울", "12,408", "31", "84"], ["부산", "7,992", "12", "91"], ["대구", "5,127", "44", "133"],
            ["광주", "3,381", "7", "77"], ["제주", "1,090", "0", "65"]]
    for j, h in enumerate(headers):
        d.rectangle([tx + j * cw, ty, tx + (j + 1) * cw, ty + rh], outline=(150, 150, 160), fill=(225, 230, 240))
        d.text((tx + j * cw + 8, ty + rh / 2), h, font=font(11, True), anchor="lm", fill=(10, 10, 10))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            y0 = ty + (i + 1) * rh
            d.rectangle([tx + j * cw, y0, tx + (j + 1) * cw, y0 + rh], outline=(190, 190, 200))
            d.text((tx + j * cw + 8, y0 + rh / 2), v, font=font(11), anchor="lm", fill=(10, 10, 10))
    truth["table"] = {"box": [tx, ty, tx + 4 * cw, ty + 6 * rh], "headers": headers, "rows": rows}

    # counting targets: red dots and blue squares in a panel
    px0, py0, px1, py1 = 200, 700, 1800, 1900
    d.rectangle([px0, py0, px1, py1], outline=(120, 120, 130), width=2)
    placed: list[tuple[int, int, int]] = []

    def free(x, y, r):
        return all((x - a) ** 2 + (y - b) ** 2 > (r + c + 12) ** 2 for a, b, c in placed)

    reds = blues = 0
    while reds < 23:
        r = rng.randint(6, 11)
        x, y = rng.randint(px0 + 20, px1 - 20), rng.randint(py0 + 20, py1 - 20)
        if free(x, y, r):
            d.ellipse([x - r, y - r, x + r, y + r], fill=(220, 30, 40))
            placed.append((x, y, r))
            reds += 1
    while blues < 14:
        r = rng.randint(7, 12)
        x, y = rng.randint(px0 + 20, px1 - 20), rng.randint(py0 + 20, py1 - 20)
        if free(x, y, r):
            d.rectangle([x - r, y - r, x + r, y + r], fill=(40, 80, 220))
            placed.append((x, y, r))
            blues += 1
    truth["counts"] = {"panel": [px0, py0, px1, py1], "red_dots": reds, "blue_squares": blues}

    # line chart with a marked peak value
    cx0, cy0, cx1, cy1 = 2100, 700, 3700, 1500
    d.rectangle([cx0, cy0, cx1, cy1], outline=(120, 120, 130), width=2)
    vals = [rng.uniform(20, 80) for _ in range(40)]
    vals[27] = 97.3
    pts = [(cx0 + 20 + i * (cx1 - cx0 - 40) / 39, cy1 - 20 - v / 100 * (cy1 - cy0 - 40)) for i, v in enumerate(vals)]
    d.line(pts, fill=(0, 120, 200), width=3)
    d.text((pts[27][0] + 6, pts[27][1] - 6), "97.3", font=font(10), anchor="lb", fill=(200, 0, 0))
    truth["chart"] = {"box": [cx0, cy0, cx1, cy1], "peak_index": 27, "peak_label": "97.3"}

    # hidden icon
    icon = star(28)
    ix, iy = 3312, 1733
    im.paste(icon, (ix, iy))
    truth["icon"] = {"box": [ix, iy, ix + 28, iy + 28]}
    icon.save(OUT / "star_icon.png")
    im.save(OUT / "dashboard_4k.png")

    # a slightly modified copy for compare()
    im2 = im.copy()
    d2 = ImageDraw.Draw(im2)
    d2.rectangle([1400 + 2 * 190 + 1, 200 + 3 * 30 + 1, 1400 + 3 * 190 - 1, 200 + 4 * 30 - 1], fill=(255, 255, 255))
    d2.text((1400 + 2 * 190 + 8, 200 + 3.5 * 30), "45", font=font(11), anchor="lm", fill=(10, 10, 10))
    d2.ellipse([3000 - 9, 1800 - 9, 3000 + 9, 1800 + 9], fill=(220, 30, 40))
    im2.save(OUT / "dashboard_4k_v2.png")
    truth["compare"] = {"changes": [{"what": "table 대구/오류 44 -> 45", "box": [1780, 290, 1970, 320]},
                                    {"what": "new red dot", "center": [3000, 1800]}]}


def make_video(truth: dict) -> None:
    W, H, FPS, DUR = 1280, 720, 30, 24
    path = OUT / "demo.mp4"
    c = av.open(str(path), "w")
    try:
        vs = c.add_stream("libx264", rate=FPS, options={"crf": "20", "preset": "veryfast"})
    except Exception:
        vs = c.add_stream("mpeg4", rate=FPS)
        vs.bit_rate = 4_000_000
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    f_big, f_mid, f_small = font(48, True), font(30), font(18)
    typed = "hello"
    for n in range(FPS * DUR):
        t = n / FPS
        im = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(im)
        if t < 5:
            d.rectangle([0, 0, W, H], fill=(20, 60, 160))
            x = 100 + (t / 5) * (W - 200)
            d.ellipse([x - 40, 360 - 40, x + 40, 360 + 40], fill=(255, 255, 255))
            d.text((40, 40), "Scene 1 - 파란 화면", font=f_big, fill=(255, 255, 255))
        elif t < 10:
            d.rectangle([0, 0, W, H], fill=(30, 140, 60))
            k = int(t - 5) + 1
            d.text((40, 40), f"Scene 2 - Count: {k}", font=f_big, fill=(255, 255, 255))
            y = 200 + 300 * abs(math.sin((t - 5) * 1.3))
            d.rectangle([900, y, 1000, y + 100], fill=(255, 220, 0))
        elif t < 18:
            d.rectangle([0, 0, W, H], fill=(236, 238, 242))
            d.rectangle([0, 0, W, 56], fill=(45, 52, 70))
            d.text((20, 28), "MyApp - 설정", font=f_mid, anchor="lm", fill=(255, 255, 255))
            d.rectangle([20, 80, 300, 700], fill=(255, 255, 255), outline=(200, 200, 210))
            for i, item in enumerate(["일반", "네트워크", "계정", "로그"]):
                d.text((40, 110 + i * 50), item, font=f_mid, fill=(40, 40, 40))
            if t >= 11.0:
                d.rectangle([380, 120, 1180, 560], fill=(255, 255, 255), outline=(120, 120, 140), width=2)
                d.text((410, 150), "서버 주소 입력", font=f_mid, fill=(20, 20, 20))
                d.rectangle([410, 220, 1150, 280], outline=(90, 90, 110), width=2)
                if t >= 13.5:
                    nchar = min(len(typed), int((t - 13.5) / 0.25) + 1)
                    d.text((425, 250), typed[:nchar], font=f_mid, anchor="lm", fill=(10, 10, 10))
                color = (40, 170, 70) if t >= 16.0 else (200, 40, 40)
                d.rectangle([900, 470, 1150, 530], fill=color)
                d.text((1025, 500), "연결", font=f_mid, anchor="mm", fill=(255, 255, 255))
            if t >= 16.0:
                d.rectangle([840, 600, 1260, 690], fill=(60, 20, 20))
                d.text((860, 645), "오류: 연결 실패 (E-503)", font=f_small, anchor="lm", fill=(255, 200, 200))
        else:
            d.rectangle([0, 0, W, H], fill=(15, 15, 20))
            bx = 640 + 450 * math.sin((t - 18) * 2.1)
            by = 600 - 380 * abs(math.sin((t - 18) * 3.0))
            d.ellipse([bx - 30, by - 30, bx + 30, by + 30], fill=(255, 120, 0))
            d.text((40, 40), "Scene 4", font=f_big, fill=(230, 230, 230))
            d.text((W - 20, H - 20), f"frame {n}", font=f_small, anchor="rb", fill=(160, 160, 160))
        frame = av.VideoFrame.from_image(im)
        for pkt in vs.encode(frame):
            c.mux(pkt)
    for pkt in vs.encode():
        c.mux(pkt)
    c.close()
    truth["video"] = {"path": "samples/demo.mp4", "duration": DUR, "cuts": [5.0, 10.0, 18.0],
                      "ui_events": [{"t": 11.0, "what": "dialog opens"},
                                    {"t": 13.5, "what": "typing 'hello' (13.5-14.75s)"},
                                    {"t": 16.0, "what": "button red->green + error toast 'E-503'"}]}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    truth: dict = {}
    make_image(truth)
    make_video(truth)
    (OUT / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
