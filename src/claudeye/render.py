"""Drawing helpers: fonts, rulers, grids, Set-of-Mark labels, contact sheets.

Coordinate convention everywhere: ORIGINAL image pixels, origin top-left,
boxes are (x0, y0, x1, y1) with x1/y1 exclusive.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

BG = (22, 22, 26)
TICK = (170, 170, 180)
GRID_LINE = (255, 0, 220, 170)
GRID_LABEL = (255, 214, 10)
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
    (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128),
]


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["malgunbd.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else []
    names += ["malgun.ttf", "arial.ttf", "DejaVuSans.ttf", "NotoSansCJK-Regular.ttc",
              "AppleSDGothicNeo.ttc", "Arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def text_color_for(bg: tuple[int, ...]) -> tuple[int, int, int]:
    r, g, b = bg[:3]
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (255, 255, 255)


def tag(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, *, size: int = 13,
        fill: tuple[int, ...] = (0, 0, 0, 200), fg: tuple[int, ...] | None = None,
        anchor: str = "lt", pad: int = 2, bold: bool = True) -> tuple[float, float, float, float]:
    f = font(size, bold)
    l, t, r, b = draw.textbbox(xy, text, font=f, anchor=anchor)
    rect = (l - pad, t - pad, r + pad, b + pad)
    draw.rectangle(rect, fill=fill)
    draw.text(xy, text, font=f, anchor=anchor, fill=fg or text_color_for(fill))
    return rect


# ---------------------------------------------------------------- grid

def col_name(i: int) -> str:
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def col_index(name: str) -> int:
    n = 0
    for ch in name.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


@dataclass(frozen=True)
class Grid:
    cols: int
    rows: int

    @staticmethod
    def auto(w: int, h: int, long_div: int = 8) -> Grid:
        if w >= h:
            return Grid(long_div, max(2, int(long_div * h / w + 0.5)))
        return Grid(max(2, int(long_div * w / h + 0.5)), long_div)

    @staticmethod
    def parse(spec: str | None, w: int, h: int) -> Grid | None:
        s = (spec or "auto").strip().lower()
        if s in ("none", "off", "false", "0"):
            return None
        if s == "auto":
            return Grid.auto(w, h)
        if s == "fine":
            return Grid.auto(w, h, 16)
        m = re.fullmatch(r"(\d{1,2})\s*[x×,]\s*(\d{1,2})", s)
        if not m:
            raise ValueError(f"grid must be 'auto', 'fine', 'none' or 'COLSxROWS' (e.g. 8x5), got {spec!r}")
        cols, rows = int(m.group(1)), int(m.group(2))
        if not (1 <= cols <= 52 and 1 <= rows <= 99):
            raise ValueError("grid out of range (cols 1-52, rows 1-99)")
        return Grid(cols, rows)

    def xs(self, w: int) -> list[int]:
        return [round(w * i / self.cols) for i in range(self.cols + 1)]

    def ys(self, h: int) -> list[int]:
        return [round(h * i / self.rows) for i in range(self.rows + 1)]

    def cell_box(self, w: int, h: int, c: int, r: int) -> tuple[int, int, int, int]:
        xs, ys = self.xs(w), self.ys(h)
        return xs[c], ys[r], xs[c + 1], ys[r + 1]

    def describe(self, w: int, h: int) -> str:
        return (f"{self.cols}x{self.rows} grid: columns A-{col_name(self.cols - 1)} (left->right), "
                f"rows 1-{self.rows} (top->bottom), cell ~{w / self.cols:.0f}x{h / self.rows:.0f}px")


_CELL = re.compile(r"([A-Z]{1,2})(\d{1,3})")


def parse_region(spec: str | None, w: int, h: int, grid: Grid | None = None,
                 point_radius: int = 256) -> tuple[int, int, int, int]:
    """Region spec -> pixel box. Accepts:
    'all' | 'C3' | 'B2:D4' | 'x0,y0,x1,y1' (px) | '0.1,0.2,0.5,0.6' (fractions) | 'x,y' (point; crop around it)
    """
    if spec is None or spec.strip().lower() in ("", "all", "full", "*", "whole"):
        return 0, 0, w, h
    s = spec.strip()
    compact = re.sub(r"\s+", "", s.upper())
    m = re.fullmatch(r"([A-Z]{1,2}\d{1,3})(?::([A-Z]{1,2}\d{1,3}))?", compact)
    if m:
        g = grid or Grid.auto(w, h)
        cells = []
        for part in (m.group(1), m.group(2) or m.group(1)):
            cm = _CELL.fullmatch(part)
            c, r = col_index(cm.group(1)), int(cm.group(2)) - 1
            if not (0 <= c < g.cols and 0 <= r < g.rows):
                raise ValueError(f"cell {part} outside {g.cols}x{g.rows} grid "
                                 f"(A-{col_name(g.cols - 1)}, 1-{g.rows})")
            cells.append((c, r))
        (c0, r0), (c1, r1) = cells
        c0, c1 = sorted((c0, c1))
        r0, r1 = sorted((r0, r1))
        x0, y0, _, _ = g.cell_box(w, h, c0, r0)
        _, _, x1, y1 = g.cell_box(w, h, c1, r1)
        return x0, y0, x1, y1
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", s)]
    if len(nums) == 2:
        x, y = nums
        if max(abs(x), abs(y)) <= 1.0:
            x, y = x * w, y * h
        r = point_radius
        return clamp_box((x - r, y - r, x + r, y + r), w, h)
    if len(nums) == 4:
        if max(abs(v) for v in nums) <= 1.0:
            nums = [nums[0] * w, nums[1] * h, nums[2] * w, nums[3] * h]
        x0, y0, x1, y1 = nums
        return clamp_box((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), w, h)
    raise ValueError(f"cannot parse region {spec!r}; use 'C3', 'B2:D4', 'x0,y0,x1,y1', "
                     f"fractions '0.1,0.2,0.5,0.6', or a point 'x,y'")


def clamp_box(b, w: int, h: int, min_size: int = 4) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
    if x1 - x0 < min_size:
        x0, x1 = max(0, x0 - min_size // 2), min(w, x1 + min_size // 2)
    if y1 - y0 < min_size:
        y0, y1 = max(0, y0 - min_size // 2), min(h, y1 + min_size // 2)
    return x0, y0, x1, y1


def pad_box(b, w: int, h: int, frac: float) -> tuple[int, int, int, int]:
    if frac <= 0:
        return b
    x0, y0, x1, y1 = b
    dx, dy = (x1 - x0) * frac, (y1 - y0) * frac
    return clamp_box((x0 - dx, y0 - dy, x1 + dx, y1 + dy), w, h)


# ---------------------------------------------------------------- views

def nice_step(raw: float) -> int:
    if raw <= 1:
        return 1
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw and float(step).is_integer():
            return int(step)
    return int(10 * mag)


@dataclass
class View:
    image: Image.Image
    box: tuple[int, int, int, int]
    scale: float
    origin: tuple[int, int]

    def to_display(self, x: float, y: float) -> tuple[float, float]:
        return (self.origin[0] + (x - self.box[0]) * self.scale,
                self.origin[1] + (y - self.box[1]) * self.scale)

    @property
    def content_size(self) -> tuple[int, int]:
        return (round((self.box[2] - self.box[0]) * self.scale),
                round((self.box[3] - self.box[1]) * self.scale))


def render_view(src: Image.Image, box: tuple[int, int, int, int], *, long_edge: int,
                min_edge: int = 0, max_upscale: float = 4.0, rulers: bool = True,
                grid: Grid | None = None, full_size: tuple[int, int] | None = None,
                crop: Image.Image | None = None, grid_lines: bool = True) -> View:
    """Render `box` of `src` scaled to fit `long_edge` (incl. margins), with pixel rulers
    (top/left, in ORIGINAL coordinates) and optional grid lines + cell labels (bottom/right).
    `crop` may supply pre-processed pixels for `box` (e.g. enhanced); `src` is then unused."""
    x0, y0, x1, y1 = box
    if crop is None:
        crop = src if (x0, y0, x1, y1) == (0, 0, *src.size) else src.crop(box)
    elif crop.size != (x1 - x0, y1 - y0):
        crop = crop.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    w, h = crop.size
    ml, mt = (52, 20) if rulers else (0, 0)
    mr, mb = (30, 22) if grid else (0, 0)
    avail = max(64, long_edge - max(ml + mr, mt + mb))
    s_fit = avail / max(w, h)
    s = s_fit if s_fit < 1 else min(s_fit, max(1.0, min_edge / max(w, h)), max_upscale)
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    shown = crop if (rw, rh) == (w, h) else crop.resize((rw, rh), Image.LANCZOS)

    canvas = Image.new("RGB", (ml + rw + mr, mt + rh + mb), BG)
    canvas.paste(shown, (ml, mt))
    view = View(canvas, box, s, (ml, mt))

    if grid is not None:
        fw, fh = full_size or (src.size if src is not None else (x1, y1))
        _draw_grid(view, grid, fw, fh, lines=grid_lines)
    if rulers:
        _draw_rulers(view)
    return view


def _draw_rulers(view: View) -> None:
    d = ImageDraw.Draw(view.image)
    f = font(11)
    x0, y0, x1, y1 = view.box
    s = view.scale
    ml, mt = view.origin
    rw, rh = view.content_size
    step = nice_step(80 / s)
    minor = step / 5 if step >= 5 else 0
    if minor:
        v = math.ceil(x0 / minor) * minor
        while v <= x1:
            dx = ml + (v - x0) * s
            d.line([(dx, mt - 3), (dx, mt - 1)], fill=TICK)
            v += minor
    for v in range(math.ceil(x0 / step) * step, x1 + 1, step):
        dx = ml + (v - x0) * s
        d.line([(dx, mt - 7), (dx, mt - 1)], fill=TICK)
        d.text((min(max(dx, ml + 12), ml + rw - 12), mt - 8), str(v), font=f, anchor="mb", fill=TICK)
    step_y = nice_step(48 / s)
    minor = step_y / 5 if step_y >= 5 else 0
    if minor:
        v = math.ceil(y0 / minor) * minor
        while v <= y1:
            dy = mt + (v - y0) * s
            d.line([(ml - 3, dy), (ml - 1, dy)], fill=TICK)
            v += minor
    for v in range(math.ceil(y0 / step_y) * step_y, y1 + 1, step_y):
        dy = mt + (v - y0) * s
        d.line([(ml - 7, dy), (ml - 1, dy)], fill=TICK)
        d.text((ml - 9, min(max(dy, mt + 6), mt + rh - 6)), str(v), font=f, anchor="rm", fill=TICK)
    d.rectangle([ml - 1, mt - 1, ml + rw, mt + rh], outline=(90, 90, 100))


def _dashed(d: ImageDraw.ImageDraw, p0, p1, fill, dash: int = 7, gap: int = 5) -> None:
    (xa, ya), (xb, yb) = p0, p1
    length = math.hypot(xb - xa, yb - ya)
    if length == 0:
        return
    ux, uy = (xb - xa) / length, (yb - ya) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        d.line([(xa + ux * pos, ya + uy * pos), (xa + ux * end, ya + uy * end)], fill=fill, width=1)
        pos += dash + gap


def _draw_grid(view: View, grid: Grid, fw: int, fh: int, lines: bool = True) -> None:
    """Grid cell labels in the bottom/right margins. `lines`: dashed lines across the content
    (overview); otherwise only boundary ticks in the margins, so zoomed content stays clean."""
    x0, y0, x1, y1 = view.box
    ml, mt = view.origin
    rw, rh = view.content_size
    overlay = Image.new("RGBA", view.image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    xs, ys = grid.xs(fw), grid.ys(fh)
    for gx in xs[1:-1]:
        if x0 < gx < x1:
            dx, _ = view.to_display(gx, 0)
            if lines:
                _dashed(od, (dx, mt), (dx, mt + rh), GRID_LINE)
            od.line([(dx, mt + rh + 1), (dx, mt + rh + 8)], fill=GRID_LABEL + (255,), width=2)
    for gy in ys[1:-1]:
        if y0 < gy < y1:
            _, dy = view.to_display(0, gy)
            if lines:
                _dashed(od, (ml, dy), (ml + rw, dy), GRID_LINE)
            od.line([(ml + rw + 1, dy), (ml + rw + 8, dy)], fill=GRID_LABEL + (255,), width=2)
    view.image.paste(Image.alpha_composite(view.image.convert("RGBA"), overlay).convert("RGB"))
    d = ImageDraw.Draw(view.image)
    f = font(14, bold=True)
    for c in range(grid.cols):
        a, b = max(xs[c], x0), min(xs[c + 1], x1)
        if b - a > 0:
            dx = view.to_display((a + b) / 2, 0)[0]
            d.text((dx, mt + rh + 11), col_name(c), font=f, anchor="mm", fill=GRID_LABEL)
    for r in range(grid.rows):
        a, b = max(ys[r], y0), min(ys[r + 1], y1)
        if b - a > 0:
            dy = view.to_display(0, (a + b) / 2)[1]
            d.text((ml + rw + 15, dy), str(r + 1), font=f, anchor="mm", fill=GRID_LABEL)


# ---------------------------------------------------------------- marks

def draw_marks(view: View, items: list[dict], *, width: int = 2, label_size: int = 13) -> None:
    """Set-of-Mark overlay. item keys: id, and one of box=[x0,y0,x1,y1] | point=[x,y] |
    polygon=[[x,y],...] | line=[x0,y0,x1,y1]; optional color=[r,g,b], label=str."""
    overlay = Image.new("RGBA", view.image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    tags = []
    for i, it in enumerate(items):
        color = tuple(it.get("color") or PALETTE[i % len(PALETTE)])
        name = str(it.get("label", it.get("id", i + 1)))
        if "polygon" in it and it["polygon"]:
            pts = [view.to_display(x, y) for x, y in it["polygon"]]
            od.polygon(pts, outline=color + (255,), fill=color + (40,), width=width)
            ax, ay = min(pts, key=lambda p: (p[1], p[0]))
            tags.append(((ax, ay - 2), name, color, "lb"))
        if "box" in it and it["box"]:
            bx0, by0, bx1, by1 = it["box"]
            (dx0, dy0), (dx1, dy1) = view.to_display(bx0, by0), view.to_display(bx1, by1)
            od.rectangle([dx0, dy0, dx1, dy1], outline=color + (255,), width=width)
            if dy0 - view.origin[1] > label_size + 6:
                tags.append(((dx0, dy0 - 1), name, color, "lb"))
            else:
                tags.append(((dx0 + 1, dy0 + 1), name, color, "lt"))
        if "point" in it and it["point"]:
            px, py = view.to_display(*it["point"])
            r = 6
            od.ellipse([px - r, py - r, px + r, py + r], outline=color + (255,), width=2)
            od.line([(px - r - 4, py), (px + r + 4, py)], fill=color + (255,), width=1)
            od.line([(px, py - r - 4), (px, py + r + 4)], fill=color + (255,), width=1)
            tags.append(((px + r + 2, py - r - 2), name, color, "lb"))
        if "line" in it and it["line"]:
            ax0, ay0, ax1, ay1 = it["line"]
            p0, p1 = view.to_display(ax0, ay0), view.to_display(ax1, ay1)
            od.line([p0, p1], fill=color + (255,), width=width)
            _arrow_head(od, p0, p1, color + (255,))
            tags.append(((p0[0] + 3, p0[1] - 3), name, color, "lb"))
    base = Image.alpha_composite(view.image.convert("RGBA"), overlay)
    d = ImageDraw.Draw(base)
    for xy, name, color, anchor in tags:
        tag(d, xy, name, size=label_size, fill=color + (235,), anchor=anchor)
    view.image.paste(base.convert("RGB"))


def _arrow_head(d: ImageDraw.ImageDraw, p0, p1, color, size: float = 10) -> None:
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for sign in (1, -1):
        a = ang + math.pi - sign * math.pi / 7
        d.line([p1, (p1[0] + size * math.cos(a), p1[1] + size * math.sin(a))], fill=color, width=2)


# ---------------------------------------------------------------- contact sheets

def contact_sheet(cells: list[tuple[Image.Image, str]], *, cols: int, cell_w: int,
                  header: str | None = None, label_h: int = 20, gap: int = 4) -> Image.Image:
    """Grid of frames, each with a label bar (e.g. '#3  00:12.40') above it."""
    if not cells:
        raise ValueError("no frames")
    cols = max(1, min(cols, len(cells)))
    rows = math.ceil(len(cells) / cols)
    ar = max(im.height / im.width for im, _ in cells)
    cell_h = max(1, round(cell_w * ar))
    head_h = 24 if header else 0
    W = gap + cols * (cell_w + gap)
    H = head_h + gap + rows * (label_h + cell_h + gap)
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    if header:
        d.text((gap + 2, head_h // 2 + 1), header, font=font(14, bold=True), anchor="lm", fill=(235, 235, 235))
    lf = font(13, bold=True)
    for i, (im, label) in enumerate(cells):
        r, c = divmod(i, cols)
        x = gap + c * (cell_w + gap)
        y = head_h + gap + r * (label_h + cell_h + gap)
        d.rectangle([x, y, x + cell_w - 1, y + label_h - 1], fill=(48, 48, 56))
        d.text((x + 5, y + label_h // 2), label, font=lf, anchor="lm", fill=(255, 214, 10))
        th = round(cell_w * im.height / im.width)
        thumb = im.resize((cell_w, th), Image.LANCZOS) if im.size != (cell_w, th) else im
        sheet.paste(thumb, (x, y + label_h + (cell_h - th) // 2))
    return sheet
