"""Model-free active-perception tools for still images and video frames."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from . import refs, render
from .config import settings
from .encode import to_mcp
from .render import Grid

# Last grid used per ref, so zoom("C3") refers to the same cells look() showed.
_grids: dict[str, Grid] = {}


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def _edge(max_edge: int | None) -> int:
    s = settings()
    return max(256, min(max_edge or s.max_edge, s.hard_max_edge))


def _grid_for(r: refs.Ref, W: int, H: int, spec: str | None) -> Grid:
    if spec:
        g = Grid.parse(spec, W, H)
        if g:
            return g
    return _grids.get(str(r)) or Grid.auto(W, H)


# ---------------------------------------------------------------- look

def look(ref: str, grid: str = "auto", max_edge: int | None = None) -> list:
    img, r = refs.load(ref)
    W, H = img.size
    g = Grid.parse(grid, W, H)
    if g:
        _grids[str(r)] = g
    view = render.render_view(img, (0, 0, W, H), long_edge=_edge(max_edge), min_edge=768, grid=g)
    mimg, meta = to_mcp(view.image)
    info = {"ref": str(r), "size": f"{W}x{H}", "view_scale": round(view.scale, 3), "image": meta}
    if g:
        info["grid"] = g.describe(W, H)
    lines = [_j(info), "Rulers (top/left) show ORIGINAL pixel coordinates; grid letters bottom, numbers right."]
    if view.scale < 0.999:
        px = max(2, math.ceil(3 / view.scale))
        lines.append(f"Downscaled x{view.scale:.2f}: features/text smaller than ~{px}px are NOT reliable in this "
                     f"view. Zoom before reading small text or counting: zoom(ref, 'C3' | 'B2:D4' | 'x0,y0,x1,y1').")
    return ["\n".join(lines), mimg]


# ---------------------------------------------------------------- zoom

def _clahe(im: Image.Image) -> Image.Image:
    import cv2

    lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def _otsu(im: Image.Image) -> Image.Image:
    import cv2

    g = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(b).convert("RGB")


def _denoise(im: Image.Image) -> Image.Image:
    import cv2

    a = np.asarray(im)
    if a.shape[0] * a.shape[1] > 4_000_000:
        return im.filter(ImageFilter.MedianFilter(3))
    return Image.fromarray(cv2.fastNlMeansDenoisingColored(a, None, 7, 7, 7, 21))


def _edges(im: Image.Image) -> Image.Image:
    import cv2

    g = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(255 - cv2.Canny(g, 60, 160)).convert("RGB")


ENHANCE = {
    "autocontrast": lambda im: ImageOps.autocontrast(im, cutoff=1),
    "clahe": _clahe,
    "equalize": ImageOps.equalize,
    "sharpen": lambda im: im.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2)),
    "denoise": _denoise,
    "gray": lambda im: ImageOps.grayscale(im).convert("RGB"),
    "invert": ImageOps.invert,
    "threshold": _otsu,
    "edges": _edges,
    "brighten": lambda im: ImageEnhance.Brightness(im).enhance(1.7),
    "darken": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "saturate": lambda im: ImageEnhance.Color(im).enhance(2.0),
    "rotate90": lambda im: im.transpose(Image.Transpose.ROTATE_270),  # 90 deg clockwise
    "rotate180": lambda im: im.transpose(Image.Transpose.ROTATE_180),
    "rotate270": lambda im: im.transpose(Image.Transpose.ROTATE_90),
    "flip_h": ImageOps.mirror,
    "flip_v": ImageOps.flip,
}
GEOMETRIC = {"rotate90", "rotate180", "rotate270", "flip_h", "flip_v"}


def apply_enhance(im: Image.Image, ops: str | None) -> tuple[Image.Image, list[str]]:
    done: list[str] = []
    for op in [o.strip().lower() for o in (ops or "").replace("+", ",").split(",") if o.strip()]:
        if op not in ENHANCE:
            raise ValueError(f"unknown enhance op {op!r}; choose from {', '.join(ENHANCE)}")
        im = ENHANCE[op](im)
        done.append(op)
    return im, done


def zoom(ref: str, region: str, enhance: str | None = None, pad: float = 0.0,
         grid: str | None = None, max_edge: int | None = None) -> list:
    img, r = refs.load(ref)
    W, H = img.size
    g = _grid_for(r, W, H, grid)
    box = render.pad_box(render.parse_region(region, W, H, g), W, H, pad)
    crop, ops = apply_enhance(img.crop(box), enhance)
    geometric = any(o in GEOMETRIC for o in ops)
    if geometric:
        view = render.render_view(crop, (0, 0, *crop.size), long_edge=_edge(max_edge),
                                  min_edge=settings().zoom_min_edge, max_upscale=6.0, rulers=False)
    else:
        # render the (possibly enhanced) crop but label rulers with ORIGINAL coordinates
        view = render.render_view(img, box, long_edge=_edge(max_edge), min_edge=settings().zoom_min_edge,
                                  max_upscale=6.0, grid=g, full_size=(W, H), crop=crop, grid_lines=False)
    mimg, meta = to_mcp(view.image)
    x0, y0, x1, y1 = box
    info = {"ref": str(r), "region": [x0, y0, x1, y1], "region_size": f"{x1 - x0}x{y1 - y0}",
            "magnification": round(view.scale, 2), "image": meta}
    if ops:
        info["enhance"] = ops
    note = ("Rulers show ORIGINAL image coordinates (read positions directly from them)."
            if not geometric else "Rotated/flipped view: rulers disabled; coordinates are not mapped.")
    return [_j(info) + "\n" + note, mimg]


# ---------------------------------------------------------------- scan

def scan(ref: str, region: str | None = None, tile: int = 1024, overlap: float = 0.1,
         page: int = 1, per_page: int = 6) -> list:
    img, r = refs.load(ref)
    W, H = img.size
    bx0, by0, bx1, by1 = render.parse_region(region, W, H, _grid_for(r, W, H, None))
    bw, bh = bx1 - bx0, by1 - by0
    tile = max(256, min(tile, 2048))
    cols, rows = max(1, math.ceil(bw / tile)), max(1, math.ceil(bh / tile))
    tw, th = bw / cols, bh / rows
    ox, oy = tw * overlap, th * overlap
    tiles = []
    for rr in range(rows):
        for cc in range(cols):
            b = render.clamp_box((bx0 + cc * tw - ox, by0 + rr * th - oy,
                                  bx0 + (cc + 1) * tw + ox, by0 + (rr + 1) * th + oy), W, H)
            tiles.append((f"r{rr + 1}c{cc + 1}", b))
    per_page = max(1, min(per_page, 8))
    pages = math.ceil(len(tiles) / per_page)
    page = max(1, min(page, pages))
    chunk = tiles[(page - 1) * per_page: page * per_page]
    out: list = []
    listing = []
    for name, b in chunk:
        view = render.render_view(img, b, long_edge=min(_edge(None), tile + 200), min_edge=0)
        mimg, meta = to_mcp(view.image)
        listing.append({"tile": name, "box": list(b), "scale": round(view.scale, 2)})
        out.append(mimg)
    head = {"ref": str(r), "size": f"{W}x{H}", "tiles": f"{rows}x{cols}", "page": f"{page}/{pages}",
            "shown": listing}
    more = f"\nMore tiles: scan(ref, page={page + 1})" if page < pages else ""
    return [_j(head) + "\nImages below follow the order of 'shown'. Rulers = original px." + more, *out]


# ---------------------------------------------------------------- pixels

def pixels(ref: str, points: str | None = None, region: str | None = None, k: int = 5) -> list:
    img, r = refs.load(ref)
    W, H = img.size
    a = np.asarray(img)
    res: dict[str, Any] = {"ref": str(r), "size": f"{W}x{H}"}
    if points:
        pts = []
        for chunk in points.replace("|", ";").split(";"):
            nums = [float(v) for v in chunk.replace("(", " ").replace(")", " ").replace(",", " ").split()]
            if len(nums) != 2:
                continue
            x, y = nums
            if max(abs(x), abs(y)) <= 1.0:
                x, y = x * W, y * H
            xi, yi = int(min(max(x, 0), W - 1)), int(min(max(y, 0), H - 1))
            px = a[yi, xi].tolist()
            nb = a[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].reshape(-1, 3).mean(0)
            pts.append({"xy": [xi, yi], "rgb": px, "hex": "#%02x%02x%02x" % tuple(px),
                        "mean3x3": [round(v) for v in nb.tolist()]})
        res["points"] = pts
    if region or not points:
        x0, y0, x1, y1 = render.parse_region(region, W, H, _grid_for(r, W, H, None))
        sub = a[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        res["region"] = [x0, y0, x1, y1]
        res["mean_rgb"] = [round(v, 1) for v in sub.mean(0).tolist()]
        res["std_rgb"] = [round(v, 1) for v in sub.std(0).tolist()]
        res["dominant"] = _dominant(sub, k)
    return [_j(res)]


def _dominant(pixels: np.ndarray, k: int) -> list[dict]:
    import cv2

    if len(pixels) > 60_000:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), 60_000, replace=False)]
    k = max(1, min(k, 12, len(pixels)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    order = np.argsort(-counts)
    return [{"hex": "#%02x%02x%02x" % tuple(int(round(c)) for c in centers[i]),
             "pct": round(100 * counts[i] / counts.sum(), 1)} for i in order]


# ---------------------------------------------------------------- annotate

def annotate(ref: str, shapes: list[dict], region: str | None = "auto", max_edge: int | None = None) -> list:
    img, r = refs.load(ref)
    W, H = img.size
    items = [dict(s) for s in shapes]
    if region in (None, "", "auto"):
        xs, ys = [], []
        for s in items:
            for key in ("box", "line"):
                if s.get(key):
                    xs += [s[key][0], s[key][2]]
                    ys += [s[key][1], s[key][3]]
            if s.get("point"):
                xs.append(s["point"][0])
                ys.append(s["point"][1])
            for p in s.get("polygon") or []:
                xs.append(p[0])
                ys.append(p[1])
        if xs:
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            m = max(64, 0.4 * max(bw, bh))
            box = render.clamp_box((min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m), W, H)
        else:
            box = (0, 0, W, H)
    else:
        box = render.parse_region(region, W, H, _grid_for(r, W, H, None))
    view = render.render_view(img, box, long_edge=_edge(max_edge), min_edge=768, max_upscale=4.0)
    render.draw_marks(view, items)
    mimg, meta = to_mcp(view.image)
    return [_j({"ref": str(r), "region": list(box), "shapes": len(items), "image": meta})
            + "\nCheck that each mark sits exactly on its target; adjust coordinates and re-annotate if not.", mimg]


# ---------------------------------------------------------------- compare

def compare(ref_a: str, ref_b: str, region: str | None = None, threshold: int = 24,
            max_regions: int = 12, max_edge: int | None = None) -> list:
    import cv2

    A, ra = refs.load(ref_a)
    B, rb = refs.load(ref_b)
    W, H = A.size
    note = []
    if B.size != A.size:
        note.append(f"B resized from {B.size[0]}x{B.size[1]} to {W}x{H} for comparison")
        B = B.resize(A.size, Image.LANCZOS)
    box = render.parse_region(region, W, H, _grid_for(ra, W, H, None))
    x0, y0, x1, y1 = box
    a = np.asarray(A)[y0:y1, x0:x1].astype(np.int16)
    b = np.asarray(B)[y0:y1, x0:x1].astype(np.int16)
    d = np.abs(a - b).max(axis=2).astype(np.uint8)
    mask = (d > threshold).astype(np.uint8)
    k = max(3, int(round(max(mask.shape) / 200)) | 1)
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    comps = sorted((stats[i] for i in range(1, n)), key=lambda s: -int(s[4]))
    regions = []
    for i, (cx, cy, cw, ch, area) in enumerate(comps[:max_regions]):
        if area < 4:
            continue
        sub = d[cy:cy + ch, cx:cx + cw]
        regions.append({"id": i + 1, "box": [int(x0 + cx), int(y0 + cy), int(x0 + cx + cw), int(y0 + cy + ch)],
                        "changed_px": int(mask[cy:cy + ch, cx:cx + cw].sum()), "max_diff": int(sub.max())})
    changed = float(mask.mean())
    mse = float(((a - b).astype(np.float32) ** 2).mean())
    psnr = 99.0 if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 2)

    half = _edge(max_edge) // 2
    va = render.render_view(A, box, long_edge=half, min_edge=0)
    vb = render.render_view(B, box, long_edge=half, min_edge=0)
    marks = [{"id": g["id"], "box": g["box"]} for g in regions]
    render.draw_marks(va, marks)
    render.draw_marks(vb, marks)
    side = Image.new("RGB", (va.image.width + vb.image.width + 8, max(va.image.height, vb.image.height) + 22),
                     render.BG)
    side.paste(va.image, (0, 22))
    side.paste(vb.image, (va.image.width + 8, 22))
    from PIL import ImageDraw

    dr = ImageDraw.Draw(side)
    dr.text((6, 11), "A", font=render.font(15, True), anchor="lm", fill=(255, 214, 10))
    dr.text((va.image.width + 14, 11), "B", font=render.font(15, True), anchor="lm", fill=(255, 214, 10))
    mimg, meta = to_mcp(side)
    info = {"a": str(ra), "b": str(rb), "region": list(box), "identical": changed == 0,
            "changed_pct": round(100 * changed, 3), "psnr_db": psnr, "regions": regions, "image": meta}
    if note:
        info["note"] = note
    return [_j(info) + "\nNumbered boxes = changed areas (same ids on A and B). Zoom into each ref to inspect.", mimg]


# ---------------------------------------------------------------- blobs

def _parse_color(spec: str) -> tuple[int, int, int]:
    from PIL import ImageColor

    s = spec.strip()
    nums = [int(float(v)) for v in s.replace("(", " ").replace(")", " ").replace(",", " ").split()
            if v.replace(".", "", 1).isdigit()]
    if len(nums) == 3:
        return tuple(max(0, min(255, v)) for v in nums)  # type: ignore[return-value]
    return ImageColor.getrgb(s if s.startswith("#") or not all(c in "0123456789abcdefABCDEF" for c in s)
                             else "#" + s)[:3]


def blobs(ref: str, color: str | None = None, mode: str = "color", region: str | None = None,
          tolerance: float = 30.0, min_area: int = 6, max_area: int | None = None, separate: bool = False,
          max_results: int = 300, max_edge: int | None = None) -> list:
    """Connected regions matching a colour (CIELAB distance) or dark/light ink: exact count + boxes."""
    import cv2

    img, r = refs.load(ref)
    W, H = img.size
    box = render.parse_region(region, W, H, _grid_for(r, W, H, None))
    x0, y0, x1, y1 = box
    a = np.ascontiguousarray(np.asarray(img)[y0:y1, x0:x1])
    if mode == "color":
        if not color:
            raise ValueError("give color='#rrggbb' (read it with pixels()) or a name like 'red', or use mode dark/light")
        rgb = _parse_color(color)
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        # OpenCV 8-bit Lab: L scaled by 255/100, a/b offset by 128
        d = lab - tgt
        d[..., 0] *= 100.0 / 255.0
        mask = (np.sqrt((d ** 2).sum(-1)) <= tolerance).astype(np.uint8)
    elif mode in ("dark", "light"):
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        flag = cv2.THRESH_BINARY_INV if mode == "dark" else cv2.THRESH_BINARY
        _, m = cv2.threshold(g, 0, 1, flag + cv2.THRESH_OTSU)
        mask = m.astype(np.uint8)
    else:
        raise ValueError("mode must be color | dark | light")
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if separate:  # split touching round-ish blobs via distance-transform peaks
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        _, peaks = cv2.threshold(dist, 0.5 * dist.max() if dist.max() else 1, 1, cv2.THRESH_BINARY)
        mask = peaks.astype(np.uint8)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, n):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area or (max_area and area > max_area):
            continue
        found.append({"box": [x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh],
                      "center": [round(x0 + float(cents[i][0]), 1), round(y0 + float(cents[i][1]), 1)], "area": area})
    found.sort(key=lambda f: (f["box"][1] // 10, f["box"][0]))
    total = len(found)
    found = found[:max_results]
    for i, f in enumerate(found):
        f["id"] = i + 1
    areas = sorted(f["area"] for f in found)
    view = render.render_view(img, box, long_edge=_edge(max_edge), min_edge=768)
    render.draw_marks(view, [{"id": f["id"], "box": f["box"]} for f in found], width=1, label_size=10)
    mimg, meta = to_mcp(view.image)
    info = {"ref": str(r), "region": list(box), "mode": mode, "color": color, "tolerance": tolerance,
            "count": total, "area_px": {"min": areas[0], "median": areas[len(areas) // 2], "max": areas[-1]}
            if areas else None, "image": meta}
    tips = ("Check the overlay: merged neighbours -> separate=True; noise -> raise min_area; misses -> raise "
            "tolerance. A blob with ~2x the median area is often two touching objects.")
    return [_j(info) + "\nBLOBS:\n" + "\n".join(_j({k: f[k] for k in ("id", "center", "box", "area")}) for f in found)
            + "\n" + tips, mimg]


# ---------------------------------------------------------------- locate

def locate(ref: str, template: str, template_region: str | None = None, region: str | None = None,
           threshold: float = 0.8, max_results: int = 10, multiscale: bool = True,
           max_edge: int | None = None) -> list:
    import cv2

    img, r = refs.load(ref)
    W, H = img.size
    timg, tr = refs.load(template)
    if template_region:
        tw_, th_ = timg.size
        timg = timg.crop(render.parse_region(template_region, tw_, th_, _grid_for(tr, tw_, th_, None)))
    box = render.parse_region(region, W, H, _grid_for(r, W, H, None))
    x0, y0, x1, y1 = box
    hay = cv2.cvtColor(np.asarray(img)[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    needle0 = cv2.cvtColor(np.asarray(timg), cv2.COLOR_RGB2GRAY)
    scales = np.geomspace(0.5, 2.0, 13) if multiscale else [1.0]
    cands = []
    for s in scales:
        nw, nh = max(4, round(needle0.shape[1] * s)), max(4, round(needle0.shape[0] * s))
        if nw >= hay.shape[1] or nh >= hay.shape[0]:
            continue
        needle = cv2.resize(needle0, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        if needle.std() < 1e-3:
            continue
        res = cv2.matchTemplate(hay, needle, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.nonzero(res >= threshold)
        for yy, xx in zip(ys, xs):
            cands.append((float(res[yy, xx]), int(xx), int(yy), nw, nh, float(s)))
    cands.sort(key=lambda c: -c[0])
    picked: list[tuple] = []
    for c in cands:
        if len(picked) >= max_results:
            break
        if all(_iou((c[1], c[2], c[1] + c[3], c[2] + c[4]), (p[1], p[2], p[1] + p[3], p[2] + p[4])) < 0.3
               for p in picked):
            picked.append(c)
    matches = [{"id": i + 1, "score": round(c[0], 3), "scale": round(c[5], 2),
                "box": [x0 + c[1], y0 + c[2], x0 + c[1] + c[3], y0 + c[2] + c[4]],
                "center": [x0 + c[1] + c[3] // 2, y0 + c[2] + c[4] // 2]} for i, c in enumerate(picked)]
    view = render.render_view(img, box, long_edge=_edge(max_edge), min_edge=768)
    render.draw_marks(view, [{"id": m["id"], "box": m["box"]} for m in matches])
    mimg, meta = to_mcp(view.image)
    info = {"ref": str(r), "template": str(tr), "template_size": f"{timg.width}x{timg.height}",
            "found": len(matches), "matches": matches, "image": meta}
    tip = "" if matches else f"\nNo match >= {threshold}. Try a lower threshold (e.g. 0.6) or a tighter template_region."
    return [_j(info) + tip, mimg]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
