"""ClaudeYe MCP server: precise image & video perception tools for Claude Code.

Heavy modules (numpy/cv2/av/models) are imported inside tools so the server starts fast.
"""

from __future__ import annotations

import functools
from typing import Annotated, Any

import anyio.from_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from . import __version__

INSTRUCTIONS = """\
ClaudeYe gives you precise eyes for images and videos. Your own view of an image is downscaled and you cannot
watch video at all, so use these tools instead of guessing.
IMAGES: look(ref) first (labelled grid + ORIGINAL-pixel rulers). Before reading small text, counting, or judging
fine detail, zoom(ref, region) with region = 'C3', 'B2:D4' or 'x0,y0,x1,y1'. Never guess text you cannot read
clearly: zoom further (enhance='sharpen' or 'clahe' helps) or run ocr(). Report coordinates read from the rulers
and verify locations with annotate(). Use scan() to sweep a large image tile by tile at native resolution.
VIDEOS: video_info -> video_storyboard (timestamped contact sheet of shots/changes) -> video_frames(start,end,count)
for a denser look at a span -> zoom('video.mp4@SECONDS', region) for detail. Screen recordings / UI bugs:
video_changes (what changed, when, where). Speech: transcribe(). Every image tool accepts 'video.mp4@SECONDS'.
Also: compare(a, b) for visual diffs, locate(ref, template) to find an icon/element, pixels() for exact colours,
detect()/segment()/ocr() for objects, masks and text with exact boxes (call vision_status() to see which models
are installed), ask_vision() for a second opinion from another vision model. Say when evidence is weak."""

_mcp = MCPServer("claudeye", instructions=INSTRUCTIONS, version=__version__)
RO = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


def _safe(fn):
    """Surface the real error text to the model (the SDK hides messages of unexpected exceptions)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except FileNotFoundError as e:
            raise ToolError(f"{e}. Use an absolute path, or a path relative to the project root.") from e
        except ImportError as e:
            raise ToolError(f"Optional component missing ({e}). Run vision_status() for install hints.") from e
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper


class _Registry:
    """`@mcp.tool(...)` that also wraps each tool with `_safe`."""

    def __init__(self, server: MCPServer):
        self.server = server

    def tool(self, **kwargs):
        def deco(fn):
            self.server.tool(**kwargs)(_safe(fn))
            return fn

        return deco


mcp = _Registry(_mcp)

from .params import MaxEdge, OptRegion, Ref, Region, Time, VideoPath  # noqa: E402


def _progress(ctx: Context | None):
    if ctx is None:
        return None

    def cb(frac: float, msg: str) -> None:
        try:
            anyio.from_thread.run(ctx.report_progress, frac, 1.0, msg)
        except Exception:
            pass

    return cb


# ---------------------------------------------------------------- images

@mcp.tool(annotations=RO, structured_output=False)
def look(ref: Ref,
         grid: Annotated[str, Field(description="'auto' (8 cells on the long side), 'fine' (16), 'none', "
                                                "or 'COLSxROWS' e.g. '6x4'.")] = "auto",
         max_edge: MaxEdge = None, ctx: Context | None = None) -> list:
    """Overview of an image or video frame with a labelled grid (columns A.., rows 1..) and rulers in ORIGINAL
    pixels. Start here, then zoom() into cells. Passing a video path without @time returns a storyboard."""
    from . import image_tools, refs, video_tools

    r = refs.parse_ref(ref)
    if r.is_video and r.time is None:
        return video_tools.video_storyboard(ref, progress=_progress(ctx))
    return image_tools.look(ref, grid=grid, max_edge=max_edge)


@mcp.tool(annotations=RO, structured_output=False)
def zoom(ref: Ref, region: Region,
         enhance: Annotated[str | None, Field(description="Optional comma list applied in order: sharpen, clahe, "
                                                          "autocontrast, equalize, denoise, gray, invert, threshold, "
                                                          "edges, brighten, darken, saturate, rotate90, rotate180, "
                                                          "rotate270, flip_h, flip_v.")] = None,
         pad: Annotated[float, Field(description="Expand the region by this fraction on each side (context).")] = 0.0,
         grid: Annotated[str | None, Field(description="Grid used for cell names if different from look().")] = None,
         max_edge: MaxEdge = None) -> list:
    """Crop a region at NATIVE resolution (small regions are upscaled up to 6x) with rulers in original pixel
    coordinates. Use to read small text, inspect UI details, count items, or check fine structure."""
    from . import image_tools

    return image_tools.zoom(ref, region, enhance=enhance, pad=pad, grid=grid, max_edge=max_edge)


@mcp.tool(annotations=RO, structured_output=False)
def scan(ref: Ref, region: OptRegion = None,
         tile: Annotated[int, Field(description="Tile size in original px (256-2048).")] = 1024,
         overlap: Annotated[float, Field(description="Tile overlap fraction.")] = 0.1,
         page: Annotated[int, Field(description="Page of tiles to return (see 'page' in the result).")] = 1,
         per_page: Annotated[int, Field(description="Tiles per call (max 8).")] = 6) -> list:
    """Sweep a large image (or a region) systematically in overlapping native-resolution tiles, a page at a
    time. Use for exhaustive reading or searching for small things in big screenshots, maps, documents."""
    from . import image_tools

    return image_tools.scan(ref, region=region, tile=tile, overlap=overlap, page=page, per_page=per_page)


@mcp.tool(annotations=RO, structured_output=False)
def pixels(ref: Ref,
           points: Annotated[str | None, Field(description="Points 'x,y; x,y; ...' (px or fractions).")] = None,
           region: OptRegion = None,
           k: Annotated[int, Field(description="Number of dominant colours for a region.")] = 5) -> list:
    """Exact colour values: RGB/hex at points, or mean/std and dominant colours (k-means) of a region."""
    from . import image_tools

    return image_tools.pixels(ref, points=points, region=region, k=k)


@mcp.tool(annotations=RO, structured_output=False)
def annotate(ref: Ref,
             shapes: Annotated[list[dict[str, Any]], Field(description=(
                 "Shapes in ORIGINAL px: {'box':[x0,y0,x1,y1]} | {'point':[x,y]} | {'line':[x0,y0,x1,y1]} (arrow) | "
                 "{'polygon':[[x,y],...]}; optional 'label' and 'color':[r,g,b]."))],
             region: Annotated[str | None, Field(description="'auto' (fit the shapes), 'all', or a region.")] = "auto",
             max_edge: MaxEdge = None) -> list:
    """Draw boxes/points/arrows on the image to VERIFY your own localisation (or to show the user). If a mark
    is not exactly on its target, correct the coordinates and annotate again."""
    from . import image_tools

    return image_tools.annotate(ref, shapes, region=region, max_edge=max_edge)


@mcp.tool(annotations=RO, structured_output=False)
def compare(ref_a: Ref, ref_b: Ref, region: OptRegion = None,
            threshold: Annotated[int, Field(description="Per-pixel difference (0-255) counted as changed.")] = 24,
            max_regions: int = 12, max_edge: MaxEdge = None) -> list:
    """Pixel-accurate visual diff of two images/frames: changed %, PSNR, numbered changed regions drawn on
    both sides. Use for UI regressions, before/after states, spotting subtle differences."""
    from . import image_tools

    return image_tools.compare(ref_a, ref_b, region=region, threshold=threshold, max_regions=max_regions,
                               max_edge=max_edge)


@mcp.tool(annotations=RO, structured_output=False)
def locate(ref: Ref,
           template: Annotated[str, Field(description="Image of the thing to find (icon, button, logo...).")],
           template_region: Annotated[str | None, Field(description="Crop the template from its image first.")] = None,
           region: OptRegion = None,
           threshold: Annotated[float, Field(description="Match score 0-1 (lower finds more, less exact).")] = 0.8,
           max_results: int = 10, multiscale: bool = True, max_edge: MaxEdge = None) -> list:
    """Find every occurrence of a template image (icon, button, glyph) inside an image or frame, with exact
    boxes and scores (multi-scale normalised cross-correlation)."""
    from . import image_tools

    return image_tools.locate(ref, template, template_region=template_region, region=region, threshold=threshold,
                              max_results=max_results, multiscale=multiscale, max_edge=max_edge)


@mcp.tool(annotations=RO, structured_output=False)
def blobs(ref: Ref,
          color: Annotated[str | None, Field(description="Target colour '#rrggbb', 'r,g,b' or a name ('red'). "
                                                         "Read exact colours with pixels() first.")] = None,
          mode: Annotated[str, Field(description="color | dark (ink on light bg) | light.")] = "color",
          region: OptRegion = None,
          tolerance: Annotated[float, Field(description="Max CIELAB distance to the colour (15 strict - 45 "
                                                        "loose).")] = 30.0,
          min_area: Annotated[int, Field(description="Ignore blobs smaller than this many px.")] = 6,
          max_area: int | None = None,
          separate: Annotated[bool, Field(description="Split touching round blobs.")] = False,
          max_results: int = 300, max_edge: MaxEdge = None) -> list:
    """EXACT counting/locating of flat-coloured things (chart markers, dots, status icons, badges, highlighted
    cells, ink marks): connected pixel regions of a colour, with count, centres, boxes and a numbered overlay.
    Deterministic - prefer it over eyeballing whenever the targets share a distinct colour."""
    from . import image_tools

    return image_tools.blobs(ref, color=color, mode=mode, region=region, tolerance=tolerance, min_area=min_area,
                             max_area=max_area, separate=separate, max_results=max_results, max_edge=max_edge)


# ---------------------------------------------------------------- video

@mcp.tool(annotations=RO, structured_output=False)
def video_info(path: VideoPath) -> list:
    """Video metadata: duration, fps, resolution, codec, rotation, audio. Fast; no decoding of the whole file."""
    from . import video_tools

    return video_tools.video_info(path)


@mcp.tool(annotations=RO, structured_output=False)
def video_storyboard(path: VideoPath,
                     start: Annotated[Time | None, Field(description="Window start (seconds or MM:SS).")] = None,
                     end: Annotated[Time | None, Field(description="Window end (seconds or MM:SS).")] = None,
                     frames: Annotated[int, Field(description="Frames on the sheet (1-36).")] = 16,
                     mode: Annotated[str, Field(description="auto|hybrid (shots+changes+gaps), scenes, changes, "
                                                            "uniform.")] = "auto",
                     cols: int | None = None, max_edge: MaxEdge = None, ctx: Context | None = None) -> list:
    """Watch a video: one timestamped contact sheet of representative frames (one per shot, plus frames right
    after visual changes), with shot boundaries and a frame ref for each cell. First call per video analyses it
    (cached). Narrow start/end to zoom in time."""
    from . import video_tools

    return video_tools.video_storyboard(path, start=start, end=end, frames=frames, mode=mode, cols=cols,
                                        max_edge=max_edge, progress=_progress(ctx))


@mcp.tool(annotations=RO, structured_output=False)
def video_frames(path: VideoPath,
                 times: Annotated[list[Time] | None, Field(description="Exact timestamps to grab.")] = None,
                 start: Annotated[Time | None, Field(description="Span start (with count or fps).")] = None,
                 end: Annotated[Time | None, Field(description="Span end.")] = None,
                 count: Annotated[int | None, Field(description="Evenly spaced frames in the span (max 36).")] = None,
                 fps: Annotated[float | None, Field(description="Or: sample the span at this rate.")] = None,
                 layout: Annotated[str, Field(description="auto | sheet | separate (full-size images).")] = "auto",
                 cols: int | None = None, max_edge: MaxEdge = None) -> list:
    """Dense look at part of a video: frames at exact times, or N frames / an fps over a span, as a labelled
    sheet (or separate full-size images). Each frame gets a ref usable with zoom/ocr/compare."""
    from . import video_tools

    return video_tools.video_frames(path, times=times, start=start, end=end, count=count, fps=fps,
                                    layout=layout, cols=cols, max_edge=max_edge)


@mcp.tool(annotations=RO, structured_output=False)
def video_changes(path: VideoPath,
                  start: Annotated[Time | None, Field(description="Window start.")] = None,
                  end: Annotated[Time | None, Field(description="Window end.")] = None,
                  sensitivity: Annotated[str, Field(description="high (typed characters) | medium | low "
                                                                "(dialogs/pages only).")] = "medium",
                  region: Annotated[str | None, Field(description="Only watch this area 'x0,y0,x1,y1' "
                                                                  "(e.g. one UI panel).")] = None,
                  max_events: int = 40, sheet: bool = True, max_edge: MaxEdge = None,
                  ctx: Context | None = None) -> list:
    """Timeline of visual change events (when, where, how big) with before/after frame refs and a sheet of the
    state after each change. Ideal for screen recordings, UI bug videos, CCTV-like footage."""
    from . import video_tools

    return video_tools.video_changes(path, start=start, end=end, sensitivity=sensitivity, region=region,
                                     max_events=max_events, sheet=sheet, max_edge=max_edge, progress=_progress(ctx))


def run() -> None:
    import os
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore")
    from . import models_tools

    models_tools.register(mcp, RO, _progress)
    _mcp.run()
