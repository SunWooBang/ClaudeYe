"""Call a ClaudeYe tool function directly and save returned images (dev helper).

usage: python scripts/try_tool.py <tool> [positional ...] [key=value ...] [--out DIR]
  python scripts/try_tool.py look samples/dashboard_4k.png
  python scripts/try_tool.py zoom samples/dashboard_4k.png B2 enhance=sharpen
  python scripts/try_tool.py video_storyboard samples/demo.mp4 frames=12
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

from claudeye import image_tools, video_tools

TOOLS = {**{n: getattr(image_tools, n) for n in ("look", "zoom", "scan", "pixels", "annotate", "compare", "locate")},
         **{n: getattr(video_tools, n) for n in ("video_info", "video_storyboard", "video_frames", "video_changes")}}
try:
    from claudeye import models_tools  # noqa: F401

    TOOLS.update({n: getattr(models_tools, n) for n in models_tools.__all__})
except ImportError:
    pass


def _val(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main(argv: list[str]) -> None:
    out = Path("out")
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    name, *rest = argv
    args = [_val(a) if a[:1] in "[{" else a for a in rest if "=" not in a or a.startswith(("{", "["))]
    kwargs = {k: _val(v) for k, v in (a.split("=", 1) for a in rest if "=" in a and not a.startswith(("{", "[")))}
    t0 = time.perf_counter()
    result = TOOLS[name](*args, **kwargs)
    dt = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in result:
        if isinstance(item, str):
            print(item)
        else:
            c = item.to_image_content()
            ext = c.mime_type.split("/")[-1]
            p = out / f"{name}_{n}.{ext}"
            p.write_bytes(base64.b64decode(c.data))
            print(f"[image] {p}")
            n += 1
    print(f"[{dt:.2f}s]")


if __name__ == "__main__":
    main(sys.argv[1:])
