"""End-to-end MCP check: spawn the server over stdio exactly like Claude Code does, list tools,
call a few, and save any returned images.

usage: python scripts/mcp_smoke.py [--out DIR]
"""

from __future__ import annotations

import asyncio
import base64
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
CALLS = [
    ("look", {"ref": "samples/dashboard_4k.png"}),
    ("zoom", {"ref": "samples/dashboard_4k.png", "region": "A1", "enhance": "sharpen"}),
    ("pixels", {"ref": "samples/dashboard_4k.png", "points": "3326,1747; 100,100"}),
    ("blobs", {"ref": "samples/dashboard_4k.png", "color": "red", "region": "200,700,1800,1900"}),
    ("locate", {"ref": "samples/dashboard_4k.png", "template": "samples/star_icon.png"}),
    ("compare", {"ref_a": "samples/dashboard_4k.png", "ref_b": "samples/dashboard_4k_v2.png"}),
    ("annotate", {"ref": "samples/dashboard_4k.png", "shapes": [{"box": [3312, 1733, 3340, 1761], "label": "star"}]}),
    ("video_info", {"path": "samples/demo_speech.mp4"}),
    ("video_storyboard", {"path": "samples/demo.mp4", "frames": 8}),
    ("video_frames", {"path": "samples/demo.mp4", "start": 13, "end": 15, "count": 6}),
    ("video_changes", {"path": "samples/demo.mp4", "start": 10, "end": 18}),
    ("zoom", {"ref": "samples/demo.mp4@16.5", "region": "0.6,0.8,1,1"}),
    ("ocr", {"ref": "samples/demo.mp4@16.5", "query": "오류"}),
    ("transcribe", {"path": "samples/demo_speech.mp4", "language": "ko"}),
    ("video_search", {"path": "samples/demo.mp4", "query": "settings dialog", "top_k": 3}),
    ("vision_status", {}),
    ("look", {"ref": "does/not/exist.png"}),
    ("zoom", {"ref": "samples/dashboard_4k.png", "region": "Z99"}),
]


async def main(out: Path) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "claudeye"], cwd=str(ROOT))
    t0 = time.perf_counter()
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        print(f"initialize: {init.server_info.name} {init.server_info.version} in {time.perf_counter() - t0:.2f}s")
        print(f"instructions: {len(init.instructions or '')} chars")
        tools = await session.list_tools()
        print(f"{len(tools.tools)} tools:", ", ".join(t.name for t in tools.tools))
        out.mkdir(parents=True, exist_ok=True)
        for i, (name, args) in enumerate(CALLS):
            t1 = time.perf_counter()
            res = await session.call_tool(name, args)
            kinds = []
            for j, c in enumerate(res.content):
                if c.type == "text":
                    kinds.append(f"text({len(c.text)})")
                elif c.type == "image":
                    p = out / f"smoke_{i}_{name}_{j}.{c.mime_type.split('/')[-1]}"
                    p.write_bytes(base64.b64decode(c.data))
                    kinds.append(f"image({c.mime_type}, {p.stat().st_size // 1024}KB)")
            first = next((c.text for c in res.content if c.type == "text"), "")[:160].replace("\n", " | ")
            print(f"- {name:17s} error={res.is_error!s:5s} {time.perf_counter() - t1:5.2f}s {kinds}\n    {first}")
            if res.structured_content is not None:
                print("    ! structured_content present")


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "out"
    asyncio.run(main(out))
