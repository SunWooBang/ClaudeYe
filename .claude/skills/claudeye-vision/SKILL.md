---
name: claudeye-vision
description: Precise image and video perception with the ClaudeYe MCP tools. Use whenever a task involves looking at screenshots, UI, photos, charts, scanned documents or diagrams, or any video / screen recording (you cannot watch video natively) - reading small text, counting, locating or measuring things, comparing images, finding UI elements, understanding what happens in a video, or debugging from a screen recording.
---

# ClaudeYe perception protocol

Your own view of an image is downscaled (detail smaller than a few pixels is lost), you are unreliable at
exact counts and pixel coordinates, and you cannot see video at all. The ClaudeYe tools (`mcp__claudeye__*`)
close these gaps. Gather visual evidence with tools first, then conclude.

## Ground rules

1. **Look, then zoom.** `look(ref)` returns a gridded overview (columns A.., rows 1..) with rulers in ORIGINAL
   pixels. Before reading small text, counting, or judging fine detail, `zoom(ref, "C3" | "B2:D4" | "x0,y0,x1,y1")`.
   Zoom again inside the zoom until glyphs are unambiguous. Faint/blurry: `enhance="sharpen,clahe"`;
   low contrast: `invert` or `threshold`; sideways/upside-down: `rotate90/180/270`.
2. **Never guess text.** If a zoom is not crisp, run `ocr(ref, region)`. Lines marked `verify:true`, and any
   ID, number, code or name that matters, must be confirmed by zooming into the returned box - the image is
   the ground truth (OCR drops spaces and confuses similar glyphs).
3. **Count with tools, not by eye.** Flat-coloured marks (dots, markers, badges, icons): read the colour with
   `pixels(ref, points="x,y")`, then `blobs(ref, color="#rrggbb", region=...)` - exact. Natural objects: `detect(ref,
   "person, car")`, then check the overlay (adjust `threshold`, `tile="on"` for small objects in big images).
   Up to ~10 items: zoom and count, then confirm with `annotate`.
4. **Coordinates come from the rulers.** Every view labels ORIGINAL pixel coordinates. Whenever you state a
   location or click target, verify it with `annotate(ref, [{"box": [x0,y0,x1,y1]}])` or `{"point": [x,y]}` and
   correct until the mark sits exactly on the target.
5. **Finding UI elements:** by label `ocr(ref, query="Save")`; by appearance `locate(ref, template)`; by
   description `ask_vision(q, [ref], ground="the settings gear icon")` - then verify with `annotate`.
6. **Comparing:** before/after screenshots or frames -> `compare(a, b)` gives exact changed regions; then zoom
   into each region in both images before describing the difference.
7. **Masks, areas, depth:** `segment(ref, prompt | points | boxes)` for exact outlines and area %; `depth(ref)`
   for near/far reasoning in photos.
8. **Report evidence:** coordinates, timestamps, frame refs, quoted text. Say explicitly when evidence is weak.

## Video (build the picture, coarse to fine)

1. `video_info(path)` - duration, fps, resolution, audio.
2. `video_storyboard(path)` - one timestamped contact sheet: a frame per shot plus frames right after visual
   changes. Every frame has a ref `path@SECONDS`.
3. Audio present -> `transcribe(path)`: what is said and when (speech + frames beats frames alone).
4. Screen recording / UI bug -> `video_changes(path)`: every change event (when, where, how big) with
   before/after refs. `sensitivity="high"` catches typing; `region="x0,y0,x1,y1"` watches one panel.
5. Long video, looking for a moment -> `video_search(path, "short english description")` -> timestamps.
6. Drill down: `video_storyboard(path, start=, end=)` or `video_frames(path, start=, end=, count=|fps=)`;
   any frame is an image: `zoom("clip.mp4@12.5", region)`, `ocr("clip.mp4@12.5")`,
   `compare("clip.mp4@10.0", "clip.mp4@11.0")`.
7. Optional second opinion: `ask_video(path, question)` (local model, or Gemini if the user enabled cloud).
   Verify every timestamp it cites with `video_frames`.

## Budget

Each returned image costs roughly 0.5-3k tokens and a single tool result is capped (~25k tokens). Prefer
targeted zooms to large views and one contact sheet to many single frames. For long videos or many images,
delegate to the `vision-analyst` subagent and work from its text report.

## Recipes

- **Dense table / dashboard:** look -> zoom row blocks -> `ocr(region)` -> spot-check key numbers with zoom.
- **Chart values:** zoom the axes to read tick labels, zoom the series, convert pixels to values with the
  rulers; `blobs()` for markers; state the precision of each reading.
- **Bug from a screen recording:** video_info -> `video_changes(sensitivity="high")` -> storyboard ->
  transcribe (if audio) -> `compare(before_ref, after_ref)` for the key event -> `ocr` the error text ->
  steps to reproduce, each with a timestamp.
- **"Where do I click?"** `ocr(ref, query=label)` or `locate(ref, template)` -> `annotate` a point -> give pixel
  coordinates.
- **Counting in a photo:** `detect(ref, "...")` -> review the overlay -> adjust -> report the count with the
  overlay-based caveats.

If a tool reports a missing backend, call `vision_status()` and pass the install hint to the user.
