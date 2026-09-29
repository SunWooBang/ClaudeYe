"""Shared parameter types for tool signatures (descriptions end up in the MCP JSON schema)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

Ref = Annotated[str, Field(description="Image file path, or a video frame as 'video.mp4@SECONDS' (also MM:SS.ms).")]
Region = Annotated[str, Field(description="Grid cell 'C3', cell range 'B2:D4', pixel box 'x0,y0,x1,y1', "
                                          "fractions '0.1,0.2,0.5,0.6', or a point 'x,y' (crops around it).")]
OptRegion = Annotated[str | None, Field(description="Optional region (same syntax as zoom); default whole image.")]
MaxEdge = Annotated[int | None, Field(description="Long edge of the returned image in px (default 1568, max 2000).")]
Time = float | str
OptTime = Annotated[Time | None, Field(description="Seconds or MM:SS(.ms).")]
VideoPath = Annotated[str, Field(description="Video file path.")]
