from __future__ import annotations

import base64
import binascii
from datetime import datetime

from app.schemas import ScreenFrame, VisualAnalysis, VisualContext, utc_now


_EXTENSION_BY_MIME_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/png": ".png",
}


def screen_frame_filename(sequence: int, mime_type: str) -> str:
    """Deterministic, predictable filename for a saved raw frame, e.g. frame_000001.jpg."""
    extension = _EXTENSION_BY_MIME_TYPE.get(mime_type.lower(), ".jpg")
    return f"frame_{sequence:06d}{extension}"


def decode_frame_data(data: str) -> bytes:
    """Decode the base64 payload of a screen.frame message. Raises ValueError (not
    a raw binascii.Error) so callers can catch one exception type for any malformed
    incoming frame payload."""
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 screen frame data: {exc}") from exc


def build_screen_frame(
    *,
    frame_id: str,
    timestamp: float,
    captured_at: datetime | None,
    mime_type: str,
    width: int,
    height: int,
    change_score: float | None,
) -> ScreenFrame:
    return ScreenFrame(
        id=frame_id,
        timestamp=timestamp,
        captured_at=captured_at or utc_now(),
        mime_type=mime_type,
        width=width,
        height=height,
        change_score=change_score,
    )


def build_visual_context(*, visual_id: str, frame: ScreenFrame, analysis: VisualAnalysis | None = None) -> VisualContext:
    """Build the VisualContext for an accepted frame. With no VisionProvider result
    (analysis=None, i.e. NullVisionProvider today), all semantic fields stay empty -
    never fabricated."""
    analysis = analysis or VisualAnalysis()
    return VisualContext(
        id=visual_id,
        frame_id=frame.id,
        timestamp=frame.timestamp,
        captured_at=frame.captured_at,
        summary=analysis.summary,
        visible_text=list(analysis.visible_text),
        entities=list(analysis.entities),
        systems=list(analysis.systems),
        numbers=list(analysis.numbers),
    )
