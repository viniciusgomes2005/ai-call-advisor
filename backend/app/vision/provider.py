from __future__ import annotations

import abc

from app.schemas import ScreenFrame, VisualAnalysis


class VisionProvider(abc.ABC):
    """Analyzes a captured screen frame into structured visual context.

    This is the seam a future multimodal/vision model plugs into
    (Screenshot -> VisionProvider -> VisualContext) without touching the
    capture/change-detection/assembly pipeline built in this V1. No model is
    wired in yet - see NullVisionProvider, the only implementation for now.
    """

    @abc.abstractmethod
    async def analyze_frame(self, image_bytes: bytes, frame: ScreenFrame) -> VisualAnalysis | None:
        """Return a VisualAnalysis for this frame, or None if nothing was produced
        (e.g. analysis disabled, provider unavailable, or - as with
        NullVisionProvider - no model is wired in at all)."""
        raise NotImplementedError


class NullVisionProvider(VisionProvider):
    """V1 default: performs no analysis. Keeps VisualContext's semantic fields
    (summary/visible_text/entities/systems/numbers) empty rather than inventing
    fake content - only a real VisionProvider should ever populate them."""

    async def analyze_frame(self, image_bytes: bytes, frame: ScreenFrame) -> VisualAnalysis | None:
        return None
