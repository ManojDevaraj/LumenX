"""Public LumenX engine factories; v0.1.0 ships the CPU fake engine only."""

from __future__ import annotations

import cv2

from .contracts import Engine, EngineUnavailable

if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
    raise ImportError(
        "LumenX requires cv2.ximgproc.guidedFilter. Install the pinned "
        "opencv-contrib-python dependency and remove other OpenCV wheels."
    )

__version__ = "0.1.0"
__all__ = ["__version__", "create_engine", "create_fake_engine"]


def create_engine(config_path: str | None = None, device: str | None = None) -> Engine:
    """Refuse real inference until a release implementing RealEngine is installed."""
    raise EngineUnavailable(
        "RealEngine is not shipped in core-v0.1.0. "
        "Use create_fake_engine() explicitly for mock development."
    )


def create_fake_engine(stage_delay_ms: int = 0, seed: int = 0) -> Engine:
    """Create the deterministic CPU-only engine without loading model weights."""
    from .fake import FakeEngine

    return FakeEngine(stage_delay_ms=stage_delay_ms, seed=seed)
