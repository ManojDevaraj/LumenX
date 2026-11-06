"""Public factories for the real classical engine and explicitly selected CPU fake."""

from __future__ import annotations

import cv2

from .contracts import Engine

if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
    raise ImportError(
        "LumenX requires cv2.ximgproc.guidedFilter. Install the pinned "
        "opencv-contrib-python dependency and remove other OpenCV wheels."
    )

__version__ = "0.4.0"
__all__ = ["__version__", "create_engine", "create_fake_engine"]


def create_engine(config_path: str | None = None, device: str | None = None) -> Engine:
    """Create the real engine; no_tuner is explicit until full adaptive tuning ships."""
    from .engine import RealEngine

    return RealEngine(config_path=config_path, device=device)


def create_fake_engine(stage_delay_ms: int = 0, seed: int = 0) -> Engine:
    """Create the deterministic CPU-only engine without loading model weights."""
    from .fake import FakeEngine

    return FakeEngine(stage_delay_ms=stage_delay_ms, seed=seed)
