"""LumenX low-light restoration engine public API."""

from __future__ import annotations

from .contracts import (
    ABLATION_FLAGS,
    CONTRACT_VERSION,
    Cancelled,
    EnhanceRequest,
    EnhanceResult,
    Engine,
    EngineError,
    EngineUnavailable,
    InvalidRequest,
    MetricSuite,
    StageEvent,
    Visual,
)
from .fake import FakeEngine, create_fake_engine

__version__ = "0.4.0"


def create_engine(config_path: str | None = None, device: str | None = None) -> Engine:
    """Create the real engine, selecting CUDA when available and requested."""

    if device is not None and device not in ("auto", "cpu", "cuda") and not device.startswith("cuda:"):
        raise EngineUnavailable("device must be 'auto', 'cpu', 'cuda', or a CUDA device such as 'cuda:0'.")
    try:
        import cv2
    except ImportError as exc:
        raise EngineUnavailable(
            "OpenCV contrib is required for the real engine; install lumenx-core[vision]."
        ) from exc
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        raise EngineUnavailable(
            "cv2.ximgproc.guidedFilter is missing; install the opencv-contrib-python wheel."
        )
    from .config import load_config
    from .engine import RealEngine

    try:
        config = load_config(config_path)
    except RuntimeError as exc:
        raise EngineUnavailable(str(exc)) from exc
    return RealEngine(config, device=device)


__all__ = [
    "ABLATION_FLAGS",
    "CONTRACT_VERSION",
    "Cancelled",
    "EnhanceRequest",
    "EnhanceResult",
    "Engine",
    "EngineError",
    "EngineUnavailable",
    "FakeEngine",
    "InvalidRequest",
    "MetricSuite",
    "StageEvent",
    "Visual",
    "__version__",
    "create_engine",
    "create_fake_engine",
]
