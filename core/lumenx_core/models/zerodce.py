"""Zero-DCE++ adapter: shared model for live, direct baseline and still fallback."""

from __future__ import annotations

import threading

import numpy as np

from ..contracts import Cancelled
from .cidnet import pad_stride


class ZeroDCERunner:
    """Pad for configured curve-network scale without altering image resolution."""

    def __init__(self, runner: object, scale: int) -> None:
        """Use the already-loaded shared tensor runner, not a per-frame model."""
        self.runner = runner
        self.scale = scale

    def infer(self, image: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        """Return native-size float RGB; the tensor runner extracts enhanced output."""
        if cancel is not None and cancel.is_set():
            raise Cancelled("Enhancement cancelled before Zero-DCE++.")
        padded, (height, width) = pad_stride(image, self.scale)
        prediction = self.runner.infer(padded)
        return np.ascontiguousarray(prediction[:height, :width])
