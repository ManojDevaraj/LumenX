"""Memory-bounded CIDNet inference and independently testable cosine tile blending."""

from __future__ import annotations

from collections.abc import Callable
import threading

import cv2
import numpy as np

from ..contracts import Cancelled, EngineError


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Cover an axis exactly; never add an empty trailing tile."""
    if length < 1 or tile < 1 or not 0 <= overlap < tile:
        raise ValueError("Positive geometry and 0 <= overlap < tile are required.")
    last = max(0, length - tile)
    starts = list(range(0, last + 1, tile - overlap))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _axis_weights(length: int, overlap: int, before: bool, after: bool) -> np.ndarray:
    """Use cosine shoulders only at internal borders; outer image edges stay nonzero."""
    weight = np.ones(length, np.float32)
    count = min(overlap, length)
    if count:
        ramp = ((1 - np.cos(np.linspace(0, np.pi, count, dtype=np.float32))) / 2).astype(np.float32)
        if before:
            weight[:count] *= ramp
        if after:
            weight[-count:] *= ramp[::-1]
    return weight


def pad_stride(image: np.ndarray, stride: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Replicate-pad to a network stride, including singleton axes."""
    if stride < 1:
        raise ValueError("Network stride must be positive.")
    height, width = image.shape[:2]
    bottom, right = (-height) % stride, (-width) % stride
    return cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_REPLICATE), (height, width)


def tiled_inference(image: np.ndarray, infer: Callable[[np.ndarray], np.ndarray], *,
                    tile: int = 512, overlap: int = 32, stride: int = 8,
                    cancel: threading.Event | None = None) -> np.ndarray:
    """Run one tile at a time, retain accumulation on CPU, and normalize feather weights."""
    height, width = image.shape[:2]
    rows, columns = tile_starts(height, tile, overlap), tile_starts(width, tile, overlap)
    output = np.zeros_like(image, dtype=np.float32)
    denominator = np.zeros((height, width), np.float32)
    for top in rows:
        for left in columns:
            if cancel is not None and cancel.is_set():
                raise Cancelled("Enhancement cancelled between model tiles.")
            patch = np.ascontiguousarray(image[top:top + tile, left:left + tile])
            padded, shape = pad_stride(patch, stride)
            prediction = infer(padded)
            if prediction.shape != padded.shape or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
                raise EngineError("Model tile output has invalid shape, dtype or finite range.")
            h, w = shape
            wy = _axis_weights(h, overlap, top > 0, top + h < height)
            wx = _axis_weights(w, overlap, left > 0, left + w < width)
            weights = wy[:, None] * wx[None, :]
            output[top:top + h, left:left + w] += prediction[:h, :w] * weights[..., None]
            denominator[top:top + h, left:left + w] += weights
            del prediction, padded, patch
    if not np.all(denominator > 0):
        raise EngineError("Tiling left an uncovered pixel.")
    output /= denominator[..., None]
    return np.ascontiguousarray(np.clip(output, 0, 1))


class CIDNetRunner:
    """Thin model-specific adapter; registry owns model lifecycle and placement."""

    def __init__(self, runner: object, cfg: object) -> None:
        """Accept the shared tensor runner and immutable configuration."""
        self.runner = runner
        self.cfg = cfg

    def infer(self, image: np.ndarray, cancel: threading.Event | None = None) -> np.ndarray:
        """Tile above the configured pixel threshold; preserve native geometry."""
        c = self.cfg["models"]
        if image.shape[0] * image.shape[1] > c["tile_above_pixels"]:
            return tiled_inference(image, self.runner.infer, tile=c["tile"], overlap=c["overlap"],
                                   stride=c["stride"], cancel=cancel)
        if cancel is not None and cancel.is_set():
            raise Cancelled("Enhancement cancelled before model inference.")
        padded, (height, width) = pad_stride(image, c["stride"])
        prediction = self.runner.infer(padded)
        return np.ascontiguousarray(prediction[:height, :width])
