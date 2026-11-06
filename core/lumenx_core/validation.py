"""Shared request validation for real and fake engines."""

from __future__ import annotations

import numpy as np

from .contracts import ABLATION_FLAGS, EnhanceRequest, InvalidRequest

_MAX_IMAGE_PIXELS = 12_000_000


def validate_request(req: EnhanceRequest) -> None:
    """Raise a contract error before the engine yields any events."""

    if not isinstance(req, EnhanceRequest):
        raise InvalidRequest("invalid_request", "request must be an EnhanceRequest")
    image = req.image
    if not isinstance(image, np.ndarray):
        raise InvalidRequest("invalid_image", "image must be a NumPy array")
    if image.dtype != np.float32:
        raise InvalidRequest("invalid_dtype", "image must have dtype float32")
    if image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
        raise InvalidRequest("invalid_shape", "image must have non-empty HxWx3 RGB shape")
    if image.shape[0] * image.shape[1] > _MAX_IMAGE_PIXELS:
        raise InvalidRequest("image_too_large", "processed image must be at most 12 megapixels")
    if not image.flags.c_contiguous:
        raise InvalidRequest("invalid_layout", "image must be C-contiguous")
    if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise InvalidRequest("invalid_range", "image must contain finite values in [0, 1]")
    if req.mode not in ("hybrid", "classical", "dl", "zerodce"):
        raise InvalidRequest("unknown_mode", f"unsupported mode: {req.mode!r}")
    if not isinstance(req.original_size, tuple) or len(req.original_size) != 2:
        raise InvalidRequest("invalid_original_size", "original_size must be (width, height)")
    if any(type(value) is not int or value <= 0 for value in req.original_size):
        raise InvalidRequest("invalid_original_size", "original_size values must be positive integers")
    if type(req.seed) is not int:
        raise InvalidRequest("invalid_seed", "seed must be an integer")
    if not isinstance(req.ablate, frozenset):
        raise InvalidRequest("invalid_ablation", "ablate must be a frozenset")
    unknown = req.ablate - ABLATION_FLAGS
    if unknown:
        raise InvalidRequest(
            "unknown_ablation", f"unknown ablation flags: {sorted(unknown, key=str)}"
        )
    if req.ablate and req.mode != "hybrid":
        raise InvalidRequest("invalid_ablation", "ablation flags are only valid in hybrid mode")
    if {"no_dl", "no_classical"}.issubset(req.ablate):
        raise InvalidRequest(
            "contradictory_ablation", "at least one enhancement path must remain enabled"
        )
    if req.alpha is not None:
        alpha = req.alpha
        if not isinstance(alpha, np.ndarray) or alpha.dtype != np.float32:
            raise InvalidRequest("invalid_alpha", "alpha must be a float32 NumPy array")
        if alpha.shape != image.shape[:2]:
            raise InvalidRequest("invalid_alpha", "alpha must match image height and width")
        if not np.isfinite(alpha).all() or float(alpha.min()) < 0.0 or float(alpha.max()) > 1.0:
            raise InvalidRequest("invalid_alpha", "alpha must contain finite values in [0, 1]")
