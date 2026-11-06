"""Strict in-memory request validation shared by fake and future real engines."""

from __future__ import annotations

import threading

import numpy as np

from .contracts import ABLATION_FLAGS, EnhanceRequest, InvalidRequest


def validate_rgb(image: np.ndarray, *, floating: bool = True) -> None:
    """Validate image dtype, positive RGB shape, contiguity, size and value domain."""
    if not isinstance(image, np.ndarray):
        raise InvalidRequest("image_type", "Image must be a NumPy array.")
    dtype = np.float32 if floating else np.uint8
    if image.dtype != dtype:
        raise InvalidRequest("image_dtype", f"Image must have dtype {np.dtype(dtype)}.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise InvalidRequest("image_shape", "Image must have positive H x W x 3 RGB shape.")
    if floating and not image.flags.c_contiguous:
        raise InvalidRequest("image_contiguous", "Image must be C-contiguous.")
    if floating:
        if image.shape[0] * image.shape[1] > 12_000_000:
            raise InvalidRequest("image_size", "Image exceeds the contract's 12 MP limit.")
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise InvalidRequest("image_range", "Image must contain finite values in [0, 1].")
    elif max(image.shape[:2]) > 1280:
        raise InvalidRequest("frame_size", "Live frame long side must be at most 1280.")


def validate_request(req: EnhanceRequest) -> None:
    """Raise InvalidRequest before any processing event can be emitted."""
    if not isinstance(req, EnhanceRequest):
        raise InvalidRequest("request_type", "Expected EnhanceRequest.")
    validate_rgb(req.image)
    if req.mode not in ("hybrid", "classical", "dl", "zerodce"):
        raise InvalidRequest("mode", "Unknown enhancement mode.")
    if not isinstance(req.ablate, frozenset) or any(type(x) is not str for x in req.ablate):
        raise InvalidRequest("ablate_type", "Ablations must be a frozenset of strings.")
    if req.ablate - ABLATION_FLAGS:
        raise InvalidRequest("ablate_unknown", "Unknown ablation flag.")
    if req.mode != "hybrid" and req.ablate:
        raise InvalidRequest("ablate_mode", "Ablations are supported only in hybrid mode.")
    if "no_dl" in req.ablate and req.ablate & {"no_fusion", "no_classical"}:
        raise InvalidRequest("ablate_conflict", "Contradictory ablation flags.")
    if (
        not isinstance(req.original_size, tuple)
        or len(req.original_size) != 2
        or any(type(x) is not int or x < 1 for x in req.original_size)
    ):
        raise InvalidRequest("original_size", "Original size must be positive (width, height).")
    if type(req.seed) is not int:
        raise InvalidRequest("seed", "Seed must be an integer.")
    if req.cancel is not None and not isinstance(req.cancel, threading.Event):
        raise InvalidRequest("cancel", "Cancellation must be a threading.Event or None.")
    if req.alpha is not None:
        alpha = req.alpha
        if (
            not isinstance(alpha, np.ndarray)
            or alpha.dtype != np.float32
            or alpha.shape != req.image.shape[:2]
        ):
            raise InvalidRequest("alpha_shape", "Alpha must be float32 H x W matching the image.")
        if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
            raise InvalidRequest("alpha_range", "Alpha must contain finite values in [0, 1].")


def expected_stages(req: EnhanceRequest) -> list[str]:
    """Return the exact per-mode stage schedule from frozen contract section 5."""
    stages = ["diagnose", "recoverability"]
    if req.mode == "hybrid":
        if "no_tuner" not in req.ablate:
            stages.append("tuner")
        if "no_classical" not in req.ablate:
            stages.append("classical")
        if "no_dl" not in req.ablate:
            stages.append("deep")
        if "no_fusion" not in req.ablate:
            stages.append("fusion")
        stages.append("denoise")
    elif req.mode == "classical":
        stages.extend(["classical", "denoise"])
    else:
        stages.append("deep")
    return stages + ["guardrails"]
