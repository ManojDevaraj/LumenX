"""Confidence-aware classical/deep/safe fusion, smoothed on an input-guided proxy."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EngineConfig
from .diagnose import channel_max
from .types import DiagnosisResult, FusionResult, RecoveryResult


def fuse(image: np.ndarray, deep: np.ndarray | None, classical: np.ndarray | None,
         diagnosis: DiagnosisResult, recovery: RecoveryResult, cfg: EngineConfig, *,
         recovery_enabled: bool = True, alpha_fraction: float = 1.0,
         repair_mask: np.ndarray | None = None) -> FusionResult:
    """Smooth on a proxy, upsample, then enforce native confidence/highlight limits."""
    c = cfg["fusion"]
    epsilon = cfg["real"]["epsilon"]
    height, width = image.shape[:2]
    small_y = diagnosis.proxy_y
    size = (small_y.shape[1], small_y.shape[0])
    light = cv2.resize(diagnosis.illumination, size, interpolation=cv2.INTER_AREA)
    confidence = cv2.resize(recovery.soft, size, interpolation=cv2.INTER_AREA) if recovery_enabled else np.ones_like(light)
    safe_weight = np.clip((light - c["highlight_start"]) / (1 - c["highlight_start"]), 0, 1)
    alpha = c["alpha"] * alpha_fraction
    deep_weight = (1 / (1 + np.exp((light - c["dark_threshold"]) / c["dark_width"]))) * confidence * alpha * (1 - safe_weight)
    if deep is None:
        deep_weight.fill(0)
    classical_weight = np.maximum(1 - safe_weight - deep_weight, 0)
    if classical is None:
        safe_weight += classical_weight
        classical_weight.fill(0)
    small_weights = cv2.merge([deep_weight, classical_weight, safe_weight])
    small_weights = cv2.ximgproc.guidedFilter(small_y, small_weights, c["guide_radius"], c["guide_epsilon"])
    weights = cv2.resize(small_weights, (width, height), interpolation=cv2.INTER_LINEAR)
    np.maximum(weights, 0, out=weights)
    total = np.sum(weights, axis=2)
    empty = total <= epsilon
    weights /= np.maximum(total, epsilon)[..., None]
    weights[empty] = [0, 0, 1]
    if deep is None:
        weights[..., 1] += weights[..., 0]
        weights[..., 0] = 0
    if classical is None:
        weights[..., 2] += weights[..., 1]
        weights[..., 1] = 0
    # Guided smoothing cannot restore deep contribution beyond the local evidence cap.
    maximum_deep = alpha * recovery.soft if recovery_enabled else alpha
    limited = np.minimum(weights[..., 0], maximum_deep)
    difference = weights[..., 0] - limited
    weights[..., 0] = limited
    weights[..., 1 if classical is not None else 2] += difference
    native_highlight = np.clip((channel_max(image) - c["highlight_start"]) / (1 - c["highlight_start"]), 0, 1)
    if repair_mask is not None:
        native_highlight = np.maximum(native_highlight, repair_mask * cfg["repair"]["highlight_boost"])
    weights[..., :2] *= (1 - native_highlight[..., None])
    weights[..., 2] = 1 - weights[..., 0] - weights[..., 1]
    np.maximum(weights, 0, out=weights)
    weights /= np.maximum(np.sum(weights, axis=2, keepdims=True), epsilon)
    safe = image ** c["safe_gamma"]
    output = safe * weights[..., 2:3]
    if deep is not None:
        output += deep * weights[..., 0:1]
    if classical is not None:
        output += classical * weights[..., 1:2]
    return FusionResult(np.ascontiguousarray(np.clip(output, 0, 1), dtype=np.float32), weights)
