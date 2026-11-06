"""Illumination- and confidence-aware combination of restoration paths."""

from __future__ import annotations

import cv2
import numpy as np

from .diagnose import luminance
from .types import DiagnosisResult, RecoveryResult


def _smooth_weight(guide: np.ndarray, weight: np.ndarray, radius: int, eps: float) -> np.ndarray:
    if min(guide.shape) <= 1:
        return weight.astype(np.float32, copy=True)
    radius = max(1, min(int(radius), (min(guide.shape) - 1) // 2))
    return cv2.ximgproc.guidedFilter(
        guide=guide.astype(np.float32, copy=False),
        src=weight.astype(np.float32, copy=False),
        radius=radius,
        eps=float(eps),
    ).astype(np.float32, copy=False)


def fuse_paths(
    image: np.ndarray,
    classical: np.ndarray,
    deep: np.ndarray,
    diagnosis: DiagnosisResult,
    recovery: RecoveryResult,
    cfg: dict[str, dict[str, float | int]],
    *,
    params: dict[str, float] | None = None,
    use_recoverability: bool = True,
    use_classical: bool = True,
    use_deep: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fused RGB output and normalized ``(deep, classical, safe)`` weights."""

    values = cfg["fusion"]
    alpha = float(params.get("fusion_alpha", values["alpha_default"])) if params else float(
        values["alpha_default"]
    )
    illumination = diagnosis.illumination
    evidence = recovery.soft_score if use_recoverability else np.ones_like(illumination)
    darkness = 1.0 / (1.0 + np.exp(np.clip(
        (illumination - float(values["dark_threshold"])) / float(values["dark_slope"]),
        -30.0,
        30.0,
    )))
    deep_weight = darkness * evidence * float(values["deep_weight_max"]) * np.clip(alpha, 0.0, 1.0)

    safe_start = float(values["safe_highlight_start"])
    safe_weight = np.clip(
        (illumination - safe_start) / max(1.0 - safe_start, 1e-6), 0.0, 1.0
    ) * float(values["safe_weight_max"])
    if use_recoverability:
        safe_weight = np.maximum(
            safe_weight,
            (1.0 - evidence) * float(values["weak_signal_safe_weight"]),
        )
    if not use_deep:
        deep_weight = np.zeros_like(deep_weight)
    if not use_classical:
        classical_weight = np.zeros_like(deep_weight)
    else:
        classical_weight = np.maximum(1.0 - deep_weight - safe_weight, 0.0)

    guide = luminance(image)
    radius = int(values["weight_guided_radius"])
    eps = float(values["weight_guided_eps"])
    smooth = np.stack(
        [
            _smooth_weight(guide, weight, radius, eps)
            for weight in (deep_weight, classical_weight, safe_weight)
        ],
        axis=-1,
    )
    smooth = np.clip(np.nan_to_num(smooth, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    denominator = np.sum(smooth, axis=-1, keepdims=True)
    missing = denominator[..., 0] <= 1e-8
    smooth[missing] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    denominator = np.sum(smooth, axis=-1, keepdims=True)
    weights = smooth / np.maximum(denominator, 1e-8)

    safe = np.power(np.clip(image, 0.0, 1.0), float(values["safe_gamma"])).astype(np.float32)
    highlight_mask = np.clip(
        (illumination - safe_start) / max(1.0 - safe_start, 1e-6), 0.0, 1.0
    )[..., None]
    safe = safe * (1.0 - highlight_mask) + image * highlight_mask
    output = (
        deep * weights[..., 0:1]
        + classical * weights[..., 1:2]
        + safe * weights[..., 2:3]
    )
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False), weights.astype(
        np.float32, copy=False
    )

