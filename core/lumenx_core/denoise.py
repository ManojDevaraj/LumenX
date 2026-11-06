"""Gain- and recoverability-aware guided smoothing."""

from __future__ import annotations

import cv2
import numpy as np

from .diagnose import luminance
from .types import DiagnosisResult, DenoiseResult, RecoveryResult


def _guided(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    radius = max(1, min(radius, (min(guide.shape) - 1) // 2))
    if min(guide.shape) <= 1:
        return source.astype(np.float32, copy=True)
    result = cv2.ximgproc.guidedFilter(
        guide=guide.astype(np.float32, copy=False),
        src=source.astype(np.float32, copy=False),
        radius=radius,
        eps=float(eps),
    )
    return np.asarray(result, dtype=np.float32)


def _noise_sigma(gray: np.ndarray) -> float:
    residual = gray - cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(residual))
    return float(np.median(np.abs(residual - median)) * 1.4826)


def denoise(
    image: np.ndarray,
    restored: np.ndarray,
    diagnosis: DiagnosisResult,
    recovery: RecoveryResult,
    cfg: dict[str, dict[str, float | int]],
    *,
    use_recoverability: bool = True,
    uniform: bool = False,
    enabled: bool = True,
    strength_multiplier: float = 1.0,
) -> DenoiseResult:
    """Blend light, medium, and strong guided luminance smoothing by local need."""

    values = cfg["denoise"]
    input_y = luminance(image)
    output_y = luminance(restored)
    gain = output_y / np.maximum(input_y, float(values["gain_floor"]))
    gain = np.clip(
        gain,
        float(values["gain_min_for_strength"]),
        float(values["gain_max"]),
    )
    strength = (
        float(values["strength_scale"])
        * float(np.clip(strength_multiplier, 0.0, 3.0))
        * gain
        * diagnosis.noise_map
        / max(float(values["reference_sigma"]), 1e-6)
    )
    if use_recoverability:
        strength += float(values["weak_signal_boost"]) * (
            recovery.classes.astype(np.float32) / 2.0
        )

    gx = cv2.Sobel(input_y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(input_y, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy)
    edge_scale = float(values["edge_scale"])
    reliable_edge = np.clip(edge / (edge + edge_scale), 0.0, 1.0)
    if use_recoverability:
        reliable_edge *= recovery.soft_score
    strength *= 1.0 - float(values["edge_protection"]) * reliable_edge
    if uniform:
        strength = np.full_like(strength, float(np.mean(strength, dtype=np.float64)))
    if not enabled:
        strength = np.zeros_like(strength)
    strength = np.clip(strength, 0.0, 1.0).astype(np.float32)

    light = _guided(
        input_y,
        output_y,
        int(values["guided_radius_light"]),
        float(values["guided_eps_light"]),
    )
    medium = _guided(
        input_y,
        output_y,
        int(values["guided_radius_medium"]),
        float(values["guided_eps_medium"]),
    )
    strong = _guided(
        input_y,
        output_y,
        int(values["guided_radius_strong"]),
        float(values["guided_eps_strong"]),
    )
    weight_strong = np.clip((strength - 0.66) / 0.34, 0.0, 1.0)
    weight_medium = (
        np.clip((strength - 0.22) / 0.44, 0.0, 1.0) * (1.0 - weight_strong)
    )
    weight_light = np.clip(1.0 - weight_strong - weight_medium, 0.0, 1.0)
    filtered_y = light * weight_light + medium * weight_medium + strong * weight_strong
    clean_y = output_y * (1.0 - strength) + filtered_y * strength

    chroma = restored - output_y[..., None]
    chroma_smoothed = np.stack(
        [cv2.GaussianBlur(chroma[..., channel], (3, 3), 0) for channel in range(3)],
        axis=-1,
    )
    chroma_weight = np.clip(
        strength * float(values["chroma_smoothing"]), 0.0, 1.0
    )[..., None]
    clean_chroma = chroma * (1.0 - chroma_weight) + chroma_smoothed * chroma_weight
    output = np.clip(clean_y[..., None] + clean_chroma, 0.0, 1.0).astype(np.float32)
    if use_recoverability:
        output = np.minimum(output, image + recovery.gain_limit[..., None])
    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
    output_y = luminance(output)
    gain = np.clip(
        output_y / np.maximum(input_y, float(values["gain_floor"])),
        0.0,
        float(values["gain_max"]),
    )
    return DenoiseResult(
        image=output,
        gain_map=np.asarray(gain, dtype=np.float32),
        strength_map=strength,
        noise_sigma_after=_noise_sigma(luminance(output)),
    )
