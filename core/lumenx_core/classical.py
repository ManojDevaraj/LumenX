"""Retinex, CLAHE, highlight protection, and gentle cast correction."""

from __future__ import annotations

import cv2
import numpy as np

from .types import ClassicalResult, DiagnosisResult


def restore_classical(
    image: np.ndarray,
    diagnosis: DiagnosisResult,
    cfg: dict[str, dict[str, float | int]],
    params: dict[str, float] | None = None,
    *,
    correct_color: bool = True,
) -> ClassicalResult:
    """Apply adaptive single-scale Retinex and Lab lightness equalization."""

    values = cfg["classical"]
    gray = (
        image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    ).astype(np.float32)
    dark_level = float(
        np.clip(1.0 - diagnosis.mean_luminance / float(values["dark_adaptation_scale"]), 0.0, 1.0)
    )
    gamma = float(values["gamma_max"]) - dark_level * (
        float(values["gamma_max"]) - float(values["gamma_min"])
    )
    if params is not None:
        gamma = float(params.get("gamma", gamma))
    illumination = np.maximum(diagnosis.illumination, float(values["illumination_floor"]))
    reflectance = np.clip(image / illumination[..., None], 0.0, 1.0)
    adjusted = np.power(illumination, gamma)

    highlight_threshold = float(values["highlight_threshold"])
    highlight_amount = np.clip(
        (illumination - highlight_threshold) / max(1.0 - highlight_threshold, 1e-6),
        0.0,
        1.0,
    )
    highlight_weight = highlight_amount * float(values["highlight_blend"])
    adjusted = adjusted * (1.0 - highlight_weight) + illumination * highlight_weight
    retinex = np.clip(reflectance * adjusted[..., None], 0.0, 1.0)

    rgb8 = np.rint(retinex * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    tile_size = int(values["clahe_tile_size"])
    clip_limit = float(values["clahe_clip_limit"])
    if params is not None:
        clip_limit = float(params.get("clahe_clip", clip_limit))
    if diagnosis.severity in ("extreme", "severe"):
        clip_limit *= 1.12
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    lab[..., 0] = clahe.apply(lab[..., 0])
    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    midtones = (gray > float(values["color_mid_low"])) & (gray < float(values["color_mid_high"]))
    if np.any(midtones):
        channel_means = np.mean(image[midtones], axis=0, dtype=np.float64).astype(np.float32)
    else:
        channel_means = np.mean(image, axis=(0, 1), dtype=np.float64).astype(np.float32)
    target = float(np.mean(channel_means))
    gains = target / np.maximum(channel_means, float(values["color_channel_floor"]))
    gains = np.clip(
        gains,
        float(values["color_gain_min"]),
        float(values["color_gain_max"]),
    )
    correction_strength = (
        float(values["color_correction_max"]) * diagnosis.color_cast_strength
        if correct_color
        else 0.0
    )
    if params is not None and correct_color:
        correction_strength = float(params.get("color_strength", correction_strength))
    corrected = equalized * (
        1.0 - correction_strength + correction_strength * gains[None, None, :]
    )
    restored = np.clip(corrected, 0.0, 1.0).astype(np.float32, copy=False)

    return ClassicalResult(
        image=restored,
        reflectance=reflectance.astype(np.float32, copy=False),
        adjusted_illumination=adjusted.astype(np.float32, copy=False),
        gamma=gamma,
        clahe_clip=clip_limit,
        color_strength=correction_strength,
    )
