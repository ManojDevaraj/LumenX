"""Highlight-protected Retinex with partial CLAHE and conservative cast correction."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EngineConfig
from .diagnose import channel_max
from .types import ClassicalResult, DiagnosisResult, EnhanceParams


def choose_params(diagnosis: DiagnosisResult, cfg: EngineConfig) -> EnhanceParams:
    """Adapt to measured severity and cast, with an explicit already-bright identity."""
    c = cfg["real"]["classical"]
    if diagnosis.raw["already_bright"] or diagnosis.raw["empty_visible_area"]:
        return EnhanceParams(1.0, 0.0, 0.0)
    color = min(c["color_max_strength"], diagnosis.raw["cast_magnitude"] / c["color_scale"] * c["color_max_strength"])
    if diagnosis.raw["cast"] == "none":
        color = 0.0
    clahe = c["clahe_strength"] if min(diagnosis.luminance.shape) >= c["minimum_clahe_side"] else 0.0
    return EnhanceParams(float(c["gamma"][diagnosis.raw["severity"]]), float(clahe), float(color))


def enhance_classical(image: np.ndarray, diagnosis: DiagnosisResult, cfg: EngineConfig, *,
                      params: EnhanceParams | None = None, color_enabled: bool = True) -> ClassicalResult:
    """Decompose/recompose light, protect highlights, then apply partial Lab changes."""
    c = cfg["real"]["classical"]
    epsilon = cfg["real"]["epsilon"]
    from dataclasses import replace

    if params is None or diagnosis.raw["already_bright"] or diagnosis.raw["empty_visible_area"]:
        params = choose_params(diagnosis, cfg)
    if not color_enabled:
        params = replace(params, color_strength=0.0)
    light = diagnosis.illumination
    light_rgb = cv2.merge([np.maximum(light, epsilon)] * 3)
    reflectance = cv2.divide(image, light_rgb)
    if image.size == 3:
        reflectance = np.clip(reflectance, 0, 1)
    else:
        reflectance = cv2.min(reflectance, (1, 1, 1, 1))
    if params.gamma == 1:
        return ClassicalResult(image.copy(), reflectance, light.copy(), params)
    adjusted = light ** params.gamma
    highlight = np.clip((light - c["highlight_start"]) / (1 - c["highlight_start"]), 0, 1)
    adjusted += (light - adjusted) * highlight
    enhanced = cv2.multiply(reflectance, cv2.merge([adjusted] * 3))
    if params.clahe_strength or params.color_strength:
        lab8 = cv2.cvtColor(cv2.convertScaleAbs(enhanced, alpha=255), cv2.COLOR_RGB2Lab)
        channels = list(cv2.split(lab8))
        if params.clahe_strength:
            clahe = cv2.createCLAHE(c["clahe_clip"], (c["clahe_tiles"], c["clahe_tiles"]))
            equalized = clahe.apply(channels[0])
            channels[0] = cv2.addWeighted(channels[0], 1 - params.clahe_strength, equalized, params.clahe_strength, 0)
        if params.color_strength:
            for index, key in ((1, "cast_a"), (2, "cast_b")):
                offset = params.color_strength * diagnosis.raw[key]
                if channels[index].size <= 4:
                    channels[index] = np.rint(np.clip(channels[index].astype(np.float32) - offset, 0, 255)).astype(np.uint8)
                else:
                    channels[index] = cv2.add(channels[index], -offset)
        enhanced = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_Lab2RGB).astype(np.float32) / 255
    # Any directly recorded channel highlight also receives the safe input blend.
    # A native-resolution maximum proves that no highlight blend is needed.
    # This shortcut cannot miss a tiny lamp averaged away by the diagnosis proxy.
    if cv2.minMaxLoc(image.reshape(-1))[1] > c["highlight_start"]:
        protect = np.clip((channel_max(image) - c["highlight_start"]) / (1 - c["highlight_start"]), 0, 1)
        enhanced = cv2.add(enhanced, cv2.multiply(cv2.subtract(image, enhanced), cv2.merge([protect] * 3)))
    # CLAHE's quantised Lab black point is additive.  Pixels with no recorded
    # signal and no local variation must remain no-signal rather than acquire a
    # synthetic pedestal.
    source_y = np.max(image, axis=2)
    mean = cv2.boxFilter(source_y, cv2.CV_32F, (3, 3))
    variance = np.maximum(cv2.boxFilter(source_y * source_y, cv2.CV_32F, (3, 3)) - mean * mean, 0)
    empty = (source_y <= epsilon) & (variance <= cfg["real"]["recoverability"]["black_variance"])
    enhanced[empty] = 0
    enhanced = np.clip(enhanced, 0, 1) if enhanced.size == 3 else cv2.min(cv2.max(enhanced, 0), (1, 1, 1, 1))
    return ClassicalResult(enhanced, reflectance, adjusted, params)
