"""Three-strength guided denoising controlled by gain, noise, and reliable edges."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EngineConfig
from .diagnose import luminance
from .recoverability import enforce_gain
from .types import DenoiseResult, DiagnosisResult, RecoveryResult


def denoise(image: np.ndarray, diagnosis: DiagnosisResult, recovery: RecoveryResult, cfg: EngineConfig, *,
            recovery_enabled: bool = True, uniform: bool = False) -> DenoiseResult:
    """Denoise RGB via YCrCb, then enforce confidence limits on the actual RGB result."""
    c = cfg["real"]["denoise"]
    epsilon = cfg["real"]["epsilon"]
    y = luminance(image, cfg)
    if diagnosis.raw["already_bright"] or diagnosis.raw["noise_sigma"] < c["minimum_sigma"]:
        result = enforce_gain(image, diagnosis.luminance, recovery, cfg, output_y=y) if recovery_enabled else image
        gain = luminance(result, cfg) / np.maximum(diagnosis.luminance, epsilon)
        return DenoiseResult(result, gain, np.zeros_like(diagnosis.luminance))
    else:
        gain = y / np.maximum(diagnosis.luminance, epsilon)
        strength = c["k"] * gain * diagnosis.noise / c["sigma_reference"]
        if recovery_enabled:
            strength += c["confidence_boost"] * (1 - recovery.soft)
        strength = np.clip(strength, 0, 1)
        if uniform:
            strength.fill(c["uniform_strength"])
        gx = cv2.Sobel(diagnosis.luminance, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(diagnosis.luminance, cv2.CV_32F, 0, 1)
        reliable = (cv2.magnitude(gx, gy) > c["edge_threshold"]) & (diagnosis.snr >= cfg["real"]["recoverability"]["snr_high"])
        if not uniform:
            strength[reliable] *= c["edge_strength_fraction"]
        source = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        mixed = np.zeros_like(source)
        coordinate = strength * 2
        for index, filter_epsilon in enumerate(c["epsilons"]):
            weight = np.maximum(1 - np.abs(coordinate - index), 0)
            scale = c["filter_scale"] if min(image.shape[:2]) * c["filter_scale"] >= 1 else 1.0
            filtered = cv2.ximgproc.guidedFilter(diagnosis.luminance, source, c["radius"], filter_epsilon, scale=scale)
            cv2.add(mixed, cv2.multiply(filtered, cv2.merge([weight] * 3)), dst=mixed)
        channels, original = list(cv2.split(mixed)), cv2.split(source)
        for index in (1, 2):
            channels[index] = cv2.addWeighted(channels[index], c["chroma_strength"], original[index], 1 - c["chroma_strength"], 0)
        mixed = cv2.merge(channels)
        result = cv2.cvtColor(mixed, cv2.COLOR_YCrCb2RGB)
    result = enforce_gain(result, diagnosis.luminance, recovery, cfg) if recovery_enabled else np.clip(result, 0, 1)
    gain = luminance(result, cfg) / np.maximum(diagnosis.luminance, epsilon)
    return DenoiseResult(result, gain, strength)
