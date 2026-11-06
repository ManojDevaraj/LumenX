"""Input-based confidence controls; these estimates cannot certify recovered detail."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EngineConfig
from .types import DiagnosisResult, RecoveryResult


def estimate_recoverability(diagnosis: DiagnosisResult, cfg: EngineConfig) -> RecoveryResult:
    """Combine SNR, estimator confidence and joint darkness; smooth at proxy geometry."""
    c = cfg["real"]["recoverability"]
    y = diagnosis.proxy_y
    size = (y.shape[1], y.shape[0])
    snr = cv2.resize(diagnosis.snr, size, interpolation=cv2.INTER_AREA)
    score = np.clip((snr - c["snr_low"]) / (c["snr_high"] - c["snr_low"]), 0, 1)
    confidence = diagnosis.raw["noise_estimator_confidence"]
    score *= c["confidence_floor"] + (1 - c["confidence_floor"]) * confidence
    local_mean = cv2.boxFilter(y, cv2.CV_32F, (3, 3))
    variance = np.maximum(cv2.boxFilter(y * y, cv2.CV_32F, (3, 3)) - local_mean * local_mean, 0)
    black = (y <= c["black_luminance"]) & (variance <= c["black_variance"])
    score[black | ~diagnosis.proxy_valid] = 0
    score = cv2.ximgproc.guidedFilter(y, score, c["guide_radius"], c["guide_epsilon"])
    np.clip(score, 0, 1, out=score)
    score[black | ~diagnosis.proxy_valid] = 0
    full = cv2.resize(score, (diagnosis.luminance.shape[1], diagnosis.luminance.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Reassert exact native black pixels so interpolation cannot lift a clipped hole.
    full[(diagnosis.luminance == 0) | ~diagnosis.valid] = 0
    classes = (full >= c["class_low"]).astype(np.uint8)
    classes += (full >= c["class_high"]).astype(np.uint8)
    gain_cap = c["gain_low"] + (c["gain_high"] - c["gain_low"]) * full
    valid_count = diagnosis.raw["valid_pixels"]
    counts = np.bincount(classes[diagnosis.valid], minlength=3) if valid_count else np.array([1, 0, 0])
    pcts = counts / max(1, int(np.sum(counts))) * 100
    weak = float(pcts[0])
    sentence = f"About {weak:.1f}% had weak recorded light; brightening was limited there."
    if not valid_count:
        sentence = "No visible pixels were supplied; the confidence overlay contains no visible evidence."
    summary = {
        "recovered_pct": float(pcts[2]), "estimated_pct": float(pcts[1]),
        "unrecoverable_pct": weak, "sentence": sentence,
        "estimator_confidence": float(confidence), "valid_pixels": valid_count,
    }
    return RecoveryResult(full, classes, gain_cap.astype(np.float32), summary)


def enforce_gain(image: np.ndarray, input_y: np.ndarray, recovery: RecoveryResult, cfg: EngineConfig,
                 output_y: np.ndarray | None = None) -> np.ndarray:
    """Cap actual output luminance using a common RGB multiplier, preserving hue."""
    from .diagnose import luminance

    limit = input_y * recovery.gain_cap
    if output_y is None:
        output_y = luminance(image, cfg)
    if not np.any(output_y > limit):
        return image
    scale = np.minimum(1, limit / np.maximum(output_y, cfg["real"]["epsilon"]))
    result = cv2.multiply(image, cv2.merge([scale] * 3))
    return np.clip(result, 0, 1) if result.size == 3 else cv2.min(cv2.max(result, 0), (1, 1, 1, 1))
