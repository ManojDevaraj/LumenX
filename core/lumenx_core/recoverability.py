"""Input-evidence recoverability estimates and local gain limits."""

from __future__ import annotations

import cv2
import numpy as np

from .diagnose import luminance
from .types import DiagnosisResult, RecoveryResult


def _percent(mask: np.ndarray) -> float:
    return float(np.mean(mask, dtype=np.float64) * 100.0)


def estimate_recoverability(
    image: np.ndarray,
    diagnosis: DiagnosisResult,
    cfg: dict[str, dict[str, float | int]],
) -> RecoveryResult:
    """Classify recorded signal while lowering confidence for weak estimates."""

    values = cfg["recoverability"]
    gray = luminance(image)
    snr = np.maximum(diagnosis.snr_map, 0.0)
    low = float(values["snr_low"])
    high = float(values["snr_high"])
    log_snr = np.log1p(snr)
    score = np.clip(
        (log_snr - np.log1p(low)) / max(np.log1p(high) - np.log1p(low), 1e-6),
        0.0,
        1.0,
    )

    near_black = (
        (gray < float(values["near_black_luminance"]))
        & (diagnosis.local_variance < float(values["low_local_variance"]))
    )
    score = np.where(near_black, score * 0.18, score)
    confidence = float(np.clip(diagnosis.noise_confidence, 0.0, 1.0))
    confidence_floor = float(values["confidence_floor"])
    score *= confidence_floor + (1.0 - confidence_floor) * confidence

    guide = gray.astype(np.float32, copy=False)
    if min(gray.shape) > 1:
        radius = max(1, min(int(values["guided_radius"]), (min(gray.shape) - 1) // 2))
        score = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=score.astype(np.float32),
            radius=radius,
            eps=float(values["guided_eps"]),
        )
    score = np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)

    classes = np.ones(score.shape, dtype=np.uint8)
    classes[score < float(values["unrecoverable_below"])] = 2
    classes[score >= float(values["recovered_above"])] = 0
    recovered = classes == 0
    estimated = classes == 1
    unrecoverable = classes == 2
    recovered_pct = _percent(recovered)
    estimated_pct = _percent(estimated)
    unrecoverable_pct = _percent(unrecoverable)

    overlay = np.zeros((*gray.shape, 4), dtype=np.uint8)
    overlay[recovered] = (56, 184, 105, 115)
    overlay[estimated] = (235, 177, 52, 115)
    overlay[unrecoverable] = (216, 72, 68, 115)

    minimum_lift = float(values["minimum_lift"])
    additional_lift = float(values["additional_lift"])
    gain_limit = minimum_lift + additional_lift * score
    if unrecoverable_pct >= 1.0:
        sentence = (
            f"About {unrecoverable_pct:.1f}% of pixels had weak local signal evidence; "
            "LumenX limited brightening there and applied stronger smoothing."
        )
    else:
        sentence = (
            f"Most pixels showed usable local signal evidence; brightening was still limited "
            f"in the {unrecoverable_pct:.1f}% with weaker evidence."
        )

    return RecoveryResult(
        soft_score=score,
        classes=classes,
        overlay=overlay,
        gain_limit=np.asarray(gain_limit, dtype=np.float32),
        recovered_pct=recovered_pct,
        estimated_pct=estimated_pct,
        unrecoverable_pct=unrecoverable_pct,
        estimator_confidence=confidence,
        sentence=sentence,
    )
