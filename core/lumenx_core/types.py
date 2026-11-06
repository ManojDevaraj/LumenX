"""Internal typed results passed between classical pipeline components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import GuardrailCheck


@dataclass
class DiagnosisResult:
    illumination: np.ndarray
    noise_map: np.ndarray
    snr_map: np.ndarray
    local_variance: np.ndarray
    mean_luminance: float
    median_luminance: float
    crushed_shadow_pct: float
    highlight_clip_pct: float
    shadow_clip_pct: float
    illumination_nonuniformity: float
    noise_sigma: float
    noise_confidence: float
    hotspot_boxes: list[tuple[int, int, int, int]]
    color_cast: str
    color_cast_strength: float
    severity: str


@dataclass
class RecoveryResult:
    soft_score: np.ndarray
    classes: np.ndarray
    overlay: np.ndarray
    gain_limit: np.ndarray
    recovered_pct: float
    estimated_pct: float
    unrecoverable_pct: float
    estimator_confidence: float
    sentence: str


@dataclass
class ClassicalResult:
    image: np.ndarray
    reflectance: np.ndarray
    adjusted_illumination: np.ndarray
    gamma: float
    clahe_clip: float
    color_strength: float


@dataclass
class DenoiseResult:
    image: np.ndarray
    gain_map: np.ndarray
    strength_map: np.ndarray
    noise_sigma_after: float


@dataclass
class GuardrailResult:
    checks: list[GuardrailCheck]
    edge_retention: float | None
    chroma_delta_e: float | None
    clip_pct_before: float
    clip_pct_after: float
