"""Array-bearing internal results, separate from the frozen public dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiagnosisResult:
    """Measured input properties and aligned full-resolution maps."""

    luminance: np.ndarray
    illumination: np.ndarray
    noise: np.ndarray
    snr: np.ndarray
    valid: np.ndarray
    proxy_rgb: np.ndarray
    proxy_y: np.ndarray
    proxy_valid: np.ndarray
    raw: dict[str, Any]
    hotspots: list[tuple[int, int, int, int]]


@dataclass(frozen=True)
class RecoveryResult:
    """Estimated confidence, class IDs, actual gain limits and area statistics."""

    soft: np.ndarray
    classes: np.ndarray
    gain_cap: np.ndarray
    summary: dict[str, Any]


@dataclass(frozen=True)
class EnhanceParams:
    """Diagnosis-derived classical parameters; no search runs in classical mode."""

    gamma: float
    clahe_strength: float
    color_strength: float


@dataclass(frozen=True)
class ClassicalResult:
    """Recomposed RGB, reflectance, adjusted light and chosen parameters."""

    image: np.ndarray
    reflectance: np.ndarray
    illumination: np.ndarray
    params: EnhanceParams


@dataclass(frozen=True)
class DenoiseResult:
    """Denoised output plus measured gain and applied filter-strength maps."""

    image: np.ndarray
    gain: np.ndarray
    strength: np.ndarray


@dataclass(frozen=True)
class DeepResult:
    """Output/provenance of an actual model or an explicit classical fallback request."""

    image: np.ndarray | None
    method: str
    checkpoint: str | None
    checkpoint_hash: str | None
    device: str
    precision: str
    fallback_chain: list[str]
    warnings: list[Any]


@dataclass(frozen=True)
class FusionResult:
    """Hybrid output and nonnegative normalized deep/classical/safe weights."""

    image: np.ndarray
    weights: np.ndarray
