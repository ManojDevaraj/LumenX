"""Display-only uint8 arrays; encoding, resizing and publication belong to server."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EngineConfig
from .contracts import Visual
from .types import DiagnosisResult, RecoveryResult


def uint8(image: np.ndarray) -> np.ndarray:
    """Convert normalized samples to display bytes, preserving input storage."""
    nonnegative = np.maximum(image, 0) if image.size <= 4 else cv2.max(image, 0)
    return cv2.convertScaleAbs(nonnegative, alpha=255)


def colormap(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Return RGB viridis with a fixed numerical scale (never per-image controls)."""
    nonnegative = np.maximum(image, 0) if image.size <= 4 else cv2.max(image, 0)
    gray = cv2.convertScaleAbs(nonnegative, alpha=255 / scale)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def diagnosis_visuals(d: DiagnosisResult, cfg: EngineConfig) -> list[Visual]:
    """Produce every required diagnostic asset at its natural map geometry."""
    c = cfg["real"]["render"]
    gray = uint8(d.illumination)
    return [
        Visual("illumination", colormap(d.illumination), "stage", 768, True),
        Visual("L_gray", gray, "map", 512, True),
        Visual("noise_map", colormap(d.noise, c["noise_max"]), "map", 512, True),
        Visual("snr", colormap(d.snr, c["snr_max"]), "map", 512, True),
        Visual("heightmap", cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA), "map", None, True),
    ]


def recovery_visual(recovery: RecoveryResult, valid: np.ndarray, cfg: EngineConfig) -> Visual:
    """Render green/amber/red confidence with transparency outside visible pixels."""
    c = cfg["real"]["render"]
    colors = np.asarray(c["recovery_colors"], np.uint8)[recovery.classes]
    alpha = np.full(valid.shape, round(c["overlay_alpha"] * 255), np.uint8)
    alpha[~valid] = 0
    return Visual("recoverability_overlay", np.dstack([colors, alpha]), "map", 512, True)


def final_visuals(image: np.ndarray, alpha: np.ndarray | None) -> list[Visual]:
    """Keep the final at full processed size, retaining supplied alpha exactly."""
    rgb = uint8(image)
    final = rgb if alpha is None else np.dstack([rgb, uint8(alpha)])
    return [Visual("final", final, "final", None, True),
            Visual("final_preview", rgb.copy(), "stage", 1280, False)]
