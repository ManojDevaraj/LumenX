"""Convert internal float maps into server-ready uint8 visuals."""

from __future__ import annotations

import cv2
import numpy as np

from .contracts import Visual


def rgb_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def fit_long_side(image: np.ndarray, max_side: int, *, area: bool = True) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image.copy()
    scale = max_side / long_side
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interpolation = cv2.INTER_AREA if area else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def scalar_u8(
    values: np.ndarray,
    *,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(np.percentile(values[finite], 1.0)) if low is None else float(low)
    hi = float(np.percentile(values[finite], 99.0)) if high is None else float(high)
    if hi <= lo + 1e-8:
        hi = lo + 1.0
    normalized = np.clip((np.nan_to_num(values, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def heat_rgb(gray_u8: np.ndarray) -> np.ndarray:
    bgr = cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stage_visual(key: str, image: np.ndarray, max_side: int, *, lossless: bool) -> Visual:
    return Visual(key, fit_long_side(rgb_u8(image), max_side), "stage", max_side, lossless)


def diagnostic_visuals(
    illumination: np.ndarray,
    noise_map: np.ndarray,
    snr_map: np.ndarray,
) -> list[Visual]:
    illum_gray = scalar_u8(illumination, low=0.0, high=1.0)
    noise_gray = scalar_u8(noise_map, low=0.0)
    snr_gray = scalar_u8(snr_map, low=0.0)
    return [
        Visual(
            "illumination",
            fit_long_side(heat_rgb(illum_gray), 768),
            "stage",
            768,
            True,
        ),
        Visual(
            "L_gray",
            fit_long_side(illum_gray, 512),
            "map",
            512,
            True,
        ),
        Visual("noise_map", fit_long_side(noise_gray, 512), "map", 512, True),
        Visual("snr", fit_long_side(snr_gray, 512), "map", 512, True),
        Visual(
            "heightmap",
            cv2.resize(illum_gray, (128, 128), interpolation=cv2.INTER_AREA),
            "map",
            None,
            True,
        ),
    ]


def map_visual(
    key: str,
    values: np.ndarray,
    max_side: int = 512,
    value_range: tuple[float, float] | None = None,
) -> Visual:
    low, high = value_range if value_range is not None else (None, None)
    gray = scalar_u8(values, low=low, high=high)
    return Visual(key, fit_long_side(gray, max_side), "map", max_side, True)


def final_visuals(image: np.ndarray, alpha: np.ndarray | None = None) -> list[Visual]:
    rgb = rgb_u8(image)
    final = np.dstack((rgb, np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8))) if alpha is not None else rgb
    preview = fit_long_side(rgb, 1280)
    return [
        Visual("final", final, "final", None, True),
        Visual("final_preview", preview, "stage", 1280, False),
    ]
