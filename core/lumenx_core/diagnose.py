"""Fast illumination diagnosis with noise measured on native-resolution patches."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EngineConfig
from .types import DiagnosisResult


def luminance(image: np.ndarray, cfg: EngineConfig) -> np.ndarray:
    """Compute contiguous float32 BT.709 luminance without RGB temporaries."""
    matrix = np.asarray([cfg["real"]["luminance_weights"]], np.float32)
    return cv2.transform(image, matrix)


def channel_max(image: np.ndarray) -> np.ndarray:
    """Reduce contiguous RGB channels in OpenCV instead of a strided Python-side loop."""
    return cv2.reduce(image.reshape(-1, 3), 1, cv2.REDUCE_MAX).reshape(image.shape[:2])


def proxy(image: np.ndarray, side: int) -> np.ndarray:
    """Downsample without enlarging tiny images, preserving channel dimensions."""
    scale = min(1.0, side / max(image.shape[:2]))
    if scale == 1:
        return np.ascontiguousarray(image)
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def native_noise(
    y: np.ndarray, valid: np.ndarray, cfg: EngineConfig
) -> tuple[np.ndarray, float, float, int]:
    """Interpolate deterministic native-patch MAD estimates; report confidence honestly."""
    c = cfg["real"]["diagnosis"]
    block = c["noise_block"]
    height, width = y.shape
    rows, cols = max(1, (height + block - 1) // block), max(1, (width + block - 1) // block)
    grid = np.full((rows, cols), np.nan, np.float32)
    # A stratified regular grid prevents a low-gradient corner from monopolizing samples.
    positions = np.unique(np.linspace(0, rows * cols - 1, min(rows * cols, c["noise_samples"]), dtype=int))
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    samples: list[float] = []
    for index in positions:
        row, col = divmod(int(index), cols)
        top, left = min(row * block, max(0, height - block)), min(col * block, max(0, width - block))
        patch = y[top:top + block, left:left + block]
        mask = valid[top:top + block, left:left + block]
        if float(np.mean(mask)) < c["patch_valid_fraction"] or min(patch.shape) < 3:
            continue
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
        gradient = float(np.mean(cv2.magnitude(gx, gy)[mask]))
        # A checkerboard has a near-zero Sobel response at alternating pixels.
        # Its one-pixel residual correlation is nevertheless close to -1, unlike
        # independent sensor noise.  Reject correlated high-frequency structure.
        centered = patch - cv2.boxFilter(patch, cv2.CV_32F, (5, 5))
        horizontal = (centered[:, :-1] * centered[:, 1:])[mask[:, :-1] & mask[:, 1:]]
        vertical = (centered[:-1, :] * centered[1:, :])[mask[:-1, :] & mask[1:, :]]
        pairs = np.concatenate((horizontal, vertical))
        variance = float(np.mean(centered[mask] ** 2))
        correlation = float(np.mean(pairs) / max(variance, c["flat_variance_floor"])) if pairs.size else 0.0
        if gradient > c["flat_gradient"] or abs(correlation) > c["flat_correlation"]:
            continue
        residual = cv2.filter2D(patch, cv2.CV_32F, kernel)[1:-1, 1:-1]
        inner = mask[1:-1, 1:-1]
        values = residual[inner]
        if not values.size:
            continue
        sigma = float(np.median(np.abs(values - np.median(values)))) / (c["mad_scale"] * c["laplacian_norm"])
        grid[row, col] = sigma
        samples.append(sigma)
    if samples:
        sigma = float(np.median(samples))
        # Confidence reflects support out of the fixed deterministic candidate
        # count.  Small images reach full support only when every candidate is
        # accepted; large images use the configured 100-patch budget.
        confidence = len(samples) / len(positions)
    else:
        # Tiny/no-flat-patch fallback is explicitly uncertain; do not assert zero noise.
        values = y[valid]
        sigma = float(np.median(np.abs(values - np.median(values)))) / c["mad_scale"] if values.size else 0.0
        confidence = 0.0
    grid[~np.isfinite(grid)] = sigma
    noise = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    return noise, sigma, float(confidence), len(samples)


def diagnose(image: np.ndarray, valid: np.ndarray, cfg: EngineConfig) -> DiagnosisResult:
    """Measure the input only; maps and hotspot coordinates use processed geometry."""
    c = cfg["real"]["diagnosis"]
    epsilon = cfg["real"]["epsilon"]
    y = luminance(image, cfg)
    small = proxy(image, c["proxy_side"])
    small_y = luminance(small, cfg)
    small_valid = cv2.resize(valid.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    light = channel_max(small)
    light = cv2.ximgproc.guidedFilter(small_y, light, c["guide_radius"], c["guide_epsilon"])
    np.clip(light, epsilon, 1, out=light)
    illumination = cv2.resize(light, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    noise, sigma, confidence, patch_count = native_noise(y, valid, cfg)
    local_mean = cv2.boxFilter(y, cv2.CV_32F, (c["mean_window"], c["mean_window"]))
    snr = cv2.divide(local_mean, np.maximum(noise, epsilon))
    values = small_y[small_valid]
    if values.size:
        mean, median = float(np.mean(values)), float(np.median(values))
        crushed = float(np.mean(values < c["crushed_threshold"]) * 100)
        p05 = float(np.percentile(values, 5))
    else:
        mean = median = crushed = p05 = 0.0
    severity = "mild"
    for label in ("extreme", "severe", "moderate"):
        if median < c[f"severity_{label}"]:
            severity = label
            break
    bright = bool(values.size and p05 >= c["bright_p05"] and median >= c["bright_median"] and crushed <= c["bright_crushed_pct"])
    hotspots: list[tuple[int, int, int, int]] = []
    mask = ((light > c["hotspot_threshold"]) & small_valid).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sx, sy = image.shape[1] / small.shape[1], image.shape[0] / small.shape[0]
    for x, top, width, height, area in stats[1:]:
        if area >= c["hotspot_min_area"]:
            left, upper = int(x * sx), int(top * sy)
            right = min(image.shape[1], int(np.ceil((x + width) * sx)))
            bottom = min(image.shape[0], int(np.ceil((top + height) * sy)))
            hotspots.append((left, upper, right - left, bottom - upper))
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    mid = small_valid & (small_y >= c["midtone_low"]) & (small_y <= c["midtone_high"])
    if np.any(mid):
        a, b = map(float, np.mean(lab[..., 1:3][mid], axis=0))
        magnitude = float(np.hypot(a, b))
    else:
        a = b = magnitude = 0.0
    cast = "none"
    if magnitude >= c["cast_threshold"]:
        cast = ("warm/orange" if b >= 0 else "blue") if abs(b) >= abs(a) else ("red" if a >= 0 else "green")
    visible_rgb = small[small_valid]
    clip_pct = float(np.mean(np.any(visible_rgb > c["highlight_clip"], axis=1)) * 100) if visible_rgb.size else 0.0
    shadow_pct = float(np.mean(np.any(visible_rgb < c["shadow_clip"], axis=1)) * 100) if visible_rgb.size else 0.0
    raw = {
        "mean_y": mean, "median_y": median, "p05_y": p05, "crushed_pct": crushed,
        "noise_sigma": sigma, "noise_estimator_confidence": confidence, "noise_patch_count": patch_count,
        "severity": severity, "already_bright": bright, "highlight_clip_pct": clip_pct,
        "shadow_channel_clip_pct": shadow_pct, "hotspot_count": len(hotspots),
        "nonuniformity": float(np.std(np.log(light[small_valid]))) if values.size else 0.0,
        "cast": cast, "cast_magnitude": magnitude, "cast_a": a, "cast_b": b,
        "valid_pixels": int(np.count_nonzero(valid)), "empty_visible_area": not bool(values.size),
    }
    return DiagnosisResult(y, illumination, noise, snr, valid, small, small_y, small_valid, raw, hotspots)
