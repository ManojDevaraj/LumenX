"""Illumination, noise, clipping, colour-cast, and hotspot diagnosis."""

from __future__ import annotations

import cv2
import numpy as np

from .types import DiagnosisResult


def luminance(image: np.ndarray) -> np.ndarray:
    """Return BT.709 luminance for an RGB float32 image."""

    return (
        image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    ).astype(np.float32, copy=False)


def _guided_filter(
    guide: np.ndarray, source: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    if min(guide.shape[:2]) <= 1:
        return source.astype(np.float32, copy=True)
    radius = max(1, min(radius, (min(guide.shape[:2]) - 1) // 2))
    result = cv2.ximgproc.guidedFilter(
        guide=guide.astype(np.float32, copy=False),
        src=source.astype(np.float32, copy=False),
        radius=radius,
        eps=float(eps),
    )
    return np.asarray(result, dtype=np.float32)


def _mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)) * 1.4826)


def _noise_estimate(
    gray: np.ndarray, cfg: dict[str, float | int]
) -> tuple[float, float, np.ndarray]:
    patch_size = int(cfg["noise_patch_size"])
    max_patches = int(cfg["noise_max_patches"])
    min_dimension = int(cfg["noise_min_patch_dimension"])
    min_pixels = int(cfg["noise_min_patch_pixels"])
    flat_limit = float(cfg["flat_gradient_threshold"])
    floor = float(cfg["noise_floor"])
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - smooth

    patch_records: list[tuple[float, int, int, float]] = []
    tile_noise = np.full(
        (max(1, (h + patch_size - 1) // patch_size), max(1, (w + patch_size - 1) // patch_size)),
        np.nan,
        dtype=np.float32,
    )
    tile_rows, tile_cols = tile_noise.shape
    total_tiles = tile_rows * tile_cols
    # Uniform deterministic sampling bounds patch work even for 12 MP inputs.
    sample_count = min(total_tiles, max_patches * 4)
    sampled_tiles = np.linspace(0, total_tiles - 1, sample_count, dtype=np.int64)
    for flat_index in sampled_tiles:
        row, col = divmod(int(flat_index), tile_cols)
        y, x = row * patch_size, col * patch_size
        y2, x2 = min(y + patch_size, h), min(x + patch_size, w)
        patch = gray[y:y2, x:x2]
        patch_residual = residual[y:y2, x:x2]
        if patch.size < min_pixels:
            continue
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        flatness = float(np.mean(np.hypot(gx, gy)))
        sigma = _mad_sigma(patch_residual)
        if (
            patch.shape[0] >= min_dimension
            and patch.shape[1] >= min_dimension
            and flatness <= flat_limit
        ):
            tile_noise[row, col] = sigma
            patch_records.append((flatness, row, col, sigma))

    patch_records.sort(key=lambda record: (record[0], record[1], record[2]))
    selected = patch_records[:max_patches]
    if selected:
        samples = np.asarray([record[3] for record in selected], dtype=np.float32)
        sigma = max(floor, float(np.median(samples)))
        count_confidence = min(1.0, len(selected) / 20.0)
        consistency = 1.0 / (1.0 + float(np.std(samples)) / max(sigma, floor))
        confidence = float(np.clip(count_confidence * consistency, 0.05, 1.0))
    else:
        sigma = max(floor, _mad_sigma(residual))
        confidence = 0.05

    if (
        float(np.mean(gray, dtype=np.float64)) < float(cfg["flat_black_luminance"])
        and float(np.var(gray, dtype=np.float64)) < float(cfg["flat_black_variance"])
    ):
        confidence = min(confidence, float(cfg["flat_black_confidence_max"]))

    valid_tile = np.isfinite(tile_noise)
    if selected:
        for _, row, col, patch_sigma in selected:
            tile_noise[row, col] = patch_sigma
    if np.any(valid_tile):
        tile_noise[~np.isfinite(tile_noise)] = sigma
        tile_noise = np.maximum(tile_noise, floor)
        noise_map = cv2.resize(tile_noise, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        noise_map = np.full((h, w), sigma, dtype=np.float32)
    return sigma, confidence, np.asarray(noise_map, dtype=np.float32)


def _colour_cast(
    image: np.ndarray,
    gray: np.ndarray,
    cfg: dict[str, float | int],
) -> tuple[str, float]:
    rgb8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    midtones = (gray > float(cfg["cast_mid_low"])) & (gray < float(cfg["cast_mid_high"]))
    if not np.any(midtones):
        return "unclear", 0.0
    a_bias = float(np.mean(lab[..., 1][midtones]) - 128.0)
    b_bias = float(np.mean(lab[..., 2][midtones]) - 128.0)
    magnitude = float(np.hypot(a_bias, b_bias))
    if magnitude < float(cfg["cast_neutral_magnitude"]):
        label = "neutral"
    elif abs(b_bias) >= abs(a_bias):
        label = "warm" if b_bias > 0 else "cool"
    else:
        label = "magenta" if a_bias > 0 else "green"
    return label, float(np.clip(magnitude / float(cfg["cast_normalization"]), 0.0, 1.0))


def _hotspots(
    illumination_proxy: np.ndarray,
    full_shape: tuple[int, int],
    threshold: float,
    min_area: int,
    max_boxes: int,
):
    mask = (illumination_proxy >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    proxy_h, proxy_w = illumination_proxy.shape
    full_h, full_w = full_shape
    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[component])
        if area < min_area:
            continue
        x1 = int(round(x * full_w / proxy_w))
        y1 = int(round(y * full_h / proxy_h))
        x2 = int(round((x + width) * full_w / proxy_w))
        y2 = int(round((y + height) * full_h / proxy_h))
        boxes.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes[:max_boxes]


def diagnose(image: np.ndarray, cfg: dict[str, dict[str, float | int]]) -> DiagnosisResult:
    """Compute input-only diagnosis and full-resolution control maps."""

    h, w = image.shape[:2]
    values = cfg["diagnosis"]
    proxy_side = int(values["proxy_side"])
    scale = min(1.0, proxy_side / max(h, w))
    proxy_w = max(1, int(round(w * scale)))
    proxy_h = max(1, int(round(h * scale)))
    proxy = cv2.resize(image, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
    proxy_gray = luminance(proxy)
    max_channel = np.max(proxy, axis=2).astype(np.float32)
    illum_proxy = _guided_filter(
        proxy_gray,
        max_channel,
        int(values["illumination_radius"]),
        float(values["illumination_eps"]),
    )
    illum_proxy = np.clip(illum_proxy, float(values["illumination_min"]), 1.0)
    illumination = cv2.resize(illum_proxy, (w, h), interpolation=cv2.INTER_LINEAR)
    illumination = np.clip(illumination, float(values["illumination_min"]), 1.0)

    gray = luminance(image)
    noise_sigma, confidence, noise_map = _noise_estimate(gray, values)
    window = int(values["local_window"])
    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (window, window), normalize=True)
    local_second = cv2.boxFilter(gray * gray, cv2.CV_32F, (window, window), normalize=True)
    local_variance = np.maximum(local_second - local_mean * local_mean, 0.0)
    snr_map = local_mean / (noise_map + float(values["noise_epsilon"]))

    # Diagnosis summaries are measured on the 512-pixel proxy for predictable cost.
    mean_luma = float(np.mean(proxy_gray, dtype=np.float64))
    median_luma = float(np.median(proxy_gray))
    shadow_pct = float(
        np.mean(proxy_gray < float(values["shadow_luminance_threshold"]), dtype=np.float64)
        * 100.0
    )
    highlight_clip_pct = float(
        np.mean(
            np.max(proxy, axis=2) > float(values["highlight_clip_threshold"]),
            dtype=np.float64,
        )
        * 100.0
    )
    shadow_clip_pct = float(
        np.mean(
            np.min(proxy, axis=2) < float(values["shadow_clip_threshold"]),
            dtype=np.float64,
        )
        * 100.0
    )
    log_illum = np.log(np.maximum(illum_proxy, float(values["illumination_min"])))
    nonuniformity = float(
        np.clip(
            np.std(log_illum, dtype=np.float64) / float(values["nonuniformity_scale"]),
            0.0,
            1.0,
        )
    )
    cast_label, cast_strength = _colour_cast(proxy, proxy_gray, values)
    if mean_luma < float(values["dark_mean_extreme"]):
        severity = "extreme"
    elif mean_luma < float(values["dark_mean_severe"]):
        severity = "severe"
    elif mean_luma < float(values["dark_mean_moderate"]):
        severity = "moderate"
    elif median_luma < float(values["dark_median_mild"]):
        severity = "moderate"
    else:
        severity = "mild"

    return DiagnosisResult(
        illumination=np.asarray(illumination, dtype=np.float32),
        noise_map=np.asarray(noise_map, dtype=np.float32),
        snr_map=np.asarray(snr_map, dtype=np.float32),
        local_variance=np.asarray(local_variance, dtype=np.float32),
        mean_luminance=mean_luma,
        median_luminance=median_luma,
        crushed_shadow_pct=shadow_pct,
        highlight_clip_pct=highlight_clip_pct,
        shadow_clip_pct=shadow_clip_pct,
        illumination_nonuniformity=nonuniformity,
        noise_sigma=noise_sigma,
        noise_confidence=confidence,
        hotspot_boxes=_hotspots(
            illum_proxy,
            (h, w),
            float(values["hotspot_threshold"]),
            int(values["hotspot_min_area"]),
            int(values["hotspot_max_boxes"]),
        ),
        color_cast=cast_label,
        color_cast_strength=cast_strength,
        severity=severity,
    )
