"""Audit-only exposure, colour, detail, and halo checks."""

from __future__ import annotations

import cv2
import numpy as np

from .contracts import GuardrailCheck
from .diagnose import luminance
from .types import DiagnosisResult, GuardrailResult


def _edge_retention(
    inp: np.ndarray,
    out: np.ndarray,
    min_pixels: int,
    percentile: float,
    floor: float,
    std_epsilon: float,
) -> float | None:
    in_y = luminance(inp)
    out_y = luminance(out)
    in_mag = cv2.magnitude(
        cv2.Sobel(in_y, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(in_y, cv2.CV_32F, 0, 1, ksize=3),
    )
    out_mag = cv2.magnitude(
        cv2.Sobel(out_y, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(out_y, cv2.CV_32F, 0, 1, ksize=3),
    )
    supported = in_mag > max(float(np.percentile(in_mag, percentile)), floor)
    if int(np.count_nonzero(supported)) < min_pixels:
        return None
    left = in_mag[supported].astype(np.float64)
    right = out_mag[supported].astype(np.float64)
    if float(np.std(left)) <= std_epsilon or float(np.std(right)) <= std_epsilon:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else None


def _chroma_delta_e(
    inp: np.ndarray,
    out: np.ndarray,
    mid_low: float,
    mid_high: float,
    max_span: float,
    min_pixels: int,
) -> float | None:
    in_rgb = np.rint(np.clip(inp, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_rgb = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    in_lab = cv2.cvtColor(in_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    in_y = luminance(inp)
    chroma_span = np.max(inp, axis=2) - np.min(inp, axis=2)
    valid = (in_y >= mid_low) & (in_y <= mid_high) & (chroma_span < max_span)
    if int(np.count_nonzero(valid)) < min_pixels:
        return None
    da = in_lab[..., 1][valid] - out_lab[..., 1][valid]
    db = in_lab[..., 2][valid] - out_lab[..., 2][valid]
    return float(np.mean(np.hypot(da, db), dtype=np.float64))


def _halo_overshoot(
    inp: np.ndarray,
    out: np.ndarray,
    hotspots: list[tuple[int, int, int, int]],
    margin: int,
    blur_size: int,
) -> float | None:
    if not hotspots:
        return None
    in_y, out_y = luminance(inp), luminance(out)
    in_residual = in_y - cv2.GaussianBlur(in_y, (blur_size, blur_size), 0)
    out_residual = out_y - cv2.GaussianBlur(out_y, (blur_size, blur_size), 0)
    h, w = in_y.shape
    overshoots: list[float] = []
    for x, y, box_w, box_h in hotspots:
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + box_w + margin), min(h, y + box_h + margin)
        ring = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        inner_x1, inner_y1 = x - x1, y - y1
        inner_x2, inner_y2 = min(ring.shape[1], inner_x1 + box_w), min(ring.shape[0], inner_y1 + box_h)
        ring[:] = True
        ring[inner_y1:inner_y2, inner_x1:inner_x2] = False
        if np.any(ring):
            delta = out_residual[y1:y2, x1:x2][ring] - in_residual[y1:y2, x1:x2][ring]
            overshoots.append(float(np.percentile(np.maximum(delta, 0.0), 95)))
    return max(overshoots) if overshoots else None


def audit(
    inp: np.ndarray,
    out: np.ndarray,
    diagnosis: DiagnosisResult,
    cfg: dict[str, dict[str, float | int]],
) -> GuardrailResult:
    """Measure the classical result without modifying it."""

    values = cfg["guardrails"]
    in_clipped = np.max(inp, axis=2) > float(cfg["diagnosis"]["highlight_clip_threshold"])
    out_clipped = np.max(out, axis=2) > float(cfg["diagnosis"]["highlight_clip_threshold"])
    clip_before = float(np.mean(in_clipped, dtype=np.float64) * 100.0)
    clip_after = float(np.mean(out_clipped, dtype=np.float64) * 100.0)
    exposure_delta = clip_after - clip_before
    chroma_delta = _chroma_delta_e(
        inp,
        out,
        float(values["chroma_mid_low"]),
        float(values["chroma_mid_high"]),
        float(values["chroma_max_span"]),
        int(values["minimum_metric_pixels"]),
    )
    edge = _edge_retention(
        inp,
        out,
        int(values["minimum_edge_pixels"]),
        float(values["reliable_edge_percentile"]),
        float(values["reliable_edge_floor"]),
        float(values["edge_std_epsilon"]),
    )
    halo = _halo_overshoot(
        inp,
        out,
        diagnosis.hotspot_boxes,
        int(values["halo_ring_margin"]),
        int(values["halo_blur_size"]),
    )

    checks = [
        GuardrailCheck(
            "overexposure",
            exposure_delta,
            float(values["overexposure_delta_pct"]),
            exposure_delta <= float(values["overexposure_delta_pct"]),
            1,
            None,
            None,
        ),
        GuardrailCheck(
            "color_shift",
            chroma_delta,
            float(values["chroma_delta_e_limit"]),
            None if chroma_delta is None else chroma_delta <= float(values["chroma_delta_e_limit"]),
            1,
            None,
            "Too few valid midtone pixels." if chroma_delta is None else None,
        ),
        GuardrailCheck(
            "detail_loss",
            edge,
            float(values["minimum_edge_retention"]),
            None if edge is None else edge >= float(values["minimum_edge_retention"]),
            1,
            None,
            "Too few supported input edges." if edge is None else None,
        ),
        GuardrailCheck(
            "halo",
            halo,
            float(values["halo_overshoot_limit"]),
            None if halo is None else halo <= float(values["halo_overshoot_limit"]),
            1,
            None,
            "No bright hotspots were detected." if halo is None else None,
        ),
    ]
    return GuardrailResult(
        checks=checks,
        edge_retention=edge,
        chroma_delta_e=chroma_delta,
        clip_pct_before=clip_before,
        clip_pct_after=clip_after,
    )
