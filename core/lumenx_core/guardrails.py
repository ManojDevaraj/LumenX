"""Measured classical guardrail auditing; this milestone never repairs output."""

from __future__ import annotations

import cv2
import numpy as np

from .config import EngineConfig
from .contracts import GuardrailCheck
from .diagnose import luminance, proxy
from .types import DiagnosisResult


def audit(metrics: dict, image: np.ndarray, diagnosis: DiagnosisResult, cfg: EngineConfig) -> list[GuardrailCheck]:
    """Audit incremental clipping, chroma, reliable edges and hotspot-ring overshoot."""
    c = cfg["real"]["guardrails"]
    m = cfg["real"]["metrics"]
    checks: list[GuardrailCheck] = []
    before, after = metrics["clip_pct"]["before"], metrics["clip_pct"]["after"]
    change = after - before if before is not None and after is not None else None
    passed = after <= c["clip_absolute_pct"] or change <= c["clip_increment_pct"] if change is not None else None
    checks.append(GuardrailCheck("overexposure", change, c["clip_increment_pct"], passed, 1, None,
                                 "No visible clipping measurements." if change is None else None))
    for key, metric_key, threshold, higher in (
        ("color_shift", "chroma_delta_e", c["color_shift"], False),
        ("detail_loss", "edge_retention", c["edge_retention"], True),
    ):
        value = metrics[metric_key]
        passed = (value >= threshold if higher else value <= threshold) if value is not None else None
        checks.append(GuardrailCheck(key, value, threshold, passed, 1, None,
                                     metrics["unavailable"].get(metric_key) if value is None else None))
    small = proxy(image, cfg["real"]["diagnosis"]["proxy_side"])
    output_y = luminance(small, cfg)
    input_y = diagnosis.proxy_y
    mask = ((np.max(diagnosis.proxy_rgb, axis=2) > cfg["real"]["diagnosis"]["hotspot_threshold"])
            & diagnosis.proxy_valid).astype(np.uint8)
    kernel = np.ones((m["halo_radius"] * 2 + 1,) * 2, np.uint8)
    ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0) & diagnosis.proxy_valid
    halo = None
    if diagnosis.hotspots and np.any(ring):
        lift = output_y - input_y
        residual = lift - cv2.GaussianBlur(lift, (0, 0), m["halo_radius"])
        halo = float(np.quantile(np.maximum(residual[ring], 0), m["halo_quantile"]))
    checks.append(GuardrailCheck("halo", halo, c["halo"], halo <= c["halo"] if halo is not None else None,
                                 1, None, "No supported hotspot boundary ring." if halo is None else None))
    return checks
