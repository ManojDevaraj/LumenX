"""Real classical statistics, with explicit unavailable learned IQA in v0.2.0."""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np

from .config import EngineConfig
from .contracts import InvalidRequest
from .diagnose import luminance, native_noise


_CLIPIQA: Any | None = None
_CLIPIQA_ERROR: str | None = None


def clipiqa_batch(images: list[np.ndarray]) -> list[float] | None:
    """Score uint8 RGB images in one CLIP-IQA forward pass when pyiqa is installed."""
    global _CLIPIQA, _CLIPIQA_ERROR
    if not images or _CLIPIQA_ERROR is not None:
        return None
    try:
        if _CLIPIQA is None:
            import pyiqa
            _CLIPIQA = pyiqa.create_metric("clipiqa", device="cuda" if _cuda_available() else "cpu")
        import torch
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255)
        with torch.inference_mode():
            return [float(value) for value in _CLIPIQA(batch).detach().cpu().reshape(-1)]
    except Exception as exc:  # optional model dependency must never break processing
        _CLIPIQA_ERROR = f"CLIP-IQA unavailable: {type(exc).__name__}."
        return None


def _cuda_available() -> bool:
    """Avoid importing torch merely to report unavailable optional metrics."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def validate_metric_image(image: np.ndarray) -> None:
    """Require positive uint8 RGB metric inputs; evaluation is not size-limited."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise InvalidRequest("metric_image", "Metric images must be positive uint8 RGB arrays.")


def chroma_delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIEDE2000 with equal lightness; input arrays contain Lab a*,b* pairs.

    Implements the equations in Sharma, Wu and Dalal (2005), DOI 10.1002/col.20070.
    Formula constants are standard coefficients, not tunable engine thresholds.
    No lightness term remains after equalizing L*. Hue averages handle wraparound.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    c1, c2 = np.hypot(a[..., 0], a[..., 1]), np.hypot(b[..., 0], b[..., 1])
    mean_c = (c1 + c2) / 2
    g = (1 - np.sqrt(mean_c**7 / (mean_c**7 + 25**7))) / 2
    ap1, ap2 = a[..., 0] * (1 + g), b[..., 0] * (1 + g)
    cp1, cp2 = np.hypot(ap1, a[..., 1]), np.hypot(ap2, b[..., 1])
    h1 = np.mod(np.arctan2(a[..., 1], ap1), 2 * np.pi)
    h2 = np.mod(np.arctan2(b[..., 1], ap2), 2 * np.pi)
    difference = h2 - h1
    product = cp1 * cp2
    dh = np.where(difference > np.pi, difference - 2 * np.pi, difference)
    dh = np.where(dh < -np.pi, dh + 2 * np.pi, dh)
    dh = np.where(product == 0, 0, dh)
    delta_h = 2 * np.sqrt(product) * np.sin(dh / 2)
    average_h = (h1 + h2) / 2
    average_h = np.where((np.abs(difference) > np.pi) & (h1 + h2 < 2 * np.pi), average_h + np.pi, average_h)
    average_h = np.where((np.abs(difference) > np.pi) & (h1 + h2 >= 2 * np.pi), average_h - np.pi, average_h)
    average_h = np.where(product == 0, h1 + h2, average_h)
    t = (1 - 0.17 * np.cos(average_h - np.deg2rad(30)) + 0.24 * np.cos(2 * average_h)
         + 0.32 * np.cos(3 * average_h + np.deg2rad(6)) - 0.20 * np.cos(4 * average_h - np.deg2rad(63)))
    average_c = (cp1 + cp2) / 2
    dc = (cp2 - cp1) / (1 + 0.045 * average_c)
    dh_scaled = delta_h / (1 + 0.015 * average_c * t)
    rotation = -2 * np.sqrt(average_c**7 / (average_c**7 + 25**7)) * np.sin(
        np.deg2rad(60) * np.exp(-((np.rad2deg(average_h) - 275) / 25) ** 2)
    )
    return np.sqrt(np.maximum(dc * dc + dh_scaled * dh_scaled + rotation * dc * dh_scaled, 0))


def custom_values(inp: np.ndarray, out: np.ndarray, cfg: EngineConfig, valid: np.ndarray | None = None) -> tuple[dict[str, float | None], dict[str, str]]:
    """Measure clipping, chroma-only change, reliable edges and native noise."""
    validate_metric_image(inp)
    validate_metric_image(out)
    if inp.shape != out.shape:
        raise InvalidRequest("metric_shape", "Metric images must have the same shape.")
    first = inp.astype(np.float32) / 255
    last = out.astype(np.float32) / 255
    if valid is None:
        valid = np.ones(inp.shape[:2], bool)
    c = cfg["real"]["metrics"]
    threshold = cfg["real"]["diagnosis"]["highlight_clip"]
    values: dict[str, float | None] = dict.fromkeys((
        "clip_pct_in", "clip_pct_out", "chroma_delta_e", "edge_retention", "noise_sigma_in", "noise_sigma_out"
    ))
    reasons: dict[str, str] = {}
    if not np.any(valid):
        return values, dict.fromkeys(values, "No visible pixels are available.")
    for suffix, image in (("in", first), ("out", last)):
        values[f"clip_pct_{suffix}"] = float(np.mean(np.max(image, axis=2)[valid] > threshold) * 100)
    y1, y2 = luminance(first, cfg), luminance(last, cfg)
    noise, sigma1, confidence1, _ = native_noise(y1, valid, cfg)
    _, sigma2, confidence2, _ = native_noise(y2, valid, cfg)
    for suffix, sigma, confidence in (("in", sigma1, confidence1), ("out", sigma2, confidence2)):
        if confidence > 0:
            values[f"noise_sigma_{suffix}"] = sigma
        else:
            reasons[f"noise_sigma_{suffix}"] = "No sufficiently supported flat-patch noise estimate."
    mask = (valid & (y1 >= c["midtone_low"]) & (y1 <= c["midtone_high"])
            & (np.max(first, axis=2) < threshold) & (np.max(last, axis=2) < threshold))
    if np.any(mask):
        lab1 = cv2.cvtColor(first, cv2.COLOR_RGB2Lab)
        lab2 = cv2.cvtColor(last, cv2.COLOR_RGB2Lab)
        values["chroma_delta_e"] = float(np.mean(chroma_delta_e(lab1[..., 1:][mask], lab2[..., 1:][mask])))
    else:
        reasons["chroma_delta_e"] = "No valid unsaturated input midtones."
    gradient1 = cv2.magnitude(cv2.Sobel(y1, cv2.CV_32F, 1, 0), cv2.Sobel(y1, cv2.CV_32F, 0, 1))
    gradient2 = cv2.magnitude(cv2.Sobel(y2, cv2.CV_32F, 1, 0), cv2.Sobel(y2, cv2.CV_32F, 0, 1))
    snr = cv2.boxFilter(y1, cv2.CV_32F, (15, 15)) / np.maximum(noise, cfg["real"]["epsilon"])
    edges = valid & (gradient1 > c["gradient_threshold"]) & (snr >= c["reliable_snr"])
    if confidence1 > 0 and np.count_nonzero(edges) >= c["minimum_edges"]:
        x, y = gradient1[edges].astype(np.float64), gradient2[edges].astype(np.float64)
        x -= np.mean(x)
        y -= np.mean(y)
        denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if denominator > c["variance_epsilon"]:
            values["edge_retention"] = float(np.clip(np.sum(x * y) / denominator, -1, 1))
        else:
            reasons["edge_retention"] = "Reliable-edge correlation has zero variance."
    else:
        reasons["edge_retention"] = "Too few reliable input edges or insufficient noise-estimate support."
    return values, reasons


class ClassicalMetricSuite:
    """Stateless metric methods sharing immutable real-engine configuration."""

    def __init__(self, cfg: EngineConfig) -> None:
        """Keep only immutable configuration; no models are loaded in this milestone."""
        self._cfg = cfg

    def paired(self, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
        """Measure raw/GT-mean PSNR; perceptual/SSIM metrics arrive with full IQA."""
        validate_metric_image(pred)
        validate_metric_image(gt)
        if pred.shape != gt.shape:
            raise InvalidRequest("metric_shape", "Metric images must have the same shape.")
        difference = pred.astype(np.float64) - gt.astype(np.float64)
        mse = float(np.mean(difference * difference))
        adjusted = None
        mean = float(np.mean(pred))
        if mean > 0:
            rescaled = np.clip(pred.astype(np.float64) * (float(np.mean(gt)) / mean), 0, 255)
            error = float(np.mean((rescaled - gt) ** 2))
            if error > 0:
                adjusted = float(10 * np.log10(255**2 / error))
        return {"psnr": float(10 * np.log10(255**2 / mse)) if mse > 0 else None,
                "psnr_gt_mean": adjusted, "ssim": None, "lpips": None}

    def unpaired(self, img: np.ndarray) -> dict[str, float | None]:
        """Return optional CLIP-IQA plus honestly unavailable unintegrated metrics."""
        validate_metric_image(img)
        return {"niqe": None, "brisque": None, "clipiqa": (clipiqa_batch([img]) or [None])[0], "musiq": None}

    def custom(self, inp: np.ndarray, out: np.ndarray) -> dict[str, float | None]:
        """Return native-resolution custom metrics; caller chooses evaluation resolution."""
        return custom_values(inp, out, self._cfg)[0]

    def confidence_errors(self, inp: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
        """Reserve confidence calibration for the feature-complete evaluation milestone."""
        for image in (inp, pred, gt):
            validate_metric_image(image)
        if inp.shape != pred.shape or inp.shape != gt.shape:
            raise InvalidRequest("metric_shape", "Metric images must have the same shape.")
        return dict.fromkeys(("err_recovered", "err_estimated", "err_unrecoverable",
                              "recovered_pct", "estimated_pct", "unrecoverable_pct"))

    def directions(self) -> dict[str, Literal["higher", "lower"]]:
        """Return fixed directions, independent of per-image score normalization."""
        return {**dict.fromkeys(("psnr", "psnr_gt_mean", "ssim", "clipiqa", "musiq", "edge_retention"), "higher"),
                **dict.fromkeys(("lpips", "niqe", "brisque", "chroma_delta_e", "delta_e", "clip_pct_in",
                                 "clip_pct_out", "noise_sigma_in", "noise_sigma_out", "err_recovered",
                                 "err_estimated", "err_unrecoverable"), "lower")}

    def unavailable(self) -> dict[str, str]:
        """Supply reasons for every conditionally or permanently unavailable metric."""
        reasons = dict.fromkeys(("niqe", "brisque", "musiq", "lpips", "ssim",
                                "err_recovered", "err_estimated", "err_unrecoverable",
                                "recovered_pct", "estimated_pct", "unrecoverable_pct"),
                               "Not implemented in the classical-only v0.2.0 milestone.")
        if _CLIPIQA_ERROR is not None:
            reasons["clipiqa"] = _CLIPIQA_ERROR
        reasons.update({
            "psnr": "An exact match has infinite PSNR; returned as None.",
            "psnr_gt_mean": "Zero prediction mean or an exact adjusted match is undefined/infinite.",
            "edge_retention": "Too few reliable edges or zero variance in their correlation.",
            "chroma_delta_e": "No valid unsaturated input midtones.",
            "noise_sigma_in": "No sufficiently supported flat-patch noise estimate.",
            "noise_sigma_out": "No sufficiently supported flat-patch noise estimate.",
        })
        return reasons


def request_metrics(inp: np.ndarray, out: np.ndarray, valid: np.ndarray, cfg: EngineConfig) -> dict[str, Any]:
    """Compute proxy perceptual metrics while retaining native-resolution noise."""
    from .diagnose import proxy

    first, last = proxy(inp, 512), proxy(out, 512)
    mask = cv2.resize(valid.astype(np.uint8), (first.shape[1], first.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    values, reasons = custom_values(first, last, cfg, mask)
    # Downsampling smooths sensor noise. The displayed before/after values must
    # agree with diagnosis and therefore use the same native-patch estimator.
    native_values, native_reasons = custom_values(inp, out, cfg, valid)
    for key in ("noise_sigma_in", "noise_sigma_out"):
        values[key] = native_values[key]
        reasons.pop(key, None)
        if key in native_reasons:
            reasons[key] = native_reasons[key]
    suite = ClassicalMetricSuite(cfg)
    learned_before, learned_after = suite.unpaired(first), suite.unpaired(last)
    unavailable = suite.unavailable()
    reasons.update({key: unavailable[key] for key in ("niqe", "brisque", "musiq")})
    if learned_before["clipiqa"] is None:
        reasons["clipiqa"] = unavailable.get("clipiqa", "CLIP-IQA is unavailable.")
    for suffix, field in (("in", "before"), ("out", "after")):
        if values[f"noise_sigma_{suffix}"] is None:
            reasons[f"noise_sigma.{field}"] = reasons[f"noise_sigma_{suffix}"]
        if values[f"clip_pct_{suffix}"] is None:
            reasons[f"clip_pct.{field}"] = reasons[f"clip_pct_{suffix}"]
    if values["chroma_delta_e"] is None:
        reasons["delta_e"] = reasons["chroma_delta_e"]
    return {
        "resolution": 512,
        "before": {key: learned_before[key] for key in ("niqe", "clipiqa", "brisque")},
        "after": {key: learned_after[key] for key in ("niqe", "clipiqa", "brisque")},
        "edge_retention": values["edge_retention"],
        "noise_sigma": {"before": values["noise_sigma_in"], "after": values["noise_sigma_out"]},
        "chroma_delta_e": values["chroma_delta_e"], "delta_e": values["chroma_delta_e"],
        "clip_pct": {"before": values["clip_pct_in"], "after": values["clip_pct_out"]},
        "directions": suite.directions(), "unavailable": reasons,
    }
