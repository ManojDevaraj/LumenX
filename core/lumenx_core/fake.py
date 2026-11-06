"""Deterministic, CPU-only preview engine for interface development."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Any, Iterator

import numpy as np

from .contracts import (
    Candidate,
    Cancelled,
    Chip,
    EnhanceRequest,
    EnhanceResult,
    EngineWarning,
    GuardrailCheck,
    InvalidRequest,
    RunItem,
    StageEvent,
    StoryStep,
    Visual,
)
from .validation import validate_request as _validate_request

_CORE_VERSION = "0.4.0"
_CONTRACT_VERSION = "1.0"
_RGB_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
_WARNING_CODE = "fake_engine"
_WARNING_TEXT = (
    "FakeEngine is producing a deterministic preview; trained restoration and quality models are not loaded."
)


def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _to_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(_clip01(image) * 255.0).astype(np.uint8)


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a small display image without adding an OpenCV dependency."""

    y_idx = np.linspace(0, image.shape[0] - 1, max(1, height)).astype(np.intp)
    x_idx = np.linspace(0, image.shape[1] - 1, max(1, width)).astype(np.intp)
    return image[y_idx[:, None], x_idx[None, :]]


def _fit_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return image.copy()
    scale = max_side / side
    return _resize_nearest(
        image,
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    )


def _luminance(image: np.ndarray) -> np.ndarray:
    return np.tensordot(image, _RGB_WEIGHTS, axes=([-1], [0])).astype(np.float32)


def _pseudo_color(gray: np.ndarray) -> np.ndarray:
    x = _clip01(gray)
    red = _clip01(1.7 * x - 0.25)
    green = _clip01(1.4 - np.abs(2.0 * x - 0.9) * 1.35)
    blue = _clip01(1.05 - 1.55 * x)
    return np.stack((red, green, blue), axis=-1)


def _neighbor_mean(image: np.ndarray) -> np.ndarray:
    pad_spec = ((1, 1), (1, 1), (0, 0)) if image.ndim == 3 else ((1, 1), (1, 1))
    padded = np.pad(image, pad_spec, mode="edge")
    total = np.zeros_like(image, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    total /= 9.0
    return total


def _noise_proxy(gray: np.ndarray) -> np.ndarray:
    smooth = _neighbor_mean(gray)
    residual = np.abs(gray - smooth)
    scale = float(np.percentile(residual, 98)) if residual.size else 0.0
    if scale <= 1e-8:
        return np.zeros_like(gray, dtype=np.float32)
    return _clip01(residual / scale)


def _percent(mask: np.ndarray) -> float:
    return float(np.mean(mask, dtype=np.float64) * 100.0) if mask.size else 0.0


def _noise_sigma(gray: np.ndarray) -> float:
    residual = gray - _neighbor_mean(gray)
    median = float(np.median(residual)) if residual.size else 0.0
    mad = float(np.median(np.abs(residual - median))) if residual.size else 0.0
    return float(max(0.0, 1.4826 * mad))


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    if gray.shape[0] == 1:
        dy = np.zeros_like(gray)
    else:
        dy = np.gradient(gray, axis=0)
    if gray.shape[1] == 1:
        dx = np.zeros_like(gray)
    else:
        dx = np.gradient(gray, axis=1)
    return np.hypot(dx, dy).astype(np.float32)


def _edge_retention(inp: np.ndarray, out: np.ndarray) -> float | None:
    edge_in = _edge_magnitude(_luminance(inp)).ravel()
    edge_out = _edge_magnitude(_luminance(out)).ravel()
    if edge_in.size < 2 or float(np.std(edge_in)) <= 1e-8 or float(np.std(edge_out)) <= 1e-8:
        return None
    correlation = float(np.corrcoef(edge_in, edge_out)[0, 1])
    if not math.isfinite(correlation):
        return None
    return float(np.clip(correlation, -1.0, 1.0))


def _check_cancel(req: EnhanceRequest) -> None:
    if req.cancel is not None and req.cancel.is_set():
        raise Cancelled("LumenX request was cancelled")


class FakeMetricSuite:
    """Cheap real image statistics with perceptual models explicitly unavailable."""

    _DIRECTIONS: dict[str, str] = {
        "psnr": "higher",
        "ssim": "higher",
        "lpips": "lower",
        "psnr_gt_mean": "higher",
        "niqe": "lower",
        "brisque": "lower",
        "clipiqa": "higher",
        "musiq": "higher",
        "clip_pct_in": "lower",
        "clip_pct_out": "lower",
        "chroma_delta_e": "lower",
        "edge_retention": "higher",
        "noise_sigma_in": "lower",
        "noise_sigma_out": "lower",
        "err_recovered": "lower",
        "err_estimated": "lower",
        "err_unrecoverable": "lower",
    }

    def paired(self, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
        pred, gt = self._validate_pair(pred, gt)
        pred_f = pred.astype(np.float32) / 255.0
        gt_f = gt.astype(np.float32) / 255.0
        mse = float(np.mean(np.square(pred_f - gt_f), dtype=np.float64))
        psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else None
        return {"psnr": psnr, "ssim": None, "lpips": None, "psnr_gt_mean": None}

    def unpaired(self, img: np.ndarray) -> dict[str, float | None]:
        self._validate_rgb(img)
        return {"niqe": None, "brisque": None, "clipiqa": None, "musiq": None}

    def custom(self, inp: np.ndarray, out: np.ndarray) -> dict[str, float | None]:
        inp, out = self._validate_pair(inp, out)
        inp_f = inp.astype(np.float32) / 255.0
        out_f = out.astype(np.float32) / 255.0
        inp_gray, out_gray = _luminance(inp_f), _luminance(out_f)
        return {
            "clip_pct_in": _percent(np.max(inp, axis=2) >= 250),
            "clip_pct_out": _percent(np.max(out, axis=2) >= 250),
            "chroma_delta_e": None,
            "edge_retention": _edge_retention(inp_f, out_f),
            "noise_sigma_in": _noise_sigma(inp_gray),
            "noise_sigma_out": _noise_sigma(out_gray),
        }

    def confidence_errors(
        self, inp: np.ndarray, pred: np.ndarray, gt: np.ndarray
    ) -> dict[str, float | None]:
        self._validate_pair(inp, pred)
        self._validate_pair(pred, gt)
        return {
            "err_recovered": None,
            "err_estimated": None,
            "err_unrecoverable": None,
            "recovered_pct": None,
            "estimated_pct": None,
            "unrecoverable_pct": None,
        }

    def directions(self) -> dict[str, str]:
        return dict(self._DIRECTIONS)

    def unavailable(self) -> dict[str, str]:
        return {
            "niqe": "FakeEngine does not load the NIQE model.",
            "brisque": "FakeEngine does not load the BRISQUE model.",
            "clipiqa": "FakeEngine does not load the CLIP-IQA model.",
            "musiq": "FakeEngine does not load the MUSIQ model.",
            "lpips": "FakeEngine does not load the LPIPS model.",
            "ssim": "SSIM is not implemented in FakeEngine.",
            "psnr": "PSNR returns None for exact identity pairs to avoid an infinite value.",
            "psnr_gt_mean": "GT-mean adjusted PSNR is not implemented in FakeEngine.",
            "chroma_delta_e": "Chroma Delta E requires the production Lab metric path.",
            "confidence_errors": "The fake run does not produce a calibrated recoverability model.",
        }

    @staticmethod
    def _validate_rgb(img: np.ndarray) -> None:
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
            raise ValueError("metrics expect a uint8 NumPy array")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("metrics expect an HxWx3 RGB array")

    @classmethod
    def _validate_pair(cls, pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cls._validate_rgb(pred)
        cls._validate_rgb(gt)
        if pred.shape != gt.shape:
            raise ValueError("paired metrics expect arrays with the same shape")
        return pred, gt


class FakeEngine:
    """No-weight engine that streams plausible, deterministic preview assets."""

    def __init__(self, stage_delay_ms: int = 0, seed: int = 0):
        if not isinstance(stage_delay_ms, int) or stage_delay_ms < 0:
            raise ValueError("stage_delay_ms must be a non-negative integer")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.stage_delay_ms = stage_delay_ms
        self.seed = seed
        self._warm = False
        self._metrics = FakeMetricSuite()

    def warmup(self) -> None:
        self._warm = True

    def health(self) -> dict[str, Any]:
        return {
            "fake": True,
            "ready": True,
            "hybrid_ready": True,
            "warm": self._warm,
            "device": "cpu",
            "placement_policy": "resident",
            "models": {
                "cidnet": {"available": False, "checkpoints": []},
                "zerodce": {"available": False},
                "iqa": {
                    "clipiqa": False,
                    "niqe": False,
                    "brisque": False,
                    "musiq": False,
                    "lpips": False,
                },
            },
            "vram_mb": {"total": 0, "allocated": 0, "reserved": 0, "peak": 0},
            "degraded": ["fake_engine", "model_weights_unavailable", "iqa_unavailable"],
            "core_version": _CORE_VERSION,
            "contract_version": _CONTRACT_VERSION,
        }

    def public_config(self) -> dict[str, Any]:
        return {
            "engine": "fake",
            "stage_delay_ms": self.stage_delay_ms,
            "seed": self.seed,
            "limitations": [
                "Uses deterministic gamma and local statistics for preview only.",
                "Does not load restoration or perceptual-quality model weights.",
            ],
        }

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise InvalidRequest("invalid_frame", "frame must be a uint8 RGB array")
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise InvalidRequest("invalid_frame", "frame must have non-empty HxWx3 RGB shape")
        if max(frame.shape[:2]) > 1280:
            raise InvalidRequest("invalid_frame", "frame long side must be at most 1280 pixels")
        lifted = np.power(frame.astype(np.float32) / 255.0, 0.72)
        return _to_u8(lifted)

    def metric_suite(self) -> FakeMetricSuite:
        return self._metrics

    def run(self, req: EnhanceRequest) -> Iterator[RunItem]:
        """Validate immediately, then return the lazy event generator."""

        _validate_request(req)
        return self._run(req)

    def _run(self, req: EnhanceRequest) -> Iterator[RunItem]:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        emitted: set[str] = set()
        input_image = req.image
        warnings = [EngineWarning(_WARNING_CODE, _WARNING_TEXT)]

        def make_visual(
            key: str,
            image: np.ndarray,
            role: str,
            max_side: int | None,
            lossless: bool,
        ) -> Visual:
            if key in emitted:
                raise RuntimeError(f"duplicate fake visual key: {key}")
            emitted.add(key)
            return Visual(key, image, role, max_side, lossless)  # type: ignore[arg-type]

        def stage_event(
            name: str,
            title: str,
            caption: str,
            tech_note: str,
            visuals: list[Visual],
            primary_key: str | None,
            data: dict[str, Any],
            t_ms: float,
        ) -> StageEvent:
            return StageEvent(
                kind="stage",
                name=name,
                title=title,
                caption=caption,
                tech_note=tech_note,
                visuals=visuals,
                primary_key=primary_key,
                data=data,
                t_ms=t_ms,
            )

        def delay_after_stage() -> None:
            if self.stage_delay_ms:
                time.sleep(self.stage_delay_ms / 1000.0)

        try:
            _check_cancel(req)
            stage_started = time.perf_counter()
            gray = _luminance(input_image)
            illumination = np.clip(np.max(input_image, axis=2), 0.025, 1.0).astype(np.float32)
            noise_map = _noise_proxy(gray)
            local_signal = _neighbor_mean(gray)
            snr = _clip01(local_signal / (noise_map * 0.08 + 0.015))
            l_gray = _to_u8(illumination)
            heightmap = _to_u8(_resize_nearest(illumination, 128, 128))
            hotspot_mask = illumination > 0.86
            hotspots: list[tuple[int, int, int, int]] = []
            if bool(np.any(hotspot_mask)):
                ys, xs = np.nonzero(hotspot_mask)
                hotspots.append(
                    (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    )
                )
            noise_sigma = _noise_sigma(gray)
            mean_luminance = float(np.mean(gray, dtype=np.float64))
            median_luminance = float(np.median(gray))
            diagnosis_raw = {
                "mean_luminance": mean_luminance,
                "median_luminance": median_luminance,
                "crushed_shadow_pct": _percent(gray < 0.05),
                "highlight_clip_pct": _percent(np.max(input_image, axis=2) > 0.98),
                "shadow_clip_pct": _percent(np.min(input_image, axis=2) < 0.02),
                "noise_sigma_proxy": noise_sigma,
                "noise_estimator_confidence": None,
                "noise_estimator_note": "Illustrative local residual only; not a calibrated noise estimate.",
                "hotspot_count": len(hotspots),
                "severity": "severe" if median_luminance < 0.08 else "moderate",
            }
            diagnose_ms = (time.perf_counter() - stage_started) * 1000.0
            timings["diagnose"] = diagnose_ms
            yield stage_event(
                "diagnose",
                "Where the light falls",
                "This preview marks brighter areas and estimates where the scene is darkest.",
                "FakeEngine uses max-channel illumination and a local residual noise proxy.",
                [
                    make_visual("illumination", _to_u8(_pseudo_color(illumination)), "stage", 768, True),
                    make_visual("L_gray", l_gray, "map", 512, True),
                    make_visual("noise_map", _to_u8(noise_map), "map", 512, True),
                    make_visual("snr", _to_u8(snr), "map", 512, True),
                    make_visual("heightmap", heightmap, "map", None, True),
                ],
                "illumination",
                diagnosis_raw,
                diagnose_ms,
            )
            delay_after_stage()
            _check_cancel(req)

            # The thresholded map is a preview heuristic, not a calibrated recovery claim.
            stage_started = time.perf_counter()
            recoverability_soft = _clip01(snr * (0.75 + 0.25 * illumination))
            recovered = recoverability_soft >= 0.62
            unrecoverable = recoverability_soft < 0.24
            estimated = ~(recovered | unrecoverable)
            rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
            rgba[recovered] = (65, 190, 105, 115)
            rgba[estimated] = (235, 177, 55, 115)
            rgba[unrecoverable] = (220, 75, 68, 115)
            recovered_pct = _percent(recovered)
            estimated_pct = _percent(estimated)
            unrecoverable_pct = _percent(unrecoverable)
            recoverability = {
                "recovered_pct": recovered_pct,
                "estimated_pct": estimated_pct,
                "unrecoverable_pct": unrecoverable_pct,
                "sentence": (
                    "This preview found weak recorded signal in some areas and keeps its brightening gentle there."
                ),
                "estimator_confidence": None,
                "estimator_confidence_reason": "FakeEngine does not calibrate recoverability confidence.",
            }
            rec_ms = (time.perf_counter() - stage_started) * 1000.0
            timings["recoverability"] = rec_ms
            yield stage_event(
                "recoverability",
                "Where signal looks weak",
                "The preview highlights areas where the input has less visible signal.",
                "Heuristic classes are for interface development and are not calibrated estimates.",
                [make_visual("recoverability_overlay", rgba, "map", 512, True)],
                "recoverability_overlay",
                {
                    "recovered_pct": recovered_pct,
                    "estimated_pct": estimated_pct,
                    "unrecoverable_pct": unrecoverable_pct,
                },
                rec_ms,
            )
            delay_after_stage()
            _check_cancel(req)

            yield StageEvent(
                kind="warning",
                name=_WARNING_CODE,
                title="Preview engine",
                caption="This result is a preview because trained models are not loaded.",
                tech_note=_WARNING_TEXT,
                visuals=[],
                primary_key=None,
                data={"code": _WARNING_CODE, "message": _WARNING_TEXT},
                t_ms=0.0,
            )
            _check_cancel(req)
            selected_params = {
                "gamma": 0.74,
                "clahe_clip": 1.8,
                "denoise_k": 0.55,
                "fusion_alpha": 0.5,
                "color_strength": 0.0,
            }
            candidates: list[Candidate] = []
            search_ms: float | None = None
            if req.mode == "hybrid" and "no_tuner" not in req.ablate:
                tuner_started = time.perf_counter()
                candidate_gammas = [0.44, 0.49, 0.54, 0.59, 0.64, 0.68, 0.72, 0.77, 0.51, 0.83, 0.88, 0.98]
                chosen_id = 10
                proxy = _fit_side(input_image, 256)
                for candidate_id, gamma in enumerate(candidate_gammas):
                    _check_cancel(req)
                    candidate_preview = _to_u8(np.power(proxy, gamma))
                    thumb = _fit_side(candidate_preview, 128)
                    thumb_key = f"cand_{candidate_id:02d}"
                    chosen = candidate_id == chosen_id
                    candidate = Candidate(
                        id=candidate_id,
                        params={
                            "gamma": float(gamma),
                            "clahe_clip": float(1.2 + (candidate_id % 5) * 0.25),
                            "denoise_k": float(0.35 + (candidate_id % 4) * 0.12),
                            "fusion_alpha": float(0.35 + (candidate_id % 6) * 0.1),
                            "color_strength": 0.0,
                        },
                        scores={
                            "clipiqa": None,
                            "niqe": None,
                            "clip_pct": float(_percent(np.max(candidate_preview, axis=2) >= 250)),
                            "chroma_delta_e": None,
                            "edge_retention": None,
                            "norm_clipiqa": None,
                            "norm_niqe": None,
                            "norm_clip_pct": None,
                            "norm_chroma_delta_e": None,
                            "norm_edge_retention": None,
                        },
                        total=None,
                        feasible=True,
                        reasons=[]
                        if chosen
                        else ["Ranking metrics are unavailable in FakeEngine."],
                        thumb_key=thumb_key,
                        chosen=chosen,
                    )
                    candidates.append(candidate)
                    yield StageEvent(
                        kind="candidate",
                        name=thumb_key,
                        title=f"Candidate {candidate_id + 1}",
                        caption="Preview of one deterministic brightness setting.",
                        tech_note="Candidate scores are unavailable in the fake engine.",
                        visuals=[make_visual(thumb_key, thumb, "thumb", 128, False)],
                        primary_key=None,
                        data=asdict(candidate),
                        t_ms=0.0,
                    )
                selected_params = dict(candidates[chosen_id].params)
                search_ms = (time.perf_counter() - tuner_started) * 1000.0
                timings["tuner"] = search_ms
                yield stage_event(
                    "tuner",
                    "Choose a preview setting",
                    "The demo selects a conservative setting without learned quality scores.",
                    "FakeEngine emits twelve deterministic candidates with unavailable IQA scores.",
                    [],
                    candidates[chosen_id].thumb_key,
                    {"chosen_id": chosen_id, "candidate_count": len(candidates)},
                    search_ms,
                )
                delay_after_stage()
                _check_cancel(req)

            gamma = float(selected_params["gamma"])
            classical_image = _clip01(np.power(input_image, gamma))
            deep_gamma = 0.68 if req.mode != "zerodce" else 0.75
            deep_image = _clip01(np.power(input_image, deep_gamma))
            if req.mode == "hybrid" and "no_classical" not in req.ablate:
                reflection = _clip01(input_image / np.maximum(illumination[..., None], 0.08))
                reflectance_display = _clip01(reflection / np.maximum(reflection.max(), 1.0))
                stage_started = time.perf_counter()
                class_visuals = [
                    make_visual("classical", _to_u8(classical_image), "stage", 768, False),
                    make_visual("reflectance", _to_u8(reflectance_display), "stage", 768, False),
                ]
                class_ms = (time.perf_counter() - stage_started) * 1000.0
                timings["classical"] = class_ms
                yield stage_event(
                    "classical",
                    "Lift the darker details",
                    "A gentle brightness curve lifts the scene while keeping its colours familiar.",
                    "Preview uses a per-image gamma curve; it does not run full Retinex or CLAHE.",
                    class_visuals,
                    "classical",
                    {"gamma": gamma, "preview_only": True},
                    class_ms,
                )
                delay_after_stage()
                _check_cancel(req)
            elif req.mode == "classical":
                reflection = _clip01(input_image / np.maximum(illumination[..., None], 0.08))
                reflectance_display = _clip01(reflection / np.maximum(reflection.max(), 1.0))
                stage_started = time.perf_counter()
                class_ms = (time.perf_counter() - stage_started) * 1000.0
                timings["classical"] = class_ms
                yield stage_event(
                    "classical",
                    "Lift the darker details",
                    "A gentle brightness curve lifts the scene while keeping its colours familiar.",
                    "Preview uses a per-image gamma curve; it does not run full Retinex or CLAHE.",
                    [
                        make_visual("classical", _to_u8(classical_image), "stage", 768, False),
                        make_visual("reflectance", _to_u8(reflectance_display), "stage", 768, False),
                    ],
                    "classical",
                    {"gamma": gamma, "preview_only": True},
                    class_ms,
                )
                delay_after_stage()
                _check_cancel(req)

            need_deep = req.mode in ("hybrid", "dl", "zerodce") and not (
                req.mode == "hybrid" and "no_dl" in req.ablate
            )
            if need_deep:
                stage_started = time.perf_counter()
                deep_ms = (time.perf_counter() - stage_started) * 1000.0
                timings["deep"] = deep_ms
                yield stage_event(
                    "deep",
                    "Preview the model path",
                    "This brightness preview stands in for the trained restoration path.",
                    "FakeEngine does not execute CIDNet or Zero-DCE++.",
                    [make_visual("deep", _to_u8(deep_image), "stage", 768, False)],
                    "deep",
                    {"model": None, "preview_only": True},
                    deep_ms,
                )
                delay_after_stage()
                _check_cancel(req)

            if req.mode == "classical":
                selected_output = classical_image
            elif req.mode in ("dl", "zerodce"):
                selected_output = deep_image
            elif "no_fusion" in req.ablate:
                selected_output = deep_image
            else:
                stage_started = time.perf_counter()
                dl_scalar = 1.0 - illumination
                classical_scalar = 0.55 * illumination
                if "no_dl" in req.ablate:
                    dl_scalar = np.zeros_like(illumination)
                elif "no_recoverability" not in req.ablate:
                    dl_scalar = dl_scalar * recoverability_soft
                if "no_classical" in req.ablate:
                    classical_scalar = np.zeros_like(illumination)
                dl_weight = np.broadcast_to(dl_scalar[..., None], input_image.shape)
                classical_weight = np.broadcast_to(
                    classical_scalar[..., None], input_image.shape
                )
                safe_weight = np.broadcast_to(
                    np.where(illumination > 0.84, 0.65, 0.12)[..., None], input_image.shape
                )
                denominator = np.maximum(dl_weight + classical_weight + safe_weight, 1e-6)
                fusion_map = np.stack(
                    (
                        dl_weight[..., 0] / denominator[..., 0],
                        classical_weight[..., 0] / denominator[..., 0],
                        safe_weight[..., 0] / denominator[..., 0],
                    ),
                    axis=-1,
                )
                fused = (
                    deep_image * fusion_map[..., 0, None]
                    + classical_image * fusion_map[..., 1, None]
                    + input_image * fusion_map[..., 2, None]
                )
                fusion_ms = (time.perf_counter() - stage_started) * 1000.0
                timings["fusion"] = fusion_ms
                yield stage_event(
                    "fusion",
                    "Blend the preview paths",
                    "The demo blends its two brightness previews and protects the brightest areas.",
                    "Weights are illustrative and do not include guided-filter smoothing.",
                    [
                        make_visual("fused", _to_u8(fused), "stage", 768, False),
                        make_visual("fusion_map", _to_u8(fusion_map), "map", 512, True),
                    ],
                    "fused",
                    {"weights_sum_to_one": True},
                    fusion_ms,
                )
                delay_after_stage()
                _check_cancel(req)
                selected_output = fused

            if req.mode in ("classical", "hybrid"):
                stage_started = time.perf_counter()
                if req.mode == "hybrid" and "no_gain_denoise" not in req.ablate:
                    local_mean = _neighbor_mean(selected_output)
                    denoised = _clip01(0.88 * selected_output + 0.12 * local_mean)
                    if "no_recoverability" in req.ablate:
                        denoise_strength = np.full_like(gray, 0.4, dtype=np.float32)
                    else:
                        denoise_strength = _clip01((1.0 - recoverability_soft) * 0.65)
                else:
                    denoised = selected_output.copy()
                    denoise_strength = np.zeros_like(gray)
                if req.mode == "hybrid" and "no_recoverability" not in req.ablate:
                    local_lift_limit = 0.08 + 0.92 * recoverability_soft
                    denoised = np.minimum(
                        denoised,
                        input_image + local_lift_limit[..., None],
                    ).astype(np.float32, copy=False)
                gain = _clip01(_luminance(denoised) / np.maximum(gray, 0.02) / 4.0)
                denoise_ms = (time.perf_counter() - stage_started) * 1000.0
                timings["denoise"] = denoise_ms
                yield stage_event(
                    "denoise",
                    "Smooth the preview",
                    "A light smoothing pass reduces small variations while keeping the main edges visible.",
                    "The fake path uses a small neighborhood blend, not gain-aware guided filtering.",
                    [
                        make_visual("denoised", _to_u8(denoised), "stage", 768, False),
                        make_visual("gain", _to_u8(gain), "map", 512, True),
                        make_visual(
                            "denoise_strength", _to_u8(denoise_strength), "map", 512, True
                        ),
                    ],
                    "denoised",
                    {"preview_only": True},
                    denoise_ms,
                )
                delay_after_stage()
                _check_cancel(req)
                selected_output = denoised

            stage_started = time.perf_counter()
            guardrails = [
                GuardrailCheck(
                    check="overexposure",
                    value=None,
                    threshold=1.0,
                    passed=None,
                    attempt=1,
                    action=None,
                    reason="FakeEngine does not run calibrated guardrail metrics.",
                ),
                GuardrailCheck(
                    check="color_shift",
                    value=None,
                    threshold=5.0,
                    passed=None,
                    attempt=1,
                    action=None,
                    reason="FakeEngine does not compute chroma Delta E.",
                ),
                GuardrailCheck(
                    check="detail_loss",
                    value=None,
                    threshold=0.5,
                    passed=None,
                    attempt=1,
                    action=None,
                    reason="FakeEngine does not run a calibrated detail audit.",
                ),
                GuardrailCheck(
                    check="halo",
                    value=None,
                    threshold=0.1,
                    passed=None,
                    attempt=1,
                    action=None,
                    reason="FakeEngine does not detect halo artifacts.",
                ),
            ]
            guardrail_ms = (time.perf_counter() - stage_started) * 1000.0
            timings["guardrails"] = guardrail_ms
            yield stage_event(
                "guardrails",
                "Review safety checks",
                "Quality checks are listed, but this preview does not claim that they passed.",
                "No guardrail metric is computed by FakeEngine.",
                [],
                None,
                {"checks_available": False},
                guardrail_ms,
            )
            delay_after_stage()
            _check_cancel(req)

            render_started = time.perf_counter()
            final_rgb = _to_u8(selected_output)
            if req.alpha is not None:
                final_image = np.dstack((final_rgb, np.rint(req.alpha * 255.0).astype(np.uint8)))
            else:
                final_image = final_rgb
            preview_image = _fit_side(final_rgb, 1280)
            final_visual = make_visual("final", final_image, "final", None, True)
            preview_visual = make_visual("final_preview", preview_image, "stage", 1280, False)
            timings["render"] = (time.perf_counter() - render_started) * 1000.0

            metric_start = time.perf_counter()
            before_rgb = _to_u8(_fit_side(input_image, 512))
            after_rgb = _to_u8(_fit_side(final_rgb, 512))
            metric_values = self._metrics.custom(before_rgb, after_rgb)
            clip_in = float(metric_values["clip_pct_in"] or 0.0)
            clip_out = float(metric_values["clip_pct_out"] or 0.0)
            metrics = {
                "resolution": 512,
                "before": {"niqe": None, "clipiqa": None, "brisque": None},
                "after": {"niqe": None, "clipiqa": None, "brisque": None},
                "edge_retention": metric_values["edge_retention"],
                "noise_sigma": {
                    "before": metric_values["noise_sigma_in"],
                    "after": metric_values["noise_sigma_out"],
                },
                "chroma_delta_e": None,
                "delta_e": None,
                "clip_pct": {"before": clip_in, "after": clip_out},
                "directions": {
                    "niqe": "lower",
                    "clipiqa": "higher",
                    "brisque": "lower",
                    "edge_retention": "higher",
                    "chroma_delta_e": "lower",
                },
                "unavailable": self._metrics.unavailable(),
            }
            metric_ms = (time.perf_counter() - metric_start) * 1000.0
            timings["metrics"] = metric_ms

            if req.mode == "classical":
                actual_method = "classical"
                fallback_chain: list[str] = []
            elif req.mode == "zerodce":
                actual_method = "classical_fallback"
                fallback_chain = ["zerodce:weights_unavailable", "classical_preview"]
            elif req.mode == "dl":
                actual_method = "classical_fallback"
                fallback_chain = ["cidnet:weights_unavailable", "classical_preview"]
            else:
                actual_method = "hybrid"
                fallback_chain = ["fake_engine:restoration_models_unavailable"]

            input_hash = hashlib.sha256(input_image.tobytes(order="C")).hexdigest()[:12]
            config_blob = json.dumps(self.public_config(), sort_keys=True).encode("utf-8")
            config_hash = hashlib.sha256(config_blob).hexdigest()[:12]
            if req.mode == "hybrid":
                if "no_fusion" in req.ablate:
                    path_decision = "The fusion stage was disabled by this run's ablation."
                    preview_caption = "One brightness preview stands in for the model path."
                    preview_key = "deep"
                elif "no_dl" in req.ablate:
                    path_decision = "The deep path was disabled, so the classical preview and input were blended."
                    preview_caption = "The classical preview is blended with the original image."
                    preview_key = "fused"
                elif "no_classical" in req.ablate:
                    path_decision = "The classical path was disabled, so the model preview and input were blended."
                    preview_caption = "The model preview is blended with the original image."
                    preview_key = "fused"
                else:
                    path_decision = "The demo blended two brightness previews and kept bright regions closer to input."
                    preview_caption = "Two brightness previews are blended for the demo result."
                    preview_key = "fused"
                denoise_decision = (
                    "The denoising pass was disabled by this run's ablation."
                    if "no_gain_denoise" in req.ablate
                    else "A small smoothing pass reduced local variation in the hybrid preview."
                )
                recovery_decision = (
                    "The overlay is shown, but this run did not use it to steer brightening or smoothing."
                    if "no_recoverability" in req.ablate
                    else "The coloured overlay is a rough guide and cannot confirm what detail was recorded."
                )
                decisions = [
                    "Measured brightness was low, so a gentle curve lifted the darker areas.",
                    path_decision,
                    denoise_decision,
                    recovery_decision,
                    "Image-quality scores and safety checks are unavailable in FakeEngine.",
                ]
                story = [
                    StoryStep(
                        "Where light falls",
                        "The brightness map shows the brighter and darker parts of the scene.",
                        "illumination",
                    ),
                    StoryStep(
                        "Where signal looks weak",
                        "The overlay marks areas with less visible signal in this preview.",
                        "recoverability_overlay",
                    ),
                    StoryStep(
                        "Preview result",
                        preview_caption,
                        preview_key,
                    ),
                    StoryStep(
                        "Final image",
                        "This deterministic preview is ready to compare with the input.",
                        "final_preview",
                    ),
                ]
            elif req.mode == "classical":
                decisions = [
                    "Measured brightness was low, so a gentle curve lifted the darker areas.",
                    "The classical preview used a simple brightness curve without a trained model.",
                    "The classical preview leaves its brightness output unchanged after that curve.",
                    "The coloured overlay is a rough guide and cannot confirm what detail was recorded.",
                    "Image-quality scores and safety checks are unavailable in FakeEngine.",
                ]
                story = [
                    StoryStep(
                        "Where light falls",
                        "The brightness map shows the brighter and darker parts of the scene.",
                        "illumination",
                    ),
                    StoryStep(
                        "Where signal looks weak",
                        "The overlay marks areas with less visible signal in this preview.",
                        "recoverability_overlay",
                    ),
                    StoryStep(
                        "Preview result",
                        "A simple brightness curve lifts the darker areas.",
                        "classical",
                    ),
                    StoryStep(
                        "Final image",
                        "This deterministic preview is ready to compare with the input.",
                        "final_preview",
                    ),
                ]
            else:
                model_name = "Zero-DCE++" if req.mode == "zerodce" else "CIDNet"
                decisions = [
                    "Measured brightness was low, so a gentle curve lifted the darker areas.",
                    f"The {model_name} weights are unavailable, so this is a brightness preview.",
                    "No perceptual model ran, so image-quality scores are unavailable.",
                    "The coloured overlay is a rough guide and cannot confirm what detail was recorded.",
                    "Safety checks are listed as unavailable rather than reported as passed.",
                ]
                story = [
                    StoryStep(
                        "Where light falls",
                        "The brightness map shows the brighter and darker parts of the scene.",
                        "illumination",
                    ),
                    StoryStep(
                        "Where signal looks weak",
                        "The overlay marks areas with less visible signal in this preview.",
                        "recoverability_overlay",
                    ),
                    StoryStep(
                        "Preview result",
                        f"A simple curve stands in for the unavailable {model_name} model.",
                        "deep",
                    ),
                    StoryStep(
                        "Final image",
                        "This deterministic preview is ready to compare with the input.",
                        "final_preview",
                    ),
                ]
            report_started = time.perf_counter()
            result = EnhanceResult(
                headline=(
                    f"{diagnosis_raw['severity'].capitalize()} low-light scene preview with a gentle brightness lift."
                ),
                chips=[
                    Chip(
                        "Brightness",
                        mean_luminance,
                        "high" if mean_luminance < 0.12 else "med" if mean_luminance < 0.35 else "low",
                    ),
                    Chip(
                        "Shadow detail",
                        float(diagnosis_raw["crushed_shadow_pct"]),
                        "high"
                        if diagnosis_raw["crushed_shadow_pct"] > 35
                        else "med"
                        if diagnosis_raw["crushed_shadow_pct"] > 15
                        else "low",
                    ),
                    Chip("Noise estimate", noise_sigma, "med"),
                ],
                diagnosis_raw=diagnosis_raw,
                hotspots=hotspots,
                recoverability=recoverability,
                candidates=candidates,
                search_ms=search_ms,
                scoring_recipe="fake_preview_unscored",
                guardrails=guardrails,
                decisions=decisions,
                story=story,
                metrics=metrics,
                relight={
                    "base_key": "final_preview",
                    "L_key": "L_gray",
                    "gamma_default": 1.0,
                    "exposure_default": 1.0,
                    "gamma_range": [0.4, 2.5],
                    "exposure_range": [0.25, 4.0],
                },
                timings_ms=timings,
                warnings=warnings,
                provenance={
                    "requested_mode": req.mode,
                    "actual_method": actual_method,
                    "fallback_chain": fallback_chain,
                    "checkpoint": None,
                    "precision": "fp32",
                    "device": "cpu",
                    "placement_policy": "resident",
                    "ablate": sorted(req.ablate),
                    "seed": int(req.seed + self.seed),
                    "config_hash": f"sha256-{config_hash}",
                    "input_hash": f"sha256-{input_hash}",
                    "checkpoint_hash": None,
                    "core_version": _CORE_VERSION,
                    "contract_version": _CONTRACT_VERSION,
                    "repair_attempted": False,
                    "guardrails_unresolved": [],
                },
                visuals=[final_visual, preview_visual],
            )
            timings["report"] = (time.perf_counter() - report_started) * 1000.0
            timings["core_total"] = (time.perf_counter() - started) * 1000.0
            yield result
        finally:
            # No device allocations are made in FakeEngine. Dropping the local
            # reference is enough to release its temporary arrays on early close.
            input_image = None


def create_fake_engine(stage_delay_ms: int = 0, seed: int = 0) -> FakeEngine:
    """Create the no-weights demonstration engine."""

    return FakeEngine(stage_delay_ms=stage_delay_ms, seed=seed)
