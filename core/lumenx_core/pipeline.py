"""Complete classical, deep, and adaptive hybrid engine pipeline."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, replace
from typing import Any, Iterator

import numpy as np

from .classical import restore_classical
from .config import EngineConfig
from .contracts import (
    Cancelled,
    Candidate,
    EnhanceRequest,
    EnhanceResult,
    EngineUnavailable,
    EngineWarning,
    GuardrailCheck,
    RunItem,
    StageEvent,
    Visual,
)
from .denoise import denoise
from .diagnose import diagnose, luminance
from .fusion import fuse_paths
from .guardrails import audit
from .metrics import MetricSuiteImpl
from .models.registry import ModelRegistry
from .recoverability import estimate_recoverability
from .render import (
    diagnostic_visuals,
    final_visuals,
    fit_long_side,
    map_visual,
    rgb_u8,
    stage_visual,
)
from .report import build_adaptive_report
from .tuner import candidate_payload, default_params, search_candidates
from .types import DenoiseResult, RecoveryResult
from .validation import validate_request


_VERSION = "0.4.0"


def _safe_float(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _repair_toward_input(image: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Apply the single bounded, conservative repair used by guardrails."""

    return np.clip(output * 0.68 + image * 0.32, 0.0, 1.0).astype(np.float32)


class RealEngine:
    """CPU/GPU engine with a classical path and optional Zero-DCE++ hybrid path."""

    def __init__(self, config: EngineConfig, device: str | None = None):
        self._config = config
        self._metrics = MetricSuiteImpl(config.values)
        self._models = ModelRegistry(config.values, device=device)
        self._warm = False

    def warmup(self) -> None:
        if self._warm:
            return
        self._models.warmup()
        self._warm = True

    def health(self) -> dict[str, Any]:
        health = self._models.health()
        health["models"]["iqa"] = self._metrics.health()
        return health

    def public_config(self) -> dict[str, Any]:
        return self._config.public()

    def run(self, req: EnhanceRequest) -> Iterator[RunItem]:
        """Validate synchronously, then return the ordered event generator."""

        validate_request(req)
        self.warmup()
        return self._run(req)

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance one live uint8 RGB frame with the loaded Zero-DCE++ model."""

        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("live frames must be uint8 RGB arrays")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("live frames must have HxWx3 RGB shape")
        if max(frame.shape[:2]) > int(self._config.values["deep"]["max_side_live"]):
            raise ValueError("live frames must have a long side of at most 1280 pixels")
        self.warmup()
        if not self._models.available:
            raise EngineUnavailable(self._models.error or "Zero-DCE++ weights are unavailable.")
        result = self._models.enhance(np.ascontiguousarray(frame.astype(np.float32) / 255.0))
        return rgb_u8(result)

    def metric_suite(self) -> MetricSuiteImpl:
        return self._metrics

    def _run(self, req: EnhanceRequest) -> Iterator[RunItem]:
        started = time.perf_counter()
        cfg = self._config.values
        image = req.image
        warnings: list[EngineWarning] = []
        timings: dict[str, float] = {}
        mode = req.mode
        ablate = req.ablate
        is_hybrid = mode == "hybrid"
        use_recoverability = "no_recoverability" not in ablate
        use_classical = mode == "classical" or (is_hybrid and "no_classical" not in ablate)
        use_deep = mode in ("dl", "zerodce") or (is_hybrid and "no_dl" not in ablate)
        use_fusion = is_hybrid and "no_fusion" not in ablate
        run_tuner = is_hybrid and "no_tuner" not in ablate
        if use_deep and not self._models.available and not bool(cfg["deep"].get("fallback_enabled", True)):
            raise EngineUnavailable(
                self._models.error or "Zero-DCE++ is unavailable and fallback is disabled."
            )
        if not self._models.available and self._models.error:
            model_message = self._models.error
        else:
            model_message = ""
        deep_available = bool(use_deep and self._models.available)
        fallback_chain: list[str] = []
        repair_attempted = False

        def check_cancel() -> None:
            if req.cancel is not None and req.cancel.is_set():
                raise Cancelled("LumenX request was cancelled")

        def warning(code: str, message: str) -> StageEvent:
            item = EngineWarning(code, message)
            warnings.append(item)
            return StageEvent(
                "warning",
                code,
                "Processing note",
                "A processing path or metric was unavailable; the response records the fallback.",
                message,
                [],
                None,
                {"code": code, "message": message},
                0.0,
            )

        check_cancel()
        step = time.perf_counter()
        diagnosis = diagnose(image, cfg)
        timings["diagnose"] = (time.perf_counter() - step) * 1000.0
        diagnosis_raw = {
            "mean_luminance": _safe_float(diagnosis.mean_luminance),
            "median_luminance": _safe_float(diagnosis.median_luminance),
            "crushed_shadow_pct": _safe_float(diagnosis.crushed_shadow_pct),
            "highlight_clip_pct": _safe_float(diagnosis.highlight_clip_pct),
            "shadow_clip_pct": _safe_float(diagnosis.shadow_clip_pct),
            "illumination_nonuniformity": _safe_float(diagnosis.illumination_nonuniformity),
            "noise_sigma": _safe_float(diagnosis.noise_sigma),
            "noise_estimator_confidence": _safe_float(diagnosis.noise_confidence),
            "hotspot_count": len(diagnosis.hotspot_boxes),
            "color_cast": diagnosis.color_cast,
            "color_cast_strength": _safe_float(diagnosis.color_cast_strength),
            "severity": diagnosis.severity,
        }
        yield StageEvent(
            "stage",
            "diagnose",
            "Where the light falls",
            "The maps show brighter areas, deep shadows, and measured noise.",
            "Illumination uses guided refinement; noise uses native-resolution flat-patch estimates.",
            diagnostic_visuals(diagnosis.illumination, diagnosis.noise_map, diagnosis.snr_map),
            "illumination",
            diagnosis_raw,
            timings["diagnose"],
        )
        check_cancel()
        step = time.perf_counter()
        recovery = estimate_recoverability(image, diagnosis, cfg)
        timings["recoverability"] = (time.perf_counter() - step) * 1000.0
        yield StageEvent(
            "stage",
            "recoverability",
            "Where signal looks weak",
            "The overlay estimates where the recorded signal appears less reliable.",
            "The estimate combines local SNR, joint near-black variance, and noise-confidence penalties.",
            [Visual("recoverability_overlay", fit_long_side(recovery.overlay, 512), "map", 512, True)],
            "recoverability_overlay",
            {
                "recovered_pct": recovery.recovered_pct,
                "estimated_pct": recovery.estimated_pct,
                "unrecoverable_pct": recovery.unrecoverable_pct,
                "estimator_confidence": recovery.estimator_confidence,
            },
            timings["recoverability"],
        )
        check_cancel()

        if use_recoverability:
            control_recovery = recovery
        else:
            control_recovery = replace(
                recovery,
                soft_score=np.ones_like(recovery.soft_score),
                gain_limit=np.maximum(1.0 - image.max(axis=2), 0.0).astype(np.float32),
                classes=np.zeros_like(recovery.classes),
            )

        params = default_params(diagnosis, cfg)
        candidates: list[Candidate] = []
        search_ms: float | None = None
        deep_proxy = fit_long_side(image, int(cfg["tuner"]["proxy_side"]))
        if run_tuner:
            if deep_available:
                try:
                    deep_proxy = self._models.enhance(deep_proxy)
                except Exception as exc:
                    deep_available = False
                    fallback_chain.extend(["zerodce:inference_error", "classical"])
                    yield warning("zerodce_fallback", f"Zero-DCE++ proxy inference failed; the hybrid run is using classical and safe paths ({type(exc).__name__}).")
                    check_cancel()
            candidates, thumbnails, params, search_ms = search_candidates(
                image,
                diagnosis,
                recovery,
                deep_proxy,
                cfg,
                seed=req.seed,
                use_recoverability=use_recoverability,
                use_classical=use_classical,
                use_deep=deep_available,
                use_fusion=use_fusion,
                uniform_denoise="no_gain_denoise" in ablate,
                correct_color="no_color_correction" not in ablate,
            )
            timings["search"] = search_ms
            for candidate, thumbnail in zip(candidates, thumbnails):
                yield StageEvent(
                    "candidate",
                    candidate.thumb_key,
                    f"Candidate {candidate.id + 1} of 12",
                    "This setting was scored for brightness, clipping, colour, noise, and edges.",
                    "The proxy score uses deterministic custom image-quality measurements; learned IQA is unavailable.",
                    [Visual(candidate.thumb_key, thumbnail, "thumb", 128, False)],
                    candidate.thumb_key,
                    candidate_payload(candidate),
                    0.0,
                )
                check_cancel()
            winner = next(candidate for candidate in candidates if candidate.chosen)
            yield StageEvent(
                "stage",
                "tuner",
                "Choose image settings",
                "Twelve settings were compared against this image's brightness, clipping, colour, and edge evidence.",
                "Search uses a 256-pixel proxy and reuses the proxy Zero-DCE++ output.",
                [],
                winner.thumb_key,
                {"winner_id": winner.id, "search_ms": search_ms, "recipe": "custom_proxy_v1"},
                search_ms,
            )
            check_cancel()
        else:
            timings["search"] = 0.0

        classical_result = None
        classical_image = image
        if use_classical:
            step = time.perf_counter()
            classical_result = restore_classical(
                image,
                diagnosis,
                cfg,
                params,
                correct_color="no_color_correction" not in ablate,
            )
            classical_image = classical_result.image
            timings["classical"] = (time.perf_counter() - step) * 1000.0
            yield StageEvent(
                "stage",
                "classical",
                "Lift light and detail",
                "Retinex lifts shadows while local contrast balances brightness and colour.",
                "Adaptive illumination gamma, protected highlights, Lab CLAHE, and bounded cast correction.",
                [
                    stage_visual("classical", classical_result.image, 768, lossless=False),
                    stage_visual("reflectance", classical_result.reflectance, 768, lossless=False),
                ],
                "classical",
                {
                    "gamma": classical_result.gamma,
                    "clahe_clip": classical_result.clahe_clip,
                    "color_strength": classical_result.color_strength,
                },
                timings["classical"],
            )
            check_cancel()

        deep_image = image.copy()
        if use_deep:
            step = time.perf_counter()
            if deep_available:
                try:
                    deep_image = self._models.enhance(image)
                except Exception as exc:
                    deep_available = False
                    fallback_chain.extend(["zerodce:inference_error", "classical"])
                    yield warning("zerodce_fallback", f"Zero-DCE++ inference failed; a safer classical fallback is being used ({type(exc).__name__}).")
                    check_cancel()
            if not deep_available:
                if mode in ("dl", "zerodce"):
                    if not classical_result:
                        classical_result = restore_classical(
                            image,
                            diagnosis,
                            cfg,
                            params,
                            correct_color="no_color_correction" not in ablate,
                        )
                    deep_image = classical_result.image
                    if not fallback_chain:
                        fallback_chain.extend(["zerodce:missing_or_invalid_weights", "classical"])
                    stage_title = "Classical fallback result"
                    stage_caption = "Trained model weights were unavailable, so the classical restoration is shown instead."
                    tech_note = model_message or "The optional Zero-DCE++ checkpoint was not loaded."
                else:
                    deep_image = image.copy()
                    if not fallback_chain:
                        fallback_chain.extend(["zerodce:missing_or_invalid_weights", "classical"])
                    stage_title = "Model path unavailable"
                    stage_caption = "No trained model weights were loaded; the fusion stage will limit this path."
                    tech_note = model_message or "The optional Zero-DCE++ checkpoint was not loaded."
                if not any(item.code == "zerodce_unavailable" for item in warnings):
                    yield warning("zerodce_unavailable", tech_note)
                    check_cancel()
            else:
                stage_title = "Learned light curve"
                stage_caption = "Zero-DCE++ estimated smooth image-specific light curves for the scene."
                tech_note = f"Zero-DCE++ checkpoint; {self._models.precision} inference on {self._models.device}."
            timings["deep"] = (time.perf_counter() - step) * 1000.0
            deep_visual = stage_visual("deep", deep_image, 768, lossless=False)
            yield StageEvent(
                "stage",
                "deep",
                stage_title,
                stage_caption,
                tech_note,
                [deep_visual],
                "deep",
                {
                    "available": deep_available,
                    "model": "zerodce_pp" if deep_available else None,
                    "device": self._models.device,
                    "precision": self._models.precision,
                },
                timings["deep"],
            )
            check_cancel()

        current = deep_image if mode in ("dl", "zerodce") else classical_image
        if is_hybrid:
            if use_fusion:
                step = time.perf_counter()
                current, weights = fuse_paths(
                    image,
                    classical_image,
                    deep_image,
                    diagnosis,
                    control_recovery,
                    cfg,
                    params=params,
                    use_recoverability=use_recoverability,
                    use_classical=use_classical,
                    use_deep=deep_available,
                )
                timings["fusion"] = (time.perf_counter() - step) * 1000.0
                fusion_map = np.rint(np.clip(weights, 0.0, 1.0) * 255.0).astype(np.uint8)
                yield StageEvent(
                    "stage",
                    "fusion",
                    "Balance the paths",
                    "The mix shifts with illumination, highlight risk, and signal evidence.",
                    "Weights are guided-filtered and normalized per pixel; red=deep, green=classical, blue=safe.",
                    [
                        stage_visual("fused", current, 768, lossless=False),
                        Visual("fusion_map", fit_long_side(fusion_map, 512), "map", 512, True),
                    ],
                    "fused",
                    {
                        "weight_deep_mean": float(np.mean(weights[..., 0], dtype=np.float64)),
                        "weight_classical_mean": float(np.mean(weights[..., 1], dtype=np.float64)),
                        "weight_safe_mean": float(np.mean(weights[..., 2], dtype=np.float64)),
                    },
                    timings["fusion"],
                )
                check_cancel()
            else:
                current = classical_image if use_classical else deep_image

        denoised_result: DenoiseResult | None = None
        if mode in ("classical", "hybrid"):
            step = time.perf_counter()
            denoised_result = denoise(
                image,
                current,
                diagnosis,
                control_recovery,
                cfg,
                use_recoverability=use_recoverability,
                uniform="no_gain_denoise" in ablate,
                enabled=True,
                strength_multiplier=params.get("denoise_k", 1.0),
            )
            current = denoised_result.image
            timings["denoise"] = (time.perf_counter() - step) * 1000.0
            yield StageEvent(
                "stage",
                "denoise",
                "Reduce amplified noise",
                "Smoothing follows local noise estimates and eases around reliable edges.",
                "Three edge-aware strengths are blended spatially; recoverability controls can be ablated.",
                [
                    stage_visual("denoised", current, 768, lossless=False),
                    map_visual("gain", denoised_result.gain_map, 512, (0.0, float(cfg["denoise"]["gain_max"]))),
                    map_visual("denoise_strength", denoised_result.strength_map, 512, (0.0, 1.0)),
                ],
                "denoised",
                {
                    "noise_sigma_before": diagnosis.noise_sigma,
                    "noise_sigma_after": denoised_result.noise_sigma_after,
                    "gain_max": float(np.max(denoised_result.gain_map)),
                    "enabled": True,
                },
                timings["denoise"],
            )
            check_cancel()
        else:
            gray = luminance(current)
            source_gray = luminance(image)
            gain = gray / np.maximum(source_gray, float(cfg["denoise"]["gain_floor"]))
            denoised_result = DenoiseResult(
                image=current,
                gain_map=np.clip(gain, 0.0, float(cfg["denoise"]["gain_max"])).astype(np.float32),
                strength_map=np.zeros_like(gray, dtype=np.float32),
                noise_sigma_after=diagnosis.noise_sigma,
            )

        step = time.perf_counter()
        first_audit = audit(image, current, diagnosis, cfg)
        checks = first_audit.checks
        failed = [check for check in checks if check.passed is False]
        can_repair = mode in ("classical", "hybrid") and "no_guardrails" not in ablate
        if failed and can_repair:
            repair_attempted = True
            current = _repair_toward_input(image, current)
            second_audit = audit(image, current, diagnosis, cfg)
            flagged = {check.check for check in failed}
            checks = [
                replace(
                    check,
                    attempt=2,
                    action="blended toward the original once" if check.check in flagged else None,
                    reason=check.reason,
                )
                for check in second_audit.checks
            ]
            first_audit = replace(second_audit, checks=checks)
            if mode in ("classical", "hybrid") and denoised_result is not None:
                denoised_result = replace(denoised_result, image=current)
        timings["guardrails"] = (time.perf_counter() - step) * 1000.0
        failed_final = [check.check for check in checks if check.passed is False]
        yield StageEvent(
            "stage",
            "guardrails",
            "Review safety checks",
            "The audit checks highlight clipping, colour shift, edge loss, and halos.",
            "At most one conservative repair is attempted; remaining failures are reported.",
            [],
            None,
            {"checks": [asdict(check) for check in checks], "repair_attempted": repair_attempted},
            timings["guardrails"],
        )
        check_cancel()
        if failed_final:
            yield warning(
                "guardrail_unresolved",
                "The final audit still flags " + ", ".join(failed_final) + ".",
            )
            check_cancel()

        metrics_started = time.perf_counter()
        input_proxy = fit_long_side(rgb_u8(image), 512)
        output_proxy = fit_long_side(rgb_u8(current), 512)
        before = self._metrics.unpaired(input_proxy)
        after = self._metrics.unpaired(output_proxy)
        custom = self._metrics.custom(input_proxy, output_proxy)
        unavailable = self._metrics.unavailable()
        unavailable["lpips"] = "LPIPS requires a paired reference image and is measured by the dataset evaluator."
        metrics = {
            "resolution": 512,
            "before": before,
            "after": after,
            "edge_retention": custom["edge_retention"],
            "noise_sigma": {
                "before": custom["noise_sigma_in"],
                "after": custom["noise_sigma_out"],
            },
            "chroma_delta_e": custom["chroma_delta_e"],
            "delta_e": custom["chroma_delta_e"],
            "clip_pct": {
                "before": custom["clip_pct_in"],
                "after": custom["clip_pct_out"],
            },
            "directions": self._metrics.directions(),
            "unavailable": unavailable,
        }
        if custom["edge_retention"] is None:
            unavailable["edge_retention"] = "The input proxy has too few supported edges for a stable measurement."
        if custom["chroma_delta_e"] is None:
            unavailable["chroma_delta_e"] = "Too few valid midtone pixels were available on the 512-pixel proxy."
        timings["metrics"] = (time.perf_counter() - metrics_started) * 1000.0
        learned_unavailable = {
            name: reason
            for name, reason in unavailable.items()
            if name in {"niqe", "brisque", "clipiqa", "musiq"}
        }
        if learned_unavailable:
            yield warning(
                "iqa_unavailable",
                "Some learned quality scores are unavailable; custom measurements remain available.",
            )
            check_cancel()
        check_cancel()

        report_stages = {"diagnose", "recoverability"}
        if use_classical:
            report_stages.add("classical")
        if use_deep:
            report_stages.add("deep")
        if use_fusion:
            report_stages.add("fusion")
        if mode in ("classical", "hybrid"):
            report_stages.add("denoise")
        report_started = time.perf_counter()
        headline, chips, decisions, story = build_adaptive_report(
            diagnosis,
            recovery,
            denoised_result,
            first_audit,
            cfg,
            mode=mode,
            params=params,
            stages=report_stages,
            deep_available=deep_available,
            repair_attempted=repair_attempted,
        )
        timings["report"] = (time.perf_counter() - report_started) * 1000.0
        render_started = time.perf_counter()
        result_visuals = final_visuals(current, req.alpha)
        timings["render"] = (time.perf_counter() - render_started) * 1000.0
        input_hash = hashlib.sha256(image.tobytes(order="C")).hexdigest()[:12]
        candidates_result = candidates if run_tuner else []
        scoring_recipe = "custom_proxy_v1" if run_tuner else "diagnosis_adaptive_defaults"
        actual_method = (
            "classical"
            if mode == "classical"
            else "classical_fallback"
            if use_deep and not deep_available
            else "zerodce"
            if mode in ("dl", "zerodce")
            else "classical"
            if is_hybrid and not use_fusion and use_classical
            else "zerodce"
            if is_hybrid and not use_fusion
            else "classical"
            if is_hybrid and not deep_available
            else "hybrid"
        )
        provenance = {
            "requested_mode": mode,
            "actual_method": actual_method,
            "fallback_chain": fallback_chain,
            "checkpoint": "zerodce_pp" if deep_available else None,
            "precision": self._models.precision if deep_available else "fp32",
            "device": self._models.device,
            "placement_policy": "resident",
            "ablate": sorted(ablate),
            "seed": int(req.seed),
            "config_hash": self._config.digest,
            "input_hash": f"sha256-{input_hash}",
            "checkpoint_hash": self._models.checkpoint_hash if deep_available else None,
            "core_version": _VERSION,
            "contract_version": "1.0",
            "repair_attempted": repair_attempted,
            "guardrails_unresolved": failed_final,
        }
        timings["core_total"] = (time.perf_counter() - started) * 1000.0
        yield EnhanceResult(
            headline=headline,
            chips=chips,
            diagnosis_raw=diagnosis_raw,
            hotspots=diagnosis.hotspot_boxes,
            recoverability={
                "recovered_pct": recovery.recovered_pct,
                "estimated_pct": recovery.estimated_pct,
                "unrecoverable_pct": recovery.unrecoverable_pct,
                "sentence": recovery.sentence,
                "estimator_confidence": recovery.estimator_confidence,
            },
            candidates=candidates_result,
            search_ms=search_ms,
            scoring_recipe=scoring_recipe,
            guardrails=checks,
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
            provenance=provenance,
            visuals=result_visuals,
        )

