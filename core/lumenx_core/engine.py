"""Real restoration engine: array events, no transport, storage or scheduling."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from time import perf_counter
from typing import Any, Iterator

import cv2
import numpy as np

from .classical import enhance_classical
from .config import EngineConfig, load_config
from .contracts import (CONTRACT_VERSION, Cancelled, EngineError, EngineUnavailable, EngineWarning,
                        EnhanceRequest, EnhanceResult, RunItem, StageEvent, StoryStep, Visual)
from .denoise import denoise
from .diagnose import diagnose
from .guardrails import audit
from .fusion import fuse
from .metrics import ClassicalMetricSuite, request_metrics
from .recoverability import estimate_recoverability
from .recoverability import enforce_gain
from .render import colormap, diagnosis_visuals, final_visuals, recovery_visual, uint8
from .report import classical_report, diagnosis_chips
from .timing import timer
from .tuner import tune
from .validation import validate_request
from .types import DenoiseResult, EnhanceParams


class RealEngine:
    """Classical, direct trained baselines, and adaptive hybrid still processing."""

    def __init__(self, config_path: str | None = None, device: str | None = None) -> None:
        """Load immutable configuration once; do not download or instantiate models."""
        try:
            self._cfg: EngineConfig = load_config(config_path)
        except (OSError, TypeError, ValueError) as exc:
            raise EngineError(f"Invalid engine configuration: {exc}") from exc
        cv2.setNumThreads(self._cfg["real"]["threads"])
        from .models.registry import ModelRegistry

        self._registry = ModelRegistry(self._cfg, device)
        self._warm = False

    def warmup(self) -> None:
        """Initialize OpenCV colour/filter kernels once with a tiny classical request."""
        if not self._warm:
            image = np.full((32, 32, 3), 0.1, np.float32)
            for _ in self.run(EnhanceRequest(image, (32, 32), mode="classical")):
                pass
            self._registry.warmup()
            self._warm = True

    def public_config(self) -> dict[str, Any]:
        """Return an isolated JSON-safe copy, not a reference to engine configuration."""
        return self._cfg.public()

    def metric_suite(self) -> ClassicalMetricSuite:
        """Provide actual custom metrics and explicit reasons for absent learned IQA."""
        return ClassicalMetricSuite(self._cfg)

    def health(self) -> dict[str, Any]:
        """Report actual loaded checkpoints; full tuner readiness remains separate."""
        from . import __version__

        registry = self._registry
        checkpoints = [key for key in registry.runners if key.startswith("cidnet")]
        return {
            "fake": False, "ready": True, "hybrid_ready": bool(checkpoints or "zerodce_pp" in registry.runners), "warm": self._warm,
            "device": registry.device, "placement_policy": registry.policy,
            "modes": ["classical", "dl", "zerodce", "hybrid"],
            "hybrid_requires_no_tuner": False,
            "hybrid_without_tuner_ready": bool(checkpoints or "zerodce_pp" in registry.runners),
            "models": {"cidnet": {"available": bool(checkpoints), "checkpoints": checkpoints},
                       "zerodce": {"available": "zerodce_pp" in registry.runners},
                       "iqa": dict.fromkeys(("clipiqa", "niqe", "brisque", "musiq", "lpips"), False)},
            "vram_mb": registry.memory(),
            "degraded": (["iqa_unavailable"] + [f"{key}:{reason}" for key, reason in registry.errors.items()]
                         + [warning.code for warning in registry.warnings]),
            "core_version": __version__, "contract_version": CONTRACT_VERSION,
        }

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the shared verified Zero-DCE++ model, without a silent gamma substitute."""
        from .validation import validate_rgb

        validate_rgb(frame, floating=False)
        result = self._registry.enhance_frame(frame.astype(np.float32) / 255)
        return uint8(result)

    @staticmethod
    def _cancel(req: EnhanceRequest) -> None:
        """Honor cancellation between processing stages and array rendering."""
        if req.cancel is not None and req.cancel.is_set():
            raise Cancelled("Enhancement cancelled.")

    def run(self, req: EnhanceRequest) -> Iterator[RunItem]:
        """Validate once, then dispatch to the requested frozen schedule."""
        start = perf_counter()
        validate_request(req)
        self._cancel(req)
        if req.mode == "classical":
            yield from self._run_classical(req, start)
        else:
            yield from self._run_neural(req, start)

    def _run_classical(self, req: EnhanceRequest, start: float) -> Iterator[RunItem]:
        """Retain the validated v0.2 classical path independently of model availability."""
        from . import __version__

        timings: dict[str, float] = {}
        owned: list[Any] = []
        story: list[StoryStep] = []
        warnings = [EngineWarning("iqa_unavailable", "Learned quality metrics are not available in v0.2.0."),
                    EngineWarning("guardrails_audit_only", "Quality checks report findings without automatic repair.")]
        try:
            valid = np.ones(req.image.shape[:2], bool) if req.alpha is None else req.alpha > 0
            image = req.image
            if not np.all(valid):
                image = image.copy()
                image[~valid] = 0
            owned.append(image)
            with timer(timings, "diagnose"):
                diagnosis = diagnose(image, valid, self._cfg)
            owned.append(diagnosis)
            with timer(timings, "render"):
                visuals = diagnosis_visuals(diagnosis, self._cfg)
            event = self._stage("diagnose", "Finding the recorded light",
                                "Measured brightness, flat-patch noise and bright areas before changing the image.",
                                "Native-patch MAD noise and proxy-guided illumination.", visuals, "illumination", diagnosis.raw,
                                timings["diagnose"])
            story.append(StoryStep(event.title, event.caption, "illumination"))
            yield event
            self._cancel(req)
            if diagnosis.raw["noise_estimator_confidence"] == 0:
                warnings.append(EngineWarning("noise_estimate_uncertain", "No supported flat-patch noise estimate; enhancement uses conservative confidence."))
            for warning in warnings:
                yield StageEvent("warning", warning.code, "Processing limitation", warning.message, "", [], None, asdict(warning), 0.0)
                self._cancel(req)

            with timer(timings, "recoverability"):
                recovery = estimate_recoverability(diagnosis, self._cfg)
            owned.append(recovery)
            with timer(timings, "render"):
                overlay = recovery_visual(recovery, valid, self._cfg)
            event = self._stage("recoverability", "Locating weak recorded signal",
                                "Estimated signal confidence controls how much brightening each area can receive.",
                                "Input SNR, estimator support and joint near-black variance.", [overlay], overlay.key,
                                recovery.summary, timings["recoverability"])
            story.append(StoryStep(event.title, event.caption, overlay.key))
            yield event
            self._cancel(req)

            with timer(timings, "classical"):
                classical = enhance_classical(image, diagnosis, self._cfg)
            owned.append(classical)
            with timer(timings, "render"):
                visuals = [Visual("classical", uint8(classical.image), "stage", 768, False),
                           Visual("reflectance", uint8(classical.reflectance), "stage", 768, False)]
            event = self._stage("classical", "Adjusting the light",
                                "Used measured darkness to lift shadows while protecting already-bright areas.",
                                "Retinex, partial Lab CLAHE and bounded chroma correction.", visuals, "classical",
                                asdict(classical.params), timings["classical"])
            story.append(StoryStep(event.title, event.caption, "classical"))
            yield event
            self._cancel(req)

            with timer(timings, "denoise"):
                denoised = denoise(classical.image, diagnosis, recovery, self._cfg)
            owned.append(denoised)
            with timer(timings, "render"):
                visuals = [Visual("denoised", uint8(denoised.image), "stage", 768, False),
                           Visual("gain", colormap(denoised.gain, self._cfg["real"]["render"]["gain_max"]), "map", 512, True),
                           Visual("denoise_strength", uint8(denoised.strength), "map", 512, True)]
            event = self._stage("denoise", "Controlling amplified noise",
                                "Matched smoothing to brightening and noise, then enforced the local brightness limits.",
                                "Three guided-filter strengths with reliable-edge protection; final luminance gain cap.",
                                visuals, "denoised", {"mean_gain": float(np.mean(denoised.gain[valid])) if np.any(valid) else 0.0},
                                timings["denoise"])
            story.append(StoryStep(event.title, event.caption, "denoised"))
            yield event
            self._cancel(req)

            with timer(timings, "metrics"):
                metrics = request_metrics(uint8(image), uint8(denoised.image), valid, self._cfg)
            self._cancel(req)
            with timer(timings, "guardrails"):
                checks = audit(metrics, denoised.image, diagnosis, self._cfg)
            failed = [check.check for check in checks if check.passed is False]
            if failed:
                warnings.append(EngineWarning("guardrail_unresolved", "Some quality checks failed; this classical release reports them without automatic repair."))
            yield self._stage("guardrails", "Checking the result",
                              "Checked clipping, colour, reliable edges and lamp boundaries; findings remain visible for review.",
                              "Audit only in v0.2.0; no repair or output mutation.", [], None,
                              {"audit_only": True, "checks": [asdict(check) for check in checks]}, timings["guardrails"])
            self._cancel(req)
            with timer(timings, "report"):
                headline, chips, decisions = classical_report(diagnosis, recovery, classical, checks)
            with timer(timings, "render"):
                final = final_visuals(denoised.image, req.alpha)
            digest = sha256(str(req.image.shape).encode())
            digest.update(memoryview(req.image))
            if req.alpha is not None:
                digest.update(req.alpha.tobytes())
            timings["core_total"] = (perf_counter() - start) * 1000
            yield EnhanceResult(
                headline, chips, diagnosis.raw, diagnosis.hotspots, recovery.summary, [], None, "not_run", checks,
                decisions, story, metrics,
                {"base_key": "final_preview", "L_key": "L_gray", "gamma_default": 1.0,
                 "exposure_default": 1.0, "gamma_range": [0.4, 2.5], "exposure_range": [0.25, 4.0]},
                timings, warnings,
                {"requested_mode": req.mode, "actual_method": "classical", "fallback_chain": [],
                 "checkpoint": None, "checkpoint_hash": None, "precision": "fp32", "device": "cpu",
                 "placement_policy": "resident", "ablate": [], "seed": req.seed,
                 "config_hash": self._cfg.digest, "input_hash": digest.hexdigest()[:12],
                 "core_version": __version__, "contract_version": CONTRACT_VERSION,
                 "repair_attempted": False, "guardrails_unresolved": failed, "perf_levers": []}, final,
            )
        finally:
            owned.clear()
            # GeneratorExit discards the frame and all remaining local arrays.

    def _run_neural(self, req: EnhanceRequest, start: float) -> Iterator[RunItem]:
        """Run direct trained baselines or hybrid, with one cached-output repair."""
        from . import __version__
        from .diagnose import luminance

        cfg = self._cfg
        timings: dict[str, float] = {}
        warnings = [EngineWarning("iqa_unavailable", "Learned quality measurements are not loaded in this release.")]
        story: list[StoryStep] = []
        hybrid = req.mode == "hybrid"
        recovery_enabled = "no_recoverability" not in req.ablate
        no_fusion = "no_fusion" in req.ablate
        audit_only = not hybrid or no_fusion or "no_guardrails" in req.ablate
        deep_result = classical = None
        repair_attempted = False
        fallback_audit: list[dict[str, Any]] = []
        candidates = []
        scoring_recipe = "not_run"
        chosen_params = None
        try:
            valid = np.ones(req.image.shape[:2], bool) if req.alpha is None else req.alpha > 0
            image = req.image
            if not np.all(valid):
                image = image.copy()
                image[~valid] = 0
            with timer(timings, "diagnose"):
                diagnosis = diagnose(image, valid, cfg)
            with timer(timings, "render"):
                visuals = diagnosis_visuals(diagnosis, cfg)
            event = self._stage("diagnose", "Finding the recorded light",
                                "Measured the recorded light and noise before selecting the restoration path.",
                                "Native-patch noise; input-only diagnosis.", visuals, "illumination", diagnosis.raw, timings["diagnose"])
            story.append(StoryStep(event.title, event.caption, "illumination"))
            yield event
            self._cancel(req)
            if hybrid and "no_tuner" in req.ablate:
                warnings.append(EngineWarning("tuner_disabled", "Your no_tuner request uses the configured fixed parameters."))
            for warning in warnings:
                yield StageEvent("warning", warning.code, "Processing limitation", warning.message, "", [], None, asdict(warning), 0.0)
                self._cancel(req)
            with timer(timings, "recoverability"):
                recovery = estimate_recoverability(diagnosis, cfg)
            with timer(timings, "render"):
                visual = recovery_visual(recovery, valid, cfg)
            event = self._stage("recoverability", "Locating weak recorded signal",
                                "Estimated signal confidence limits brightening unless that control was explicitly disabled.",
                                "SNR, estimator support and joint near-black variance.", [visual], visual.key,
                                recovery.summary, timings["recoverability"])
            story.append(StoryStep(event.title, event.caption, visual.key))
            yield event
            self._cancel(req)

            if hybrid and "no_tuner" not in req.ablate:
                with timer(timings, "tuner"):
                    candidates, candidate_visuals, chosen_params, scoring_recipe = tune(image, diagnosis, cfg, req.seed)
                for candidate, visual in zip(candidates, candidate_visuals):
                    yield StageEvent("candidate", visual.key, "Trying a light setting",
                                     "Compared this setting using fixed quality checks.",
                                     "Twelve deterministic proxy candidates; CLIP-IQA is recorded when available.",
                                     [visual], visual.key, asdict(candidate), 0.0)
                    self._cancel(req)
                yield self._stage("tuner", "Choosing a setting", "Selected the feasible setting with the strongest fixed score.",
                                  "Twelve candidates; fixed normalization constants; clipiqa+custom recipe.", [],
                                  candidates[[candidate.chosen for candidate in candidates].index(True)].thumb_key,
                                  {"chosen_id": next(candidate.id for candidate in candidates if candidate.chosen),
                                   "scoring_recipe": scoring_recipe}, timings["tuner"])
                self._cancel(req)

            if hybrid and "no_classical" not in req.ablate:
                c = cfg["real"]["classical"]
                params = chosen_params or EnhanceParams(c["default_gamma"], c["clahe_strength"], c["default_color_strength"])
                with timer(timings, "classical"):
                    classical = enhance_classical(image, diagnosis, cfg, params=params,
                                                   color_enabled="no_color_correction" not in req.ablate)
                with timer(timings, "render"):
                    visuals = [Visual("classical", uint8(classical.image), "stage", 768, False),
                               Visual("reflectance", uint8(classical.reflectance), "stage", 768, False)]
                event = self._stage("classical", "Preparing the classical result",
                                    "Applied the selected brightness and colour settings while protecting highlights.",
                                    "Tuner-selected parameters, or configured no_tuner parameters.", visuals, "classical", asdict(classical.params), timings["classical"])
                story.append(StoryStep(event.title, event.caption, "classical"))
                yield event
                self._cancel(req)
            deep_image = None
            if not hybrid or "no_dl" not in req.ablate:
                with timer(timings, "deep"):
                    deep_result = self._registry.infer(image, diagnosis.raw["severity"],
                                                       zerodce=req.mode == "zerodce", cancel=req.cancel)
                    deep_image = deep_result.image
                    if deep_image is None:
                        if hybrid and "no_classical" in req.ablate:
                            raise EngineUnavailable("No trained fallback is available and classical processing was explicitly disabled.")
                        deep_image = classical.image if classical is not None else enhance_classical(image, diagnosis, cfg).image
                warnings.extend(deep_result.warnings)
                for warning in deep_result.warnings:
                    yield StageEvent("warning", warning.code, "Restoration limitation",
                                     "A processing limitation was recorded; the available safe path was used.",
                                     warning.message, [], None, asdict(warning), 0.0)
                    self._cancel(req)
                with timer(timings, "render"):
                    visual = Visual("deep", uint8(deep_image), "stage", 768, False)
                event = self._stage("deep", "Running the restoration path",
                                    "Used the recorded model or its explicitly reported fallback at the original image size.",
                                    "Sequential tiled inference above one megapixel.", [visual], "deep",
                                    {"actual_method": deep_result.method, "checkpoint": deep_result.checkpoint,
                                     "fallback_chain": deep_result.fallback_chain}, timings["deep"])
                story.append(StoryStep(event.title, event.caption, "deep"))
                yield event
                self._cancel(req)
            output = deep_image
            classical_image = classical.image if classical is not None else None
            if hybrid and not no_fusion:
                with timer(timings, "fusion"):
                    fused = fuse(image, deep_image, classical_image, diagnosis, recovery, cfg,
                                 recovery_enabled=recovery_enabled)
                output = fused.image
                with timer(timings, "render"):
                    visuals = [Visual("fused", uint8(output), "stage", 768, False),
                               Visual("fusion_map", uint8(fused.weights), "map", 512, True)]
                event = self._stage("fusion", "Combining the available evidence",
                                    "Mixed restoration paths using recorded signal confidence and protected bright areas.",
                                    "Input-guided normalized classical/deep/safe weights.", visuals, "fused", {}, timings["fusion"])
                story.append(StoryStep(event.title, event.caption, "fused"))
                yield event
                self._cancel(req)
            if hybrid:
                with timer(timings, "denoise"):
                    if no_fusion:
                        # The no_fusion baseline remains direct deep output; table still requires this stage.
                        gain = luminance(output, cfg) / np.maximum(diagnosis.luminance, cfg["real"]["epsilon"])
                        denoised = DenoiseResult(output, gain, np.zeros_like(gain))
                    else:
                        denoised = denoise(output, diagnosis, recovery, cfg, recovery_enabled=recovery_enabled,
                                           uniform="no_gain_denoise" in req.ablate)
                    output = denoised.image
                with timer(timings, "render"):
                    visuals = [Visual("denoised", uint8(output), "stage", 768, False),
                               Visual("gain", colormap(denoised.gain, cfg["real"]["render"]["gain_max"]), "map", 512, True),
                               Visual("denoise_strength", uint8(denoised.strength), "map", 512, True)]
                caption = "Preserved the direct model baseline without postprocessing." if no_fusion else "Matched noise control to amplification and enforced the enabled local gain limits."
                event = self._stage("denoise", "Controlling amplified noise", caption,
                                    "Uniform denoise only for no_gain_denoise; no_recoverability removes all confidence controls.",
                                    visuals, "denoised", {"bypassed": no_fusion}, timings["denoise"])
                story.append(StoryStep(event.title, event.caption, "denoised"))
                yield event
                self._cancel(req)
            if output is None:
                raise EngineError("No valid restoration contributor was produced.")
            with timer(timings, "metrics"):
                metrics = request_metrics(uint8(image), uint8(output), valid, cfg)
            with timer(timings, "guardrails"):
                checks = audit(metrics, output, diagnosis, cfg)
            failed = [check.check for check in checks if check.passed is False]
            if failed and not audit_only:
                repair_attempted = True
                checks = [replace(check, action="Reduced deep contribution, highlight gain and local amplification.") if check.passed is False else check for check in checks]
                with timer(timings, "repair"):
                    mask = np.clip((diagnosis.illumination - cfg["fusion"]["highlight_start"]) / (1 - cfg["fusion"]["highlight_start"]), 0, 1)
                    repaired_fusion = fuse(image, deep_image, classical_image, diagnosis, recovery, cfg,
                                          recovery_enabled=recovery_enabled,
                                          alpha_fraction=cfg["repair"]["fusion_fraction"], repair_mask=mask)
                    safer_recovery = replace(recovery, gain_cap=np.maximum(1, recovery.gain_cap * cfg["repair"]["gain_fraction"]))
                    output = denoise(repaired_fusion.image, diagnosis, safer_recovery, cfg,
                                     recovery_enabled=recovery_enabled, uniform="no_gain_denoise" in req.ablate).image
                self._cancel(req)
                with timer(timings, "metrics"):
                    metrics = request_metrics(uint8(image), uint8(output), valid, cfg)
                with timer(timings, "guardrails"):
                    second = [replace(check, attempt=2) for check in audit(metrics, output, diagnosis, cfg)]
                checks.extend(second)
                failed = [check.check for check in second if check.passed is False]
                if failed:
                    # No third repair: select the already available conservative safe/classical path.
                    output = image ** cfg["fusion"]["safe_gamma"]
                    if classical_image is not None:
                        output = np.minimum(output, classical_image)
                    if recovery_enabled:
                        output = enforce_gain(output, diagnosis.luminance, safer_recovery, cfg)
                    with timer(timings, "metrics"):
                        metrics = request_metrics(uint8(image), uint8(output), valid, cfg)
                    with timer(timings, "guardrails"):
                        fallback_audit = [asdict(check) for check in audit(metrics, output, diagnosis, cfg)]
                    warnings.append(EngineWarning("guardrail_unresolved", "The single repair failed; a conservative final result was selected and its checks are recorded."))
            elif failed:
                warnings.append(EngineWarning("guardrail_audit_failed", "The baseline audit found quality issues; its direct output was preserved."))
            yield self._stage("guardrails", "Checking the final result",
                              "Recorded quality checks and any single conservative repair without rerunning the trained model.",
                              "At most two repair-attempt check sets; conservative fallback has a separate audit.", [], None,
                              {"audit_only": audit_only, "checks": [asdict(check) for check in checks],
                               "conservative_final_checks": fallback_audit}, timings["guardrails"])
            self._cancel(req)
            with timer(timings, "report"):
                method = deep_result.method if deep_result is not None else "classical"
                decisions = [
                    ("Used classical processing because the requested model was unavailable."
                     if method == "classical_fallback" else
                     "Used a verified restoration model." if method in ("cidnet", "zerodce") else
                     "Used classical processing as explicitly requested."),
                    "Preserved the original processing dimensions and supplied transparency.",
                    ("Used configured fixed settings because tuning was explicitly disabled." if hybrid else "Preserved direct baseline output without hybrid postprocessing."),
                    ("Applied input-confidence controls to hybrid processing." if hybrid and recovery_enabled and not no_fusion else "Preserved the explicitly requested baseline or confidence-control ablation."),
                    ("Attempted one conservative repair using cached model output." if repair_attempted else "Recorded quality checks without an automatic repair."),
                    "Left unavailable learned quality scores empty with reasons.",
                ]
                headline = f"{diagnosis.raw['severity'].capitalize()} low light processed using the recorded available restoration path."
            with timer(timings, "render"):
                final = final_visuals(output, req.alpha)
            digest = sha256(str(req.image.shape).encode())
            digest.update(memoryview(req.image))
            if req.alpha is not None:
                digest.update(req.alpha.tobytes())
            actual = "hybrid" if hybrid and method != "classical_fallback" else method
            timings["core_total"] = (perf_counter() - start) * 1000
            yield EnhanceResult(
                headline, diagnosis_chips(diagnosis, recovery), diagnosis.raw, diagnosis.hotspots, recovery.summary, candidates,
                timings.get("tuner"), scoring_recipe, checks,
                decisions, story, metrics,
                {"base_key": "final_preview", "L_key": "L_gray", "gamma_default": 1.0, "exposure_default": 1.0,
                 "gamma_range": [0.4, 2.5], "exposure_range": [0.25, 4.0]}, timings, warnings,
                {"requested_mode": req.mode, "actual_method": actual,
                 "fallback_chain": deep_result.fallback_chain if deep_result else [],
                 "checkpoint": deep_result.checkpoint if deep_result else None,
                 "checkpoint_hash": deep_result.checkpoint_hash if deep_result else None,
                 "precision": deep_result.precision if deep_result else "fp32",
                 "device": deep_result.device if deep_result else "cpu", "placement_policy": self._registry.policy,
                 "ablate": sorted(req.ablate), "seed": req.seed, "config_hash": cfg.digest,
                 "input_hash": digest.hexdigest()[:12], "core_version": __version__, "contract_version": CONTRACT_VERSION,
                 "repair_attempted": repair_attempted, "guardrails_unresolved": failed,
                 "conservative_final_checks": fallback_audit, "perf_levers": []}, final,
            )
        finally:
            # Model activations have already been released inside registry contexts;
            # closing the generator discards all run-local CPU arrays.
            pass

    @staticmethod
    def _stage(name: str, title: str, caption: str, note: str, visuals: list[Visual],
               primary: str | None, data: dict[str, Any], milliseconds: float) -> StageEvent:
        """Assemble a typed event without serialization or output I/O."""
        return StageEvent("stage", name, title, caption, note, visuals, primary, data, milliseconds)
