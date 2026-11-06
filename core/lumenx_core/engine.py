"""Real classical-only v0.2 engine: array events, no transport, storage or scheduling."""

from __future__ import annotations

from dataclasses import asdict
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
from .metrics import ClassicalMetricSuite, request_metrics
from .recoverability import estimate_recoverability
from .render import colormap, diagnosis_visuals, final_visuals, recovery_visual, uint8
from .report import classical_report
from .timing import timer
from .validation import validate_request


class RealEngine:
    """CPU classical restoration; unsupported neural modes fail before any yield."""

    def __init__(self, config_path: str | None = None, device: str | None = None) -> None:
        """Load immutable configuration once; do not download or instantiate models."""
        if device not in (None, "cpu"):
            raise EngineUnavailable("v0.2.0 supports CPU classical processing only; use device='cpu'.")
        try:
            self._cfg: EngineConfig = load_config(config_path)
        except (OSError, TypeError, ValueError) as exc:
            raise EngineError(f"Invalid engine configuration: {exc}") from exc
        cv2.setNumThreads(self._cfg["real"]["threads"])
        self._warm = False

    def warmup(self) -> None:
        """Initialize OpenCV colour/filter kernels once with a tiny classical request."""
        if not self._warm:
            image = np.full((32, 32, 3), 0.1, np.float32)
            for _ in self.run(EnhanceRequest(image, (32, 32), mode="classical")):
                pass
            self._warm = True

    def public_config(self) -> dict[str, Any]:
        """Return an isolated JSON-safe copy, not a reference to engine configuration."""
        return self._cfg.public()

    def metric_suite(self) -> ClassicalMetricSuite:
        """Provide actual custom metrics and explicit reasons for absent learned IQA."""
        return ClassicalMetricSuite(self._cfg)

    def health(self) -> dict[str, Any]:
        """Distinguish available classical processing from absent hybrid/live/IQA models."""
        from . import __version__

        return {
            "fake": False, "ready": True, "hybrid_ready": False, "warm": self._warm,
            "device": "cpu", "placement_policy": "resident", "modes": ["classical"],
            "models": {"cidnet": {"available": False, "checkpoints": []},
                       "zerodce": {"available": False},
                       "iqa": dict.fromkeys(("clipiqa", "niqe", "brisque", "musiq", "lpips"), False)},
            "vram_mb": {"total": 0, "allocated": 0, "reserved": 0, "peak": 0},
            "degraded": ["neural_modes_unavailable", "iqa_unavailable", "guardrails_audit_only"],
            "core_version": __version__, "contract_version": CONTRACT_VERSION,
        }

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Refuse live processing until the actual Zero-DCE++ wrapper is released."""
        from .validation import validate_rgb

        validate_rgb(frame, floating=False)
        raise EngineUnavailable("Real live enhancement requires the Zero-DCE++ release (v0.3.0).")

    @staticmethod
    def _cancel(req: EnhanceRequest) -> None:
        """Honor cancellation between processing stages and array rendering."""
        if req.cancel is not None and req.cancel.is_set():
            raise Cancelled("Enhancement cancelled.")

    def run(self, req: EnhanceRequest) -> Iterator[RunItem]:
        """Yield real classical stages and exactly one terminal EnhanceResult."""
        from . import __version__

        validate_request(req)
        if req.mode != "classical":
            raise EngineUnavailable(f"Mode {req.mode!r} requires the neural-engine release (v0.3.0).")
        self._cancel(req)
        start = perf_counter()
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

    @staticmethod
    def _stage(name: str, title: str, caption: str, note: str, visuals: list[Visual],
               primary: str | None, data: dict[str, Any], milliseconds: float) -> StageEvent:
        """Assemble a typed event without serialization or output I/O."""
        return StageEvent("stage", name, title, caption, note, visuals, primary, data, milliseconds)
