"""Deterministic CPU mock implementation of the complete frozen engine interface.

These cheap operations illustrate the contract, not trained restoration or IQA.
No files, locks, scheduling, model downloads, or transport code belong here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from time import perf_counter, sleep
from typing import Any, Iterator, Literal

import cv2
import numpy as np

from .config import load_fake_config
from .contracts import (
    CONTRACT_VERSION,
    Cancelled,
    Candidate,
    Chip,
    EngineWarning,
    EnhanceRequest,
    EnhanceResult,
    GuardrailCheck,
    RunItem,
    StageEvent,
    StoryStep,
    Visual,
)
from .validation import expected_stages, validate_request, validate_rgb

VERSION = "0.2.0"
_LEARNED = ("niqe", "brisque", "clipiqa", "musiq", "lpips", "ssim")


def _uint8(image: np.ndarray) -> np.ndarray:
    """Round normalized samples to uint8 without changing input storage."""
    return np.ascontiguousarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))


def _metric_rgb(image: np.ndarray) -> None:
    """Validate a metric image without imposing the live-frame size limit."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        from .contracts import InvalidRequest

        raise InvalidRequest("metric_image", "Metric images must be positive uint8 RGB arrays.")


class FakeMetricSuite:
    """Return measured simple statistics; unavailable real metrics are always None."""

    def paired(self, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
        """Compute real PSNR only; do not impersonate learned or structural metrics."""
        self._pair(pred, gt)
        diff = pred.astype(np.float64) - gt.astype(np.float64)
        mse = float(np.mean(diff * diff))
        pred_mean, gt_mean = float(np.mean(pred)), float(np.mean(gt))
        adjusted = None
        if pred_mean > 0:
            scaled = np.clip(pred.astype(np.float64) * (gt_mean / pred_mean), 0, 255)
            adjusted_mse = float(np.mean((scaled - gt) ** 2))
            if adjusted_mse > 0:
                adjusted = float(10 * np.log10(255**2 / adjusted_mse))
        return {
            "psnr": float(10 * np.log10(255**2 / mse)) if mse > 0 else None,
            "ssim": None,
            "lpips": None,
            "psnr_gt_mean": adjusted,
        }

    def unpaired(self, img: np.ndarray) -> dict[str, float | None]:
        """Expose all expected quality keys without inventing quality scores."""
        _metric_rgb(img)
        return dict.fromkeys(("niqe", "brisque", "clipiqa", "musiq"))

    def custom(self, inp: np.ndarray, out: np.ndarray) -> dict[str, float | None]:
        """Measure highlight clipping; defer colour, structure, and noise algorithms."""
        self._pair(inp, out)
        cutoff = load_fake_config()[0]["fake"]["highlight_threshold"] * 255
        return {
            "clip_pct_in": float(np.mean(np.any(inp > cutoff, axis=2)) * 100),
            "clip_pct_out": float(np.mean(np.any(out > cutoff, axis=2)) * 100),
            "chroma_delta_e": None,
            "edge_retention": None,
            "noise_sigma_in": None,
            "noise_sigma_out": None,
        }

    def confidence_errors(
        self, inp: np.ndarray, pred: np.ndarray, gt: np.ndarray
    ) -> dict[str, float | None]:
        """Defer confidence errors until real recoverability exists; retain all keys."""
        self._pair(inp, pred)
        self._pair(pred, gt)
        return dict.fromkeys(
            ("err_recovered", "err_estimated", "err_unrecoverable",
             "recovered_pct", "estimated_pct", "unrecoverable_pct")
        )

    def directions(self) -> dict[str, Literal["higher", "lower"]]:
        """Return metric direction metadata for available and future methods."""
        higher = ("psnr", "ssim", "psnr_gt_mean", "clipiqa", "musiq", "edge_retention")
        lower = (
            "lpips", "niqe", "brisque", "chroma_delta_e", "delta_e", "clip_pct_in",
            "clip_pct_out", "noise_sigma_in", "noise_sigma_out", "err_recovered",
            "err_estimated", "err_unrecoverable",
        )
        return {**dict.fromkeys(higher, "higher"), **dict.fromkeys(lower, "lower")}

    def unavailable(self) -> dict[str, str]:
        """Return reasons covering all nullable values, including perfect-match PSNR."""
        keys = _LEARNED + (
            "chroma_delta_e", "edge_retention", "noise_sigma_in", "noise_sigma_out",
            "err_recovered", "err_estimated", "err_unrecoverable", "recovered_pct",
            "estimated_pct", "unrecoverable_pct",
        )
        reasons = dict.fromkeys(keys, "Not implemented by the CPU fake engine in v0.1.0.")
        reasons["psnr"] = "Unavailable only for an exact match (infinite PSNR)."
        reasons["psnr_gt_mean"] = (
            "Unavailable for zero prediction mean or an exact mean-adjusted match."
        )
        return reasons

    @staticmethod
    def _pair(first: np.ndarray, second: np.ndarray) -> None:
        """Check metric image compatibility before arithmetic."""
        _metric_rgb(first)
        _metric_rgb(second)
        if first.shape != second.shape:
            from .contracts import InvalidRequest

            raise InvalidRequest("metric_shape", "Metric images must have the same shape.")


class FakeEngine:
    """CPU-only contract fixture supporting every mode and legal ablation combination."""

    def __init__(self, stage_delay_ms: int = 0, seed: int = 0) -> None:
        """Validate fixture controls and load package-local mock configuration."""
        if type(stage_delay_ms) is not int or stage_delay_ms < 0:
            raise ValueError("stage_delay_ms must be a nonnegative integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._delay = stage_delay_ms / 1000
        self._seed = seed
        self._warm = False
        self._config, self._config_hash = load_fake_config()

    def warmup(self) -> None:
        """Mark the fake engine warm; no model, device, or file side effects occur."""
        self._warm = True

    def health(self) -> dict[str, Any]:
        """Report real CPU/mock status without claiming any trained model is available."""
        return {
            "fake": True, "ready": True, "hybrid_ready": False, "warm": self._warm,
            "device": "cpu", "placement_policy": "resident",
            "models": {
                "cidnet": {"available": False, "checkpoints": []},
                "zerodce": {"available": False},
                "iqa": dict.fromkeys(("clipiqa", "niqe", "brisque", "musiq", "lpips"), False),
            },
            "vram_mb": {"total": 0, "allocated": 0, "reserved": 0, "peak": 0},
            "degraded": ["fake_engine", "real_engine_not_implemented"],
            "core_version": VERSION, "contract_version": CONTRACT_VERSION,
        }

    def public_config(self) -> dict[str, Any]:
        """Return a fresh configuration copy that callers cannot use to mutate the engine."""
        return deepcopy(self._config)

    def metric_suite(self) -> FakeMetricSuite:
        """Return the stateless honest metric fixture."""
        return FakeMetricSuite()

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Brighten a validated uint8 frame using the configured mock gamma lookup table."""
        validate_rgb(frame, floating=False)
        gamma = self._config["fake"]["live_gamma"]
        table = _uint8((np.arange(256, dtype=np.float32) / 255) ** gamma)
        return np.ascontiguousarray(table[frame])

    @staticmethod
    def _cancel(req: EnhanceRequest) -> None:
        """Check cancellation only at documented processing boundaries."""
        if req.cancel is not None and req.cancel.is_set():
            raise Cancelled("Enhancement cancelled.")

    def _pause(self, req: EnhanceRequest) -> None:
        """Honor the demo delay after a stage, with prompt cancellation when provided."""
        if self._delay:
            if req.cancel is None:
                sleep(self._delay)
            elif req.cancel.wait(self._delay):
                raise Cancelled("Enhancement cancelled during the stage delay.")
        self._cancel(req)

    def run(self, req: EnhanceRequest) -> Iterator[RunItem]:
        """Yield contract-ordered array events, followed by exactly one final result."""
        validate_request(req)
        self._cancel(req)
        started = perf_counter()
        cfg = self._config["fake"]
        schedule = expected_stages(req)
        candidates: list[Candidate] = []
        story: list[StoryStep] = []
        timings: dict[str, float] = {}
        cache: dict[str, Any] = {}
        # Locals holding arrays are inside try/finally so close() releases references.
        try:
            image = req.image.copy()
            cache["image"] = image
            output = image
            for name in schedule:
                self._cancel(req)
                stage_start = perf_counter()
                visuals: list[Visual] = []
                data: dict[str, Any] = {"fake": True}
                primary: str | None = None
                if name == "diagnose":
                    y = np.sum(image * np.asarray(cfg["luminance_weights"], np.float32), axis=2)
                    illumination = np.max(image, axis=2)
                    cache["luminance"] = y
                    cache["illumination"] = illumination
                    gray = _uint8(illumination)
                    color = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                    heightmap = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
                    noise = np.zeros_like(gray)
                    visuals = [
                        Visual("illumination", color, "stage", 768, True),
                        Visual("L_gray", gray, "map", 512, True),
                        Visual("noise_map", noise, "map", 512, True),
                        Visual("snr", color.copy(), "map", 512, True),
                        Visual("heightmap", heightmap, "map", None, True),
                    ]
                    primary, title = "illumination", "Finding the light"
                    caption = "This demonstration marks bright and dark areas without measuring sensor noise."
                    data.update(mean_y=float(np.mean(y)), noise_sigma=None,
                                unavailable={"noise_sigma": "No sensor-noise estimator in the fake engine."})
                elif name == "recoverability":
                    y = cache["luminance"]
                    classes = (y >= cfg["confidence_low"]).astype(np.uint8)
                    classes += (y >= cfg["confidence_high"]).astype(np.uint8)
                    valid = np.ones(y.shape, bool) if req.alpha is None else req.alpha > 0
                    count = int(np.count_nonzero(valid))
                    percentages = [
                        float(np.count_nonzero((classes == i) & valid) * 100 / count)
                        if count else (100.0 if i == 0 else 0.0)
                        for i in range(3)
                    ]
                    recovery = {
                        "recovered_pct": percentages[2], "estimated_pct": percentages[1],
                        "unrecoverable_pct": percentages[0], "estimator_confidence": 0.0,
                        "sentence": "These demonstration colours show brightness groups, not measured recoverability.",
                    }
                    cache["recovery"] = recovery
                    colors = np.asarray(cfg["recovery_colors"], np.uint8)[classes]
                    alpha = np.full(y.shape, round(cfg["overlay_alpha"] * 255), np.uint8)
                    if req.alpha is not None:
                        alpha[~valid] = 0
                    visuals = [Visual("recoverability_overlay", np.dstack([colors, alpha]), "map", 512, True)]
                    primary, title = "recoverability_overlay", "Showing weak light"
                    caption = "These colours illustrate confidence groups; real signal estimates arrive in a later release."
                    data.update(recovery)
                elif name == "tuner":
                    seed_bytes = sha256(f"{self._seed}:{req.seed}".encode()).digest()
                    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
                    gammas = list(rng.uniform(*cfg["gamma_range"], size=10))
                    gammas.extend([cfg["conservative_gamma"], cfg["identity_gamma"]])
                    thumb_source = self._thumbnail(image)
                    arrays = [_uint8(thumb_source**g) for g in gammas]
                    totals = [
                        -abs(float(np.mean(a)) / 255 - cfg["brightness_target"])
                        - cfg["clipping_weight"] * float(np.mean(np.any(a > cfg["highlight_threshold"] * 255, axis=2)))
                        for a in arrays
                    ]
                    winner = int(np.argmax(totals))
                    for idx, (gamma, thumb) in enumerate(zip(gammas, arrays)):
                        self._cancel(req)
                        key = f"cand_{idx:02d}"
                        chosen = idx == winner
                        candidate = Candidate(
                            idx, {"gamma": float(gamma)},
                            {"brightness_error": abs(float(np.mean(thumb)) / 255 - cfg["brightness_target"]),
                             "clipiqa": None, "niqe": None, "norm_brightness": float(totals[idx])},
                            float(totals[idx]), True,
                            [] if chosen else ["Another demonstration brightness setting ranked higher."],
                            key, chosen,
                        )
                        candidates.append(candidate)
                        yield StageEvent(
                            "candidate", key, "Trying a light setting",
                            "This demonstration ranks brightness settings without learned quality scores.",
                            "Mock brightness surrogate; CLIP-IQA and NIQE are unavailable.",
                            [Visual(key, thumb, "thumb", 128, False)], key, asdict(candidate), 0.0,
                        )
                    cache["gamma"] = float(gammas[winner])
                    primary, title = f"cand_{winner:02d}", "Choosing a setting"
                    caption = "The highlighted setting best matches this demonstration's brightness preference."
                    data.update(chosen_id=winner, scoring_recipe="fake_brightness+custom")
                elif name == "classical":
                    gamma = cache.get("gamma", cfg["classical_gamma"])
                    output = image**gamma
                    cache["classical"] = output
                    reflectance = image / np.maximum(cache["illumination"][..., None], cfg["epsilon"])
                    visuals = [Visual("classical", _uint8(output), "stage", 768, False),
                               Visual("reflectance", _uint8(reflectance), "stage", 768, False)]
                    primary, title = "classical", "Lifting dark areas"
                    caption = "A simple brightness curve illustrates the classical processing stage."
                elif name == "deep":
                    output = image ** cfg["deep_gamma"]
                    cache["deep"] = output
                    visuals = [Visual("deep", _uint8(output), "stage", 768, False)]
                    primary, title = "deep", "Previewing the model stage"
                    caption = "This is a brightness preview; no trained model has processed this image."
                elif name == "fusion":
                    highlight = (cache["illumination"] > cfg["highlight_start"]).astype(np.float32)
                    weights = np.zeros((*image.shape[:2], 3), np.float32)
                    active = [i for i, key in enumerate(("deep", "classical")) if key in cache]
                    for i in active:
                        weights[..., i] = (1 - highlight) / len(active)
                    weights[..., 2] = highlight
                    output = image ** cfg["safe_gamma"] * weights[..., 2:3]
                    for i, key in enumerate(("deep", "classical")):
                        if key in cache:
                            output += cache[key] * weights[..., i:i + 1]
                    visuals = [Visual("fused", _uint8(output), "stage", 768, False),
                               Visual("fusion_map", _uint8(weights), "map", 512, True)]
                    primary, title = "fused", "Combining light settings"
                    caption = "The demonstration mixes two brightness curves while protecting bright areas."
                elif name == "denoise":
                    # Deliberately no pretend noise removal or gain-cap guarantee in a fixture.
                    gain = np.mean(output, axis=2) / np.maximum(np.mean(image, axis=2), cfg["epsilon"])
                    visuals = [
                        Visual("denoised", _uint8(output), "stage", 768, False),
                        Visual("gain", _uint8(gain / max(1.0, float(np.max(gain)))), "map", 512, True),
                        Visual("denoise_strength", np.zeros(image.shape[:2], np.uint8), "map", 512, True),
                    ]
                    primary, title = "denoised", "Previewing noise control"
                    caption = "Noise removal is unavailable here; this preview preserves the preceding image."
                else:
                    title = "Checking the demonstration"
                    caption = "Real quality checks are unavailable in this demonstration release."
                    data.update(audit_only=req.mode in ("dl", "zerodce"), disabled="no_guardrails" in req.ablate)
                elapsed = (perf_counter() - stage_start) * 1000
                timings[name] = elapsed
                event = StageEvent("stage", name, title, caption, "CPU fake engine; illustrative data only.",
                                   visuals, primary, data, elapsed)
                if primary is not None:
                    story.append(StoryStep(title, caption, primary))
                yield event
                self._pause(req)
                if name == "diagnose":
                    yield StageEvent(
                        "warning", "fake_engine", "Demonstration engine",
                        "This release demonstrates the interface and does not perform trained restoration.",
                        "Real inference and IQA are not implemented in core-v0.1.0.", [], None,
                        {"code": "fake_engine", "fake": True}, 0.0,
                    )
                    self._cancel(req)

            self._cancel(req)
            rgb = _uint8(output)
            final = rgb if req.alpha is None else np.dstack([rgb, _uint8(req.alpha)])
            metrics = self._request_metrics(_uint8(image), rgb)
            result = EnhanceResult(
                headline="Demonstration complete: dark areas lifted with simple brightness curves.",
                chips=[Chip("Demonstration", None, "low"),
                       Chip("Input brightness", float(np.mean(image)), "med")],
                diagnosis_raw={"fake": True, "mean_y": float(np.mean(cache["luminance"])),
                               "noise_sigma": None, "unavailable": {"noise_sigma": "Not measured by the fake engine."}},
                hotspots=[], recoverability=cache["recovery"], candidates=candidates,
                search_ms=timings.get("tuner"),
                scoring_recipe="fake_brightness+custom" if candidates else "not_run",
                guardrails=[GuardrailCheck(key, None, float(value), None, 1, None,
                                           "Real guardrails are unavailable in the fake engine.")
                            for key, value in cfg["guardrail_thresholds"].items()],
                decisions=[
                    "Used a CPU demonstration instead of a trained restoration model.",
                    "Displayed brightness groups instead of measured signal confidence.",
                    "Preserved the image dimensions and any supplied transparency.",
                    "Left unavailable quality measurements empty rather than inventing scores.",
                ],
                story=story, metrics=metrics,
                relight={"base_key": "final_preview", "L_key": "L_gray", "gamma_default": 1.0,
                         "exposure_default": 1.0, "gamma_range": [0.4, 2.5], "exposure_range": [0.25, 4.0]},
                timings_ms=timings,
                warnings=[EngineWarning("fake_engine", "Illustrative CPU processing, not trained restoration."),
                          EngineWarning("iqa_unavailable", "Learned image quality metrics are unavailable in this release.")],
                provenance={
                    "requested_mode": req.mode,
                    "actual_method": {"dl": "cidnet", "zerodce": "zerodce"}.get(req.mode, req.mode),
                    "fake": True, "fallback_chain": [], "checkpoint": None, "checkpoint_hash": None,
                    "precision": "fp32", "device": "cpu", "placement_policy": "resident",
                    "ablate": sorted(req.ablate), "seed": req.seed, "fake_seed": self._seed,
                    "input_hash": self._input_hash(req), "config_hash": self._config_hash,
                    "core_version": VERSION, "contract_version": CONTRACT_VERSION,
                    "repair_attempted": False, "guardrails_unresolved": [], "perf_levers": [],
                },
                visuals=[Visual("final", final, "final", None, True),
                         Visual("final_preview", rgb.copy(), "stage", 1280, False)],
            )
            timings["core_total"] = (perf_counter() - started) * 1000
            yield result
        finally:
            cache.clear()
            # All run-owned arrays are locals; generator.close() discards the frame.

    @staticmethod
    def _thumbnail(image: np.ndarray) -> np.ndarray:
        """Build an aspect-preserving candidate proxy without upscaling tiny inputs."""
        scale = min(1.0, 128 / max(image.shape[:2]))
        width = max(1, round(image.shape[1] * scale))
        height = max(1, round(image.shape[0] * scale))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _input_hash(req: EnhanceRequest) -> str:
        """Include geometry, RGB bytes, and optional alpha in the normalized-input hash."""
        digest = sha256(str(req.image.shape).encode())
        digest.update(memoryview(req.image))
        if req.alpha is not None:
            digest.update(req.alpha.tobytes())
        return digest.hexdigest()[:12]

    def _request_metrics(self, inp: np.ndarray, out: np.ndarray) -> dict[str, Any]:
        """Build the exact request metric shape using 512-long-side input/output proxies."""
        scale = min(1.0, 512 / max(inp.shape[:2]))
        size = (max(1, round(inp.shape[1] * scale)), max(1, round(inp.shape[0] * scale)))
        first = cv2.resize(inp, size, interpolation=cv2.INTER_AREA)
        last = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
        suite = self.metric_suite()
        custom = suite.custom(first, last)
        before = suite.unpaired(first)
        after = suite.unpaired(last)
        before.pop("musiq")
        after.pop("musiq")
        unavailable = suite.unavailable()
        unavailable["musiq"] = "Not computed at request time."
        unavailable.update({
            "noise_sigma.before": "No real noise estimator in the fake engine.",
            "noise_sigma.after": "No real noise estimator in the fake engine.",
            "delta_e": "No real chroma metric in the fake engine.",
        })
        return {
            "resolution": 512, "before": before, "after": after,
            "edge_retention": None, "noise_sigma": {"before": None, "after": None},
            "chroma_delta_e": None, "delta_e": None,
            "clip_pct": {"before": custom["clip_pct_in"], "after": custom["clip_pct_out"]},
            "directions": suite.directions(), "unavailable": unavailable,
        }
