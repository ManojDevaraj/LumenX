"""Executable validation of frozen LumenX contract v1.0 run output.

This module checks values rather than trusting dataclass annotations. It intentionally
accepts the required `L_gray` visual spelling despite the contract's generic lowercase
key comment. It uses the exact section 5 schedule, including ablated hybrid stages.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import math
import re
from typing import Any

import numpy as np

from .contracts import (
    CONTRACT_VERSION,
    Candidate,
    Chip,
    EngineWarning,
    EnhanceRequest,
    EnhanceResult,
    GuardrailCheck,
    InvalidRequest,
    RunItem,
    StageEvent,
    StoryStep,
    Visual,
)
from .validation import expected_stages, validate_request

_VISUALS: dict[str, dict[str, tuple[str, int | None, bool]]] = {
    "diagnose": {
        "illumination": ("stage", 768, True), "L_gray": ("map", 512, True),
        "noise_map": ("map", 512, True), "snr": ("map", 512, True),
        "heightmap": ("map", None, True),
    },
    "recoverability": {"recoverability_overlay": ("map", 512, True)},
    "classical": {"classical": ("stage", 768, False), "reflectance": ("stage", 768, False)},
    "deep": {"deep": ("stage", 768, False)},
    "fusion": {"fused": ("stage", 768, False), "fusion_map": ("map", 512, True)},
    "denoise": {
        "denoised": ("stage", 768, False), "gain": ("map", 512, True),
        "denoise_strength": ("map", 512, True),
    },
    "final": {"final": ("final", None, True), "final_preview": ("stage", 1280, False)},
}
_PRIMARY = {
    "diagnose": "illumination", "recoverability": "recoverability_overlay",
    "classical": "classical", "deep": "deep", "fusion": "fused", "denoise": "denoised",
}
_PROVENANCE_KEYS = {
    "requested_mode", "actual_method", "fallback_chain", "checkpoint", "precision", "device",
    "placement_policy", "ablate", "seed", "config_hash", "input_hash", "checkpoint_hash",
    "core_version", "contract_version", "repair_attempted", "guardrails_unresolved",
}


def _number(value: Any, *, nullable: bool = False) -> bool:
    """Recognize finite built-in numbers, excluding booleans and NumPy scalars."""
    return (nullable and value is None) or (
        type(value) in (int, float) and math.isfinite(value)
    )


def _json_errors(value: Any, path: str, ancestors: set[int] | None = None) -> list[str]:
    """Find non-JSON values, nonfinite numbers, and container cycles without encoding."""
    if value is None or type(value) in (str, bool, int):
        return []
    if type(value) is float:
        return [] if math.isfinite(value) else [f"{path}: nonfinite number"]
    if type(value) not in (dict, list):
        return [f"{path}: not JSON-safe ({type(value).__name__})"]
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        return [f"{path}: cyclic container"]
    active = ancestors | {id(value)}
    errors: list[str] = []
    if type(value) is dict:
        for key, entry in value.items():
            if type(key) is not str:
                errors.append(f"{path}: dictionary key is not a string")
            errors.extend(_json_errors(entry, f"{path}.{key}", active))
    else:
        for i, entry in enumerate(value):
            errors.extend(_json_errors(entry, f"{path}[{i}]", active))
    return errors


def validate_run(items: list[RunItem], req: EnhanceRequest) -> list[str]:
    """Return all detected contract violations; an empty list means a valid run.

    Invalid requests are reported as violations too. Callers should not pass partial
    cancelled runs: a cancelled generator intentionally has no terminal result.
    """
    errors: list[str] = []
    try:
        validate_request(req)
    except InvalidRequest as exc:
        return [f"request: {exc.code}: {exc}"]
    if not isinstance(items, list):
        return ["items: expected a list of completed run items"]
    if not items:
        return ["run: no items or final EnhanceResult"]
    results = [i for i, item in enumerate(items) if isinstance(item, EnhanceResult)]
    if len(results) != 1 or results[0] != len(items) - 1:
        errors.append("run: exactly one EnhanceResult must be the final item")
    seen: dict[str, Visual] = {}
    stages: list[str] = []
    candidate_events: list[Candidate] = []
    candidate_data: list[dict[str, Any]] = []
    tuner_primary: str | None = None
    tuner_seen = False
    result: EnhanceResult | None = None

    def check(condition: bool, message: str) -> None:
        """Accumulate a failed invariant without stopping remaining checks."""
        if not condition:
            errors.append(message)

    def words(value: Any, limit: int, path: str) -> None:
        """Check nonempty user-facing copy and its contract word budget."""
        check(isinstance(value, str) and bool(value.strip()), f"{path}: expected nonempty text")
        if isinstance(value, str):
            check(len(value.split()) <= limit, f"{path}: exceeds {limit} words")

    def add_visuals(visuals: Any, path: str) -> dict[str, Visual]:
        """Validate and register unique visuals without copying their arrays."""
        current: dict[str, Visual] = {}
        if not isinstance(visuals, list):
            errors.append(f"{path}: expected a visual list")
            return current
        for v in visuals:
            if not isinstance(v, Visual):
                errors.append(f"{path}: expected Visual")
                continue
            if not isinstance(v.key, str):
                errors.append(f"{path}: visual key must be a string")
                continue
            check(bool(re.fullmatch(r"(?:[a-z0-9_]+|L_gray)", v.key)), f"{path}: invalid key {v.key}")
            check(v.key != "input", f"{path}: reserved input key emitted")
            check(v.key not in seen, f"{path}: duplicate visual key {v.key}")
            check(v.role in ("stage", "map", "thumb", "final"), f"{path}.{v.key}: invalid role")
            check(v.max_side is None or (type(v.max_side) is int and v.max_side > 0),
                  f"{path}.{v.key}: invalid max_side")
            check(type(v.lossless) is bool, f"{path}.{v.key}: lossless must be bool")
            if not isinstance(v.image, np.ndarray):
                errors.append(f"{path}.{v.key}: image is not an array")
            else:
                shape = v.image.shape
                check(v.image.dtype == np.uint8, f"{path}.{v.key}: expected uint8")
                valid_shape = v.image.ndim == 2 or (v.image.ndim == 3 and shape[2] in (3, 4))
                check(valid_shape and all(x > 0 for x in shape), f"{path}.{v.key}: invalid visual shape")
                if v.key in ("L_gray", "heightmap"):
                    check(v.image.ndim == 2, f"{path}.{v.key}: expected gray image")
                if v.key == "heightmap":
                    check(shape == (128, 128), f"{path}.heightmap: expected 128 x 128")
                if v.key == "recoverability_overlay":
                    check(shape == (*req.image.shape[:2], 4), f"{path}: overlay must be full-size RGBA")
                if v.key == "final":
                    channels = 4 if req.alpha is not None else 3
                    check(shape == (*req.image.shape[:2], channels), f"{path}: final shape/alpha mismatch")
                    if channels == 4 and valid_shape and shape == (*req.image.shape[:2], 4):
                        expected = np.rint(req.alpha * 255).astype(np.uint8)
                        check(np.array_equal(v.image[..., 3], expected), f"{path}: final alpha was changed")
                if v.key not in ("heightmap", "final", "final_preview") and not v.key.startswith("cand_"):
                    check(shape[:2] == req.image.shape[:2], f"{path}.{v.key}: processed geometry mismatch")
            seen[v.key] = v
            current[v.key] = v
        return current

    def required(current: dict[str, Visual], spec: dict, path: str) -> None:
        """Check each required key and its transport metadata in its owning event."""
        for key, (role, side, lossless) in spec.items():
            if key not in current:
                errors.append(f"{path}: missing required visual {key}")
            else:
                v = current[key]
                check((v.role, v.max_side, v.lossless) == (role, side, lossless),
                      f"{path}.{key}: incorrect role/max_side/lossless")

    def check_candidate(c: Any, path: str) -> bool:
        """Validate candidate values separately from reference and aggregate checks."""
        if not isinstance(c, Candidate):
            errors.append(f"{path}: expected Candidate")
            return False
        check(type(c.id) is int and 0 <= c.id < 12, f"{path}: invalid candidate id")
        check(type(c.chosen) is bool and type(c.feasible) is bool, f"{path}: invalid candidate flags")
        check(_number(c.total, nullable=True), f"{path}: total must be finite or None")
        for field_name in ("params", "scores"):
            values = getattr(c, field_name)
            check(isinstance(values, dict), f"{path}.{field_name}: expected dict")
            if isinstance(values, dict):
                for key, value in values.items():
                    check(type(key) is str and _number(value, nullable=field_name == "scores"),
                          f"{path}.{field_name}: invalid numeric entry")
        check(isinstance(c.reasons, list) and all(isinstance(x, str) for x in c.reasons),
              f"{path}: invalid reasons")
        errors.extend(_json_errors(asdict(c), path))
        return True

    for index, item in enumerate(items):
        path = f"items[{index}]"
        if isinstance(item, EnhanceResult):
            result = item
            current = add_visuals(item.visuals, path)
            required(current, _VISUALS["final"], path)
            continue
        if not isinstance(item, StageEvent):
            errors.append(f"{path}: expected StageEvent or EnhanceResult")
            continue
        words(item.title, 6, f"{path}.title")
        words(item.caption, 20, f"{path}.caption")
        check(isinstance(item.tech_note, str), f"{path}: tech_note must be text")
        check(_number(item.t_ms) and item.t_ms >= 0, f"{path}: invalid t_ms")
        errors.extend(_json_errors(item.data, f"{path}.data"))
        current = add_visuals(item.visuals, path)
        if item.primary_key is not None:
            check(isinstance(item.primary_key, str) and item.primary_key in seen,
                  f"{path}: unresolved primary_key")
        if item.kind == "stage":
            stages.append(item.name)
            if item.name in _VISUALS:
                required(current, _VISUALS[item.name], path)
            if item.name in _PRIMARY:
                check(item.primary_key == _PRIMARY[item.name], f"{path}: incorrect primary_key")
            if item.name == "tuner":
                tuner_seen = True
                tuner_primary = item.primary_key
                check(len(candidate_events) == 12, f"{path}: tuner must follow all twelve candidates")
        elif item.kind == "candidate":
            check(stages == ["diagnose", "recoverability"] and not tuner_seen,
                  f"{path}: candidate outside the pre-tuner phase")
            check(item.t_ms == 0, f"{path}: candidate time must be zero")
            if isinstance(item.data, dict):
                try:
                    c = Candidate(**item.data)
                except TypeError:
                    errors.append(f"{path}: candidate data does not match Candidate")
                else:
                    if check_candidate(c, path):
                        candidate_events.append(c)
                        candidate_data.append(item.data)
                        key = f"cand_{c.id:02d}" if type(c.id) is int else "invalid"
                        check(item.name == key and c.thumb_key == key and item.primary_key == key,
                              f"{path}: candidate name/id/thumbnail mismatch")
                        required(current, {key: ("thumb", 128, False)}, path)
            else:
                errors.append(f"{path}: candidate data must be dict")
        elif item.kind == "warning":
            check(bool(stages) and stages[0] == "diagnose", f"{path}: warning before diagnose")
            check(item.visuals == [] and item.primary_key is None, f"{path}: warning must not emit visuals")
            check(item.t_ms == 0, f"{path}: warning time must be zero")
        else:
            errors.append(f"{path}: invalid event kind")

    check(stages == expected_stages(req), "run: incorrect stage order or mode/ablation schedule")
    tuning = req.mode == "hybrid" and "no_tuner" not in req.ablate
    expected_count = 12 if tuning else 0
    check(len(candidate_events) == expected_count, "run: incorrect candidate event count")
    if tuning:
        check([c.id for c in candidate_events] == list(range(12)), "run: candidate ids must be 0 through 11")
        chosen = [c for c in candidate_events if c.chosen]
        check(len(chosen) == 1, "run: exactly one candidate must be chosen")
        if len(chosen) == 1:
            check(tuner_primary == chosen[0].thumb_key, "run: tuner primary is not the winning candidate")
    if result is None:
        return errors

    # Check JSON safety on every final field except ndarray-bearing visuals.
    for field in fields(result):
        if field.name == "visuals":
            continue
        value = getattr(result, field.name)
        if is_dataclass(value):
            value = asdict(value)
        elif isinstance(value, list):
            value = [asdict(x) if is_dataclass(x) else x for x in value]
        # Hotspot tuples are explicitly required by the frozen dataclass.
        if field.name == "hotspots" and isinstance(value, list):
            value = [list(x) if isinstance(x, tuple) else x for x in value]
        errors.extend(_json_errors(value, f"result.{field.name}"))

    check(isinstance(result.headline, str) and bool(result.headline.strip()), "result: missing headline")
    check(isinstance(result.decisions, list) and 4 <= len(result.decisions) <= 7,
          "result: expected four to seven decisions")
    if isinstance(result.decisions, list):
        check(all(isinstance(x, str) and bool(x.strip()) for x in result.decisions), "result: invalid decisions")
    check(isinstance(result.chips, list), "result: chips must be a list")
    if isinstance(result.chips, list):
        for chip in result.chips:
            check(isinstance(chip, Chip), "result: invalid Chip")
            if isinstance(chip, Chip):
                check(chip.severity in ("high", "med", "low") and _number(chip.value, nullable=True),
                      "result: invalid chip severity/value")
    check(isinstance(result.candidates, list), "result: candidates must be a list")
    if isinstance(result.candidates, list):
        valid = [c for c in result.candidates if check_candidate(c, "result.candidates")]
        check(len(valid) == expected_count, "result: incorrect candidates length")
        check([asdict(c) for c in valid] == candidate_data, "result: candidates differ from emitted events")
        for c in valid:
            check(isinstance(c.thumb_key, str) and c.thumb_key in seen, "result: missing candidate thumbnail")
    check(_number(result.search_ms) if tuning else result.search_ms is None, "result: invalid search_ms")
    check(isinstance(result.scoring_recipe, str) and bool(result.scoring_recipe), "result: missing scoring recipe")

    if isinstance(result.story, list):
        for step in result.story:
            if not isinstance(step, StoryStep):
                errors.append("result.story: expected StoryStep")
                continue
            words(step.title, 6, "result.story.title")
            words(step.caption, 20, "result.story.caption")
            for key in (step.visual_key, step.highlight_key):
                if key is not None:
                    check(isinstance(key, str) and key in seen, "result.story: unresolved visual key")
    else:
        errors.append("result.story: expected a list")

    if isinstance(result.hotspots, list):
        for box in result.hotspots:
            valid = isinstance(box, tuple) and len(box) == 4 and all(type(x) is int for x in box)
            check(valid, "result.hotspots: expected integer (x, y, width, height)")
            if valid:
                x, y, width, height = box
                check(x >= 0 and y >= 0 and width > 0 and height > 0
                      and x + width <= req.image.shape[1] and y + height <= req.image.shape[0],
                      "result.hotspots: outside processed geometry")
    else:
        errors.append("result.hotspots: expected list")

    _check_result_mappings(result, req, check, errors)
    return errors


def _check_result_mappings(result: EnhanceResult, req: EnhanceRequest, check: Any, errors: list[str]) -> None:
    """Validate recovery, metrics, provenance, relight, checks and warning structures."""
    recovery = result.recoverability
    if isinstance(recovery, dict):
        pcts = [recovery.get(f"{name}_pct") for name in ("recovered", "estimated", "unrecoverable")]
        valid = all(_number(x) and 0 <= x <= 100 for x in pcts)
        check(valid, "result.recoverability: invalid area percentages")
        if valid:
            check(abs(sum(pcts) - 100) <= 0.1, "result.recoverability: percentages do not sum to 100")
        confidence = recovery.get("estimator_confidence")
        check(_number(confidence) and 0 <= confidence <= 1, "result.recoverability: invalid estimator confidence")
        check(isinstance(recovery.get("sentence"), str), "result.recoverability: missing sentence")
    else:
        errors.append("result.recoverability: expected dict")

    expected_relight = {
        "base_key": "final_preview", "L_key": "L_gray", "gamma_default": 1.0,
        "exposure_default": 1.0, "gamma_range": [0.4, 2.5], "exposure_range": [0.25, 4.0],
    }
    check(result.relight == expected_relight, "result.relight: incorrect contract settings")
    provenance = result.provenance
    if isinstance(provenance, dict):
        check(_PROVENANCE_KEYS <= provenance.keys(), "result.provenance: missing required keys")
        check(provenance.get("requested_mode") == req.mode, "result.provenance: requested_mode mismatch")
        check(provenance.get("ablate") == sorted(req.ablate), "result.provenance: ablation mismatch")
        check(provenance.get("seed") == req.seed, "result.provenance: seed mismatch")
        check(provenance.get("contract_version") == CONTRACT_VERSION, "result.provenance: contract version mismatch")
        check(provenance.get("placement_policy") in ("resident", "swap"), "result.provenance: invalid placement policy")
        check(provenance.get("precision") in ("fp16", "fp32"), "result.provenance: invalid precision")
        check(provenance.get("actual_method") in ("hybrid", "classical", "cidnet", "zerodce", "classical_fallback"),
              "result.provenance: invalid actual_method")
    else:
        errors.append("result.provenance: expected dict")
    if isinstance(result.timings_ms, dict):
        check("core_total" in result.timings_ms, "result.timings_ms: missing core_total")
        check(all(type(k) is str and _number(v) and v >= 0 for k, v in result.timings_ms.items()),
              "result.timings_ms: expected nonnegative finite values")
    else:
        errors.append("result.timings_ms: expected dict")
    if isinstance(result.guardrails, list):
        for guard in result.guardrails:
            if not isinstance(guard, GuardrailCheck):
                errors.append("result.guardrails: expected GuardrailCheck")
                continue
            check(guard.check in ("overexposure", "color_shift", "detail_loss", "halo"), "result.guardrails: invalid name")
            check(guard.attempt in (1, 2) and type(guard.attempt) is int, "result.guardrails: invalid attempt")
            check(_number(guard.value, nullable=True) and _number(guard.threshold), "result.guardrails: invalid value")
            check(guard.passed is None or type(guard.passed) is bool, "result.guardrails: invalid passed")
            if guard.passed is None or guard.value is None:
                check(isinstance(guard.reason, str) and bool(guard.reason), "result.guardrails: unavailable without reason")
    else:
        errors.append("result.guardrails: expected list")
    if isinstance(result.warnings, list):
        check(all(isinstance(x, EngineWarning) for x in result.warnings), "result.warnings: expected EngineWarning")
    else:
        errors.append("result.warnings: expected list")
    metrics = result.metrics
    if not isinstance(metrics, dict):
        errors.append("result.metrics: expected dict")
        return
    check(metrics.get("resolution") == 512, "result.metrics: resolution must be 512")
    required = {"resolution", "before", "after", "edge_retention", "noise_sigma", "chroma_delta_e",
                "delta_e", "clip_pct", "directions", "unavailable"}
    check(required <= metrics.keys(), "result.metrics: missing required fields")
    reasons = metrics.get("unavailable", {})
    check(isinstance(reasons, dict) and all(type(k) is str and type(v) is str and bool(v)
                                          for k, v in reasons.items()), "result.metrics: invalid unavailable reasons")
    if not isinstance(reasons, dict):
        reasons = {}
    for section, keys in (("before", ("niqe", "clipiqa", "brisque")),
                          ("after", ("niqe", "clipiqa", "brisque")),
                          ("noise_sigma", ("before", "after")), ("clip_pct", ("before", "after"))):
        mapping = metrics.get(section)
        if not isinstance(mapping, dict):
            errors.append(f"result.metrics.{section}: expected dict")
            continue
        for key in keys:
            check(key in mapping, f"result.metrics.{section}: missing {key}")
            value = mapping.get(key)
            check(_number(value, nullable=True), f"result.metrics.{section}.{key}: invalid metric")
            if value is None:
                check(key in reasons or f"{section}.{key}" in reasons or section in reasons,
                      f"result.metrics.{section}.{key}: missing unavailable reason")
    for key in ("edge_retention", "chroma_delta_e", "delta_e"):
        value = metrics.get(key)
        check(_number(value, nullable=True), f"result.metrics.{key}: invalid metric")
        if value is None:
            check(key in reasons, f"result.metrics.{key}: missing unavailable reason")
    check(metrics.get("delta_e") == metrics.get("chroma_delta_e"), "result.metrics: delta_e alias mismatch")
    directions = metrics.get("directions", {})
    check(isinstance(directions, dict) and all(v in ("higher", "lower") for v in directions.values()),
          "result.metrics: invalid direction map")
