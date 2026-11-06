"""Runtime validation helpers for the frozen LumenX stream contract."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .contracts import ABLATION_FLAGS, EnhanceRequest, EnhanceResult, RunItem, StageEvent, Visual

# `L_gray` is the one stable mixed-case key explicitly required by CONTRACT.md.
_KEY_RE = re.compile(r"^(?:[a-z0-9_]+|L_gray)$")
_MAX_IMAGE_PIXELS = 12_000_000


def _json_safe(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, np.generic):
        errors.append(f"{path} contains a NumPy scalar")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_safe(item, f"{path}[{index}]", errors)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(f"{path} has a non-string key")
            _json_safe(item, f"{path}.{key}", errors)
        return
    errors.append(f"{path} contains non-JSON type {type(value).__name__}")


def _visual_error(visual: Visual, path: str) -> str | None:
    if not isinstance(visual.key, str) or not _KEY_RE.fullmatch(visual.key):
        return f"{path}.key must match [a-z0-9_]+"
    if visual.key == "input":
        return f"{path} must not publish the input visual"
    if not isinstance(visual.image, np.ndarray) or visual.image.dtype != np.uint8:
        return f"{path}.image must be a uint8 NumPy array"
    if visual.image.ndim not in (2, 3):
        return f"{path}.image must be HxW, HxWx3, or HxWx4"
    if visual.image.ndim == 3 and visual.image.shape[2] not in (3, 4):
        return f"{path}.image must have three or four channels"
    if visual.image.shape[0] < 1 or visual.image.shape[1] < 1:
        return f"{path}.image must not be empty"
    if visual.max_side is not None and visual.max_side < 1:
        return f"{path}.max_side must be positive or None"
    if visual.role not in ("stage", "map", "thumb", "final"):
        return f"{path}.role is invalid"
    return None


def _check_visual_spec(
    visual: Visual,
    *,
    role: str,
    max_side: int | None,
    lossless: bool,
    errors: list[str],
    path: str,
    channels: int | None = None,
    shape: tuple[int, int] | None = None,
) -> None:
    if (visual.role, visual.max_side, visual.lossless) != (role, max_side, lossless):
        errors.append(
            f"{path} must have role={role}, max_side={max_side}, lossless={lossless}"
        )
    if channels is not None:
        actual_channels = 1 if visual.image.ndim == 2 else visual.image.shape[2]
        if actual_channels != channels:
            errors.append(f"{path} must have {channels} channel(s)")
    if shape is not None and visual.image.shape[:2] != shape:
        errors.append(f"{path} must have spatial shape {shape}")


def _expected_stages(req: EnhanceRequest) -> list[str]:
    stages = ["diagnose", "recoverability"]
    if req.mode == "hybrid" and "no_tuner" not in req.ablate:
        stages.append("tuner")
    if req.mode == "classical" or (
        req.mode == "hybrid" and "no_classical" not in req.ablate
    ):
        stages.append("classical")
    if req.mode in ("dl", "zerodce") or (
        req.mode == "hybrid" and "no_dl" not in req.ablate
    ):
        stages.append("deep")
    if req.mode == "hybrid" and "no_fusion" not in req.ablate:
        stages.append("fusion")
    if req.mode in ("classical", "hybrid"):
        stages.append("denoise")
    stages.append("guardrails")
    return stages


def _validate_request(req: EnhanceRequest, errors: list[str]) -> None:
    image = req.image
    if not isinstance(image, np.ndarray):
        errors.append("request.image must be a NumPy array")
    else:
        if image.dtype != np.float32:
            errors.append("request.image must have dtype float32")
        if image.ndim != 3 or (image.ndim == 3 and image.shape[2] != 3):
            errors.append("request.image must have shape HxWx3")
        elif image.shape[0] < 1 or image.shape[1] < 1:
            errors.append("request.image must not be empty")
        elif image.shape[0] * image.shape[1] > _MAX_IMAGE_PIXELS:
            errors.append("request.image must be at most 12 megapixels")
        if not image.flags.c_contiguous:
            errors.append("request.image must be C-contiguous")
        if image.dtype == np.float32:
            if not np.isfinite(image).all():
                errors.append("request.image must contain only finite values")
            elif image.size and (float(image.min()) < 0.0 or float(image.max()) > 1.0):
                errors.append("request.image values must be in [0, 1]")

    if req.mode not in ("hybrid", "classical", "dl", "zerodce"):
        errors.append(f"unknown mode {req.mode!r}")
    if not isinstance(req.original_size, tuple) or len(req.original_size) != 2:
        errors.append("request.original_size must be a (width, height) tuple")
    elif any(not isinstance(value, int) or value <= 0 for value in req.original_size):
        errors.append("request.original_size values must be positive integers")
    if not isinstance(req.ablate, frozenset):
        errors.append("request.ablate must be a frozenset")
    else:
        unknown = req.ablate - ABLATION_FLAGS
        if unknown:
            errors.append(f"unknown ablation flags: {sorted(unknown, key=str)}")
        if req.ablate and req.mode != "hybrid":
            errors.append("ablation flags are only valid in hybrid mode")
        if {"no_dl", "no_classical"}.issubset(req.ablate):
            errors.append("at least one enhancement path must remain enabled")

    if req.alpha is not None:
        if not isinstance(req.alpha, np.ndarray) or req.alpha.dtype != np.float32:
            errors.append("request.alpha must be a float32 NumPy array")
        elif isinstance(image, np.ndarray) and image.ndim == 3 and req.alpha.shape != image.shape[:2]:
            errors.append("request.alpha must match the image height and width")
        elif not np.isfinite(req.alpha).all():
            errors.append("request.alpha must contain only finite values")
        elif req.alpha.size and (float(req.alpha.min()) < 0.0 or float(req.alpha.max()) > 1.0):
            errors.append("request.alpha values must be in [0, 1]")


def validate_run(items: list[RunItem], req: EnhanceRequest) -> list[str]:
    """Return every detectable violation of the stream contract."""

    errors: list[str] = []
    _validate_request(req, errors)
    if errors:
        return errors
    if not items:
        return ["run must contain stage events followed by one EnhanceResult"]
    if not isinstance(items[-1], EnhanceResult):
        errors.append("the final run item must be an EnhanceResult")
    if sum(isinstance(item, EnhanceResult) for item in items) != 1:
        errors.append("run must contain exactly one EnhanceResult")
    if any(isinstance(item, EnhanceResult) for item in items[:-1]):
        errors.append("EnhanceResult must be the final item")
    result = items[-1] if isinstance(items[-1], EnhanceResult) else None

    events = [item for item in items if isinstance(item, StageEvent)]
    if len(events) + (1 if result is not None else 0) != len(items):
        errors.append("run contains an item that is neither StageEvent nor EnhanceResult")

    stage_names = [event.name for event in events if event.kind == "stage"]
    expected = _expected_stages(req)
    if stage_names != expected:
        errors.append(f"stage order must be {expected}, got {stage_names}")
    if any(event.kind not in ("stage", "candidate", "warning") for event in events):
        errors.append("event kind is invalid")
    diagnose_position = next(
        (index for index, event in enumerate(events) if event.kind == "stage" and event.name == "diagnose"),
        None,
    )
    if diagnose_position is not None and any(
        event.kind == "warning" and index < diagnose_position
        for index, event in enumerate(events)
    ):
        errors.append("warning events may only appear after diagnosis")
    tuner_position = next(
        (index for index, event in enumerate(events) if event.kind == "stage" and event.name == "tuner"),
        None,
    )
    if tuner_position is not None and any(
        event.kind == "candidate" and index > tuner_position
        for index, event in enumerate(events)
    ):
        errors.append("all candidate events must precede the tuner stage")

    all_visuals: dict[str, Visual] = {}
    seen_visuals: set[str] = set()
    for event_index, event in enumerate(events):
        path = f"items[{event_index}]"
        if isinstance(event.title, str) and len(event.title.split()) > 6:
            errors.append(f"{path}.title must contain at most six words")
        if not isinstance(event.title, str):
            errors.append(f"{path}.title must be a string")
        if isinstance(event.caption, str) and len(event.caption.split()) > 20:
            errors.append(f"{path}.caption must contain at most twenty words")
        if not isinstance(event.caption, str):
            errors.append(f"{path}.caption must be a string")
        if not isinstance(event.t_ms, (int, float)) or isinstance(event.t_ms, bool) or not math.isfinite(event.t_ms) or event.t_ms < 0:
            errors.append(f"{path}.t_ms must be finite and non-negative")
        _json_safe(event.data, f"{path}.data", errors)
        if event.kind == "warning" and (event.visuals or event.t_ms != 0):
            errors.append(f"{path} warning events must have no visuals and zero duration")
        if event.kind == "candidate" and event.visuals:
            if len(event.visuals) != 1:
                errors.append(f"{path} candidate event must contain exactly one visual")
        event_visual_keys = {visual.key for visual in event.visuals}
        if event.primary_key is not None and event.primary_key not in seen_visuals | event_visual_keys:
            errors.append(f"{path}.primary_key must reference a previously emitted visual")
        for visual_index, visual in enumerate(event.visuals):
            visual_path = f"{path}.visuals[{visual_index}]"
            issue = _visual_error(visual, visual_path)
            if issue:
                errors.append(issue)
            if visual.key in all_visuals:
                errors.append(f"visual key {visual.key!r} is repeated")
            else:
                all_visuals[visual.key] = visual
            seen_visuals.add(visual.key)

    # Core visual keys and their encoding hints are stable for the server.
    required_specs: dict[str, dict[str, tuple[str, int | None, bool, int | None]]] = {
        "diagnose": {
            "illumination": ("stage", 768, True, 3),
            "L_gray": ("map", 512, True, 1),
            "noise_map": ("map", 512, True, None),
            "snr": ("map", 512, True, None),
            "heightmap": ("map", None, True, 1),
        },
        "recoverability": {"recoverability_overlay": ("map", 512, True, 4)},
        "classical": {
            "classical": ("stage", 768, False, 3),
            "reflectance": ("stage", 768, False, 3),
        },
        "deep": {"deep": ("stage", 768, False, 3)},
        "fusion": {
            "fused": ("stage", 768, False, 3),
            "fusion_map": ("map", 512, True, 3),
        },
        "denoise": {
            "denoised": ("stage", 768, False, 3),
            "gain": ("map", 512, True, None),
            "denoise_strength": ("map", 512, True, None),
        },
    }
    for event in events:
        if event.kind != "stage":
            continue
        expected_visuals = required_specs.get(event.name, {})
        by_key = {visual.key: visual for visual in event.visuals}
        for key, (role, max_side, lossless, channels) in expected_visuals.items():
            visual = by_key.get(key)
            if visual is None:
                errors.append(f"{event.name} stage is missing visual {key!r}")
                continue
            _check_visual_spec(
                visual,
                role=role,
                max_side=max_side,
                lossless=lossless,
                errors=errors,
                path=f"{event.name}.{key}",
                channels=channels,
            )
        if event.name == "diagnose" and "heightmap" in by_key:
            _check_visual_spec(
                by_key["heightmap"],
                role="map",
                max_side=None,
                lossless=True,
                errors=errors,
                path="diagnose.heightmap",
                shape=(128, 128),
            )

    candidate_events = [event for event in events if event.kind == "candidate"]
    tuner_events = [event for event in events if event.kind == "stage" and event.name == "tuner"]
    tuner_expected = req.mode == "hybrid" and "no_tuner" not in req.ablate
    if tuner_expected:
        if len(candidate_events) != 12:
            errors.append("a tuner run must emit exactly twelve candidate events")
        for candidate_index, event in enumerate(candidate_events):
            expected_id = candidate_index
            if event.name != f"cand_{candidate_index:02d}":
                errors.append("candidate event names must be cand_00 through cand_11 in order")
            if not event.visuals:
                errors.append(f"candidate {event.name} is missing its thumbnail")
                continue
            visual = event.visuals[0]
            _check_visual_spec(
                visual,
                role="thumb",
                max_side=128,
                lossless=False,
                errors=errors,
                path=f"candidate.{event.name}",
            )
            if not re.fullmatch(r"cand_\d{2}", visual.key):
                errors.append("candidate thumbnail keys must use cand_XX")
            if visual.key != event.name:
                errors.append("candidate event name must match its thumbnail key")
            candidate_id = event.data.get("id") if isinstance(event.data, dict) else None
            if candidate_id != expected_id:
                errors.append(f"candidate {event.name} data id must be {expected_id}")
        if len(tuner_events) != 1:
            errors.append("a tuner run must emit exactly one tuner stage")
        chosen = [
            event
            for event in candidate_events
            if isinstance(event.data, dict) and event.data.get("chosen") is True
        ]
        if len(chosen) != 1:
            errors.append("exactly one candidate must be marked chosen")
        elif tuner_events and tuner_events[0].primary_key != chosen[0].data.get("thumb_key"):
            errors.append("tuner primary_key must identify the chosen candidate thumbnail")
    elif candidate_events or tuner_events:
        errors.append("candidate and tuner events are only valid when hybrid tuning runs")

    if result is not None:
        _json_safe(result.diagnosis_raw, "result.diagnosis_raw", errors)
        _json_safe(result.recoverability, "result.recoverability", errors)
        _json_safe(result.metrics, "result.metrics", errors)
        _json_safe(result.relight, "result.relight", errors)
        _json_safe(result.timings_ms, "result.timings_ms", errors)
        _json_safe(result.provenance, "result.provenance", errors)
        if result.search_ms is not None:
            _json_safe(result.search_ms, "result.search_ms", errors)
        for candidate_index, candidate in enumerate(result.candidates):
            _json_safe(
                {
                    "params": candidate.params,
                    "scores": candidate.scores,
                    "total": candidate.total,
                },
                f"result.candidates[{candidate_index}]",
                errors,
            )
        for chip_index, chip in enumerate(result.chips):
            _json_safe(chip.value, f"result.chips[{chip_index}].value", errors)
        if not 4 <= len(result.decisions) <= 7:
            errors.append("result.decisions must contain four to seven entries")
        for story_index, step in enumerate(result.story):
            if len(step.caption.split()) > 20:
                errors.append(f"result.story[{story_index}].caption must contain at most twenty words")
        for visual_index, visual in enumerate(result.visuals):
            visual_path = f"result.visuals[{visual_index}]"
            issue = _visual_error(visual, visual_path)
            if issue:
                errors.append(issue)
            if visual.key in all_visuals:
                errors.append(f"visual key {visual.key!r} is repeated")
            else:
                all_visuals[visual.key] = visual
        final = next((v for v in result.visuals if v.key == "final"), None)
        preview = next((v for v in result.visuals if v.key == "final_preview"), None)
        if final is None or preview is None:
            errors.append("result visuals must include final and final_preview")
        else:
            _check_visual_spec(
                final,
                role="final",
                max_side=None,
                lossless=True,
                errors=errors,
                path="result.final",
                channels=4 if req.alpha is not None else 3,
                shape=req.image.shape[:2],
            )
            _check_visual_spec(
                preview,
                role="stage",
                max_side=1280,
                lossless=False,
                errors=errors,
                path="result.final_preview",
                channels=3,
            )
        available_keys = set(all_visuals)
        for story_index, step in enumerate(result.story):
            if step.visual_key not in available_keys:
                errors.append(f"story step {story_index} references missing visual {step.visual_key!r}")
            if step.highlight_key is not None and step.highlight_key not in available_keys:
                errors.append(
                    f"story step {story_index} references missing highlight {step.highlight_key!r}"
                )
        for candidate in result.candidates:
            if candidate.thumb_key not in available_keys:
                errors.append(f"candidate {candidate.id} references missing thumbnail")
        if isinstance(result.relight, dict):
            for name in ("base_key", "L_key"):
                key = result.relight.get(name)
                if key not in available_keys:
                    errors.append(f"result.relight.{name} references missing visual {key!r}")
        if not tuner_expected and result.candidates:
            errors.append("result.candidates must be empty when tuning is disabled")
        if tuner_expected and len(result.candidates) != 12:
            errors.append("result.candidates must contain all twelve tuner candidates")
        if tuner_expected:
            result_ids = [candidate.id for candidate in result.candidates]
            if result_ids != list(range(12)):
                errors.append("result candidates must contain ids 0 through 11 in order")
            if sum(bool(candidate.chosen) for candidate in result.candidates) != 1:
                errors.append("exactly one result candidate must be marked chosen")

    # All primary references must resolve to visuals emitted by that point; final
    # cross-references are checked above after the complete key set is known.
    return errors
