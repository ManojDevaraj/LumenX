"""Deterministic, image-adaptive search over twelve enhancement settings."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .classical import restore_classical
from .contracts import Candidate
from .diagnose import luminance
from .denoise import denoise
from .fusion import fuse_paths
from .metrics import MetricSuiteImpl
from .render import fit_long_side, rgb_u8
from .types import DiagnosisResult, RecoveryResult


def _candidate_params(
    diagnosis: DiagnosisResult,
    cfg: dict[str, dict[str, float | int]],
    seed: int,
) -> list[dict[str, float]]:
    values = cfg["tuner"]
    classical = cfg["classical"]
    rng = np.random.default_rng(seed)
    count = int(values["candidate_count"])
    sampled_count = count - 2
    names = ("gamma", "clahe_clip", "denoise_k", "fusion_alpha", "color_strength")
    ranges = {
        "gamma": (float(classical["gamma_min"]), float(classical["gamma_max"])),
        "clahe_clip": (float(values["clahe_min"]), float(values["clahe_max"])),
        "denoise_k": (float(values["denoise_k_min"]), float(values["denoise_k_max"])),
        "fusion_alpha": (float(values["alpha_min"]), float(values["alpha_max"])),
        "color_strength": (float(values["color_min"]), float(values["color_max"])),
    }
    center_gamma = float(classical["gamma_max"]) - np.clip(
        1.0 - diagnosis.mean_luminance / float(classical["dark_adaptation_scale"]), 0.0, 1.0
    ) * (float(classical["gamma_max"]) - float(classical["gamma_min"]))
    focused = dict(ranges)
    focused["gamma"] = (
        max(ranges["gamma"][0], center_gamma - 0.08),
        min(ranges["gamma"][1], center_gamma + 0.08),
    )
    lhs: dict[str, np.ndarray] = {}
    for name in names:
        permutation = rng.permutation(sampled_count)
        lhs[name] = (permutation + rng.random(sampled_count)) / sampled_count
    candidates: list[dict[str, float]] = []
    for index in range(sampled_count):
        broader = index >= sampled_count - 2
        limits = ranges if broader else focused
        candidate: dict[str, float] = {}
        for name in names:
            low, high = limits[name]
            candidate[name] = float(low + lhs[name][index] * (high - low))
        candidates.append(candidate)
    candidates.append(
        {
            "gamma": min(float(classical["gamma_max"]), 0.84),
            "clahe_clip": float(values["clahe_min"]),
            "denoise_k": float(values["denoise_k_max"]),
            "fusion_alpha": float(values["alpha_min"]),
            "color_strength": 0.0,
        }
    )
    candidates.append(
        {
            "gamma": 1.0,
            "clahe_clip": 0.01,
            "denoise_k": float(values["denoise_k_min"]),
            "fusion_alpha": 0.0,
            "color_strength": 0.0,
        }
    )
    return candidates


def default_params(
    diagnosis: DiagnosisResult, cfg: dict[str, dict[str, float | int]]
) -> dict[str, float]:
    """Return the deterministic diagnosis-adaptive settings used without search."""

    classical = cfg["classical"]
    darkness = float(
        np.clip(1.0 - diagnosis.mean_luminance / float(classical["dark_adaptation_scale"]), 0.0, 1.0)
    )
    gamma = float(classical["gamma_max"]) - darkness * (
        float(classical["gamma_max"]) - float(classical["gamma_min"])
    )
    return {
        "gamma": gamma,
        "clahe_clip": float(classical["clahe_clip_limit"]),
        "denoise_k": 1.0,
        "fusion_alpha": float(cfg["fusion"]["alpha_default"]),
        "color_strength": float(classical["color_correction_max"]) * diagnosis.color_cast_strength,
    }


def search_candidates(
    image: np.ndarray,
    diagnosis: DiagnosisResult,
    recovery: RecoveryResult,
    deep: np.ndarray,
    cfg: dict[str, dict[str, float | int]],
    *,
    seed: int,
    use_recoverability: bool = True,
    use_classical: bool = True,
    use_deep: bool = True,
    use_fusion: bool = True,
    uniform_denoise: bool = False,
    correct_color: bool = True,
) -> tuple[list[Candidate], list[np.ndarray], dict[str, float], float]:
    """Evaluate twelve deterministic candidates on a 256-pixel proxy."""

    started = time.perf_counter()
    proxy = fit_long_side(image, int(cfg["tuner"]["proxy_side"]))
    deep_proxy = fit_long_side(deep, int(cfg["tuner"]["proxy_side"]))
    recovery_proxy = RecoveryResult(
        soft_score=fit_long_side(recovery.soft_score, int(cfg["tuner"]["proxy_side"])),
        classes=fit_long_side(recovery.classes, int(cfg["tuner"]["proxy_side"])),
        overlay=fit_long_side(recovery.overlay, int(cfg["tuner"]["proxy_side"])),
        gain_limit=fit_long_side(recovery.gain_limit, int(cfg["tuner"]["proxy_side"])),
        recovered_pct=recovery.recovered_pct,
        estimated_pct=recovery.estimated_pct,
        unrecoverable_pct=recovery.unrecoverable_pct,
        estimator_confidence=recovery.estimator_confidence,
        sentence=recovery.sentence,
    )
    # Recompute diagnosis once at the proxy so every candidate uses the same geometry.
    from .diagnose import diagnose

    diag_proxy = diagnose(proxy, cfg)
    suite = MetricSuiteImpl(cfg)
    generated: list[tuple[Candidate, np.ndarray]] = []
    params_list = _candidate_params(diagnosis, cfg, seed)
    for candidate_id, params in enumerate(params_list):
        classical_result = restore_classical(
            proxy, diag_proxy, cfg, params, correct_color=correct_color
        )
        if use_fusion:
            fused, _ = fuse_paths(
                proxy,
                classical_result.image,
                deep_proxy,
                diag_proxy,
                recovery_proxy,
                cfg,
                params=params,
                use_recoverability=use_recoverability,
                use_classical=use_classical,
                use_deep=use_deep,
            )
        elif use_classical:
            fused = classical_result.image
        elif use_deep:
            fused = deep_proxy
        else:
            fused = proxy
        fused = denoise(
            proxy,
            fused,
            diag_proxy,
            recovery_proxy,
            cfg,
            use_recoverability=use_recoverability,
            uniform=uniform_denoise,
            strength_multiplier=params["denoise_k"],
        ).image
        source_u8, output_u8 = rgb_u8(proxy), rgb_u8(fused)
        raw = suite.custom(source_u8, output_u8)
        target = float(cfg["tuner"]["exposure_target"])
        mean_y = float(np.mean(luminance(fused), dtype=np.float64))
        exposure_score = float(np.clip(1.0 - abs(mean_y - target) / max(target, 1e-3), 0.0, 1.0))
        clip_after = float(raw["clip_pct_out"] or 0.0)
        clip_score = float(np.clip(1.0 - clip_after / 10.0, 0.0, 1.0))
        delta = raw["chroma_delta_e"]
        color_score = float(np.clip(1.0 - (float(delta) if delta is not None else 0.0) / 20.0, 0.0, 1.0))
        edge = raw["edge_retention"]
        edge_score = float(np.clip((float(edge) + 1.0) / 2.0, 0.0, 1.0)) if edge is not None else 0.5
        noise_in = float(raw["noise_sigma_in"] or 0.0)
        noise_out = float(raw["noise_sigma_out"] or 0.0)
        noise_score = float(np.clip(1.0 - noise_out / max(noise_in, 1e-4), 0.0, 1.0))
        normalized = {
            "norm_exposure": exposure_score,
            "norm_clipping": clip_score,
            "norm_color": color_score,
            "norm_edges": edge_score,
            "norm_noise": noise_score,
        }
        total = (
            0.30 * exposure_score
            + 0.25 * clip_score
            + 0.20 * color_score
            + 0.15 * edge_score
            + 0.10 * noise_score
        )
        reasons: list[str] = []
        feasible = True
        if clip_after > float(cfg["tuner"]["clipping_limit_pct"]):
            feasible = False
            reasons.append("Highlight clipping exceeded the configured limit.")
        if delta is not None and float(delta) > float(cfg["tuner"]["color_shift_limit"]):
            feasible = False
            reasons.append("Measured chroma shift exceeded the configured limit.")
        if edge is not None and float(edge) < float(cfg["tuner"]["edge_retention_min"]):
            feasible = False
            reasons.append("Reliable-edge retention was below the configured limit.")
        if not reasons:
            reasons.append("Candidate met the measured clipping, colour, and edge limits.")
        scores: dict[str, float | None] = {
            "exposure_error": abs(mean_y - target),
            "clip_pct": clip_after,
            "chroma_delta_e": None if delta is None else float(delta),
            "edge_retention": None if edge is None else float(edge),
            "noise_sigma_in": noise_in,
            "noise_sigma_out": noise_out,
            **normalized,
        }
        candidate = Candidate(
            id=candidate_id,
            params={key: float(value) for key, value in params.items()},
            scores=scores,
            total=float(total - (0.25 if not feasible else 0.0)),
            feasible=feasible,
            reasons=reasons,
            thumb_key=f"cand_{candidate_id:02d}",
            chosen=False,
        )
        generated.append((candidate, fit_long_side(output_u8, 128)))

    winner_index = max(
        range(len(generated)),
        key=lambda index: (
            generated[index][0].feasible,
            generated[index][0].total if generated[index][0].total is not None else -1.0,
        ),
    )
    candidates: list[Candidate] = []
    thumbs: list[np.ndarray] = []
    for index, (candidate, thumbnail) in enumerate(generated):
        candidate.chosen = index == winner_index
        candidates.append(candidate)
        thumbs.append(thumbnail)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return candidates, thumbs, dict(candidates[winner_index].params), elapsed_ms


def candidate_payload(candidate: Candidate) -> dict[str, Any]:
    """Return a JSON-safe candidate payload for the streaming event."""

    return asdict(candidate)

