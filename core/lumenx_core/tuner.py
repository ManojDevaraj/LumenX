"""Deterministic twelve-candidate proxy tuner for hybrid still images."""

from __future__ import annotations

from dataclasses import asdict, replace

import cv2
import numpy as np

from .classical import enhance_classical
from .config import EngineConfig
from .contracts import Candidate, Visual
from .diagnose import proxy
from .metrics import custom_values
from .metrics import clipiqa_batch
from .render import uint8
from .types import DiagnosisResult, EnhanceParams


def _params(diagnosis: DiagnosisResult, cfg: EngineConfig, seed: int) -> list[EnhanceParams]:
    """Generate eight prior, two broad, conservative and identity candidates."""
    c = cfg["tuner"]
    rng = np.random.default_rng(seed)
    prior = c["prior_gamma"][diagnosis.raw["severity"]]
    values: list[EnhanceParams] = []
    for low, high, count in ((prior[0], prior[1], 8), (c["broad_gamma"][0], c["broad_gamma"][1], 2)):
        for gamma in rng.uniform(low, high, count):
            values.append(EnhanceParams(float(gamma), float(rng.uniform(*c["clahe_range"])),
                                        float(rng.uniform(*c["color_range"]))))
    values.append(EnhanceParams(c["conservative_gamma"], c["conservative_clahe"], c["conservative_color"]))
    values.append(EnhanceParams(1.0, 0.0, 0.0))
    return values


def tune(image: np.ndarray, diagnosis: DiagnosisResult, cfg: EngineConfig, seed: int) -> tuple[list[Candidate], list[Visual], EnhanceParams, str]:
    """Score exactly twelve classical proxy candidates using fixed custom scales.

    CLIP-IQA is added by the engine's optional learned metric layer when available;
    this numerical candidate pass remains deterministic and never normalizes per run.
    """
    small = proxy(image, cfg["tuner"]["proxy_side"])
    small_diagnosis = replace(diagnosis, luminance=proxy(diagnosis.luminance, cfg["tuner"]["proxy_side"]),
                              illumination=proxy(diagnosis.illumination, cfg["tuner"]["proxy_side"]),
                              noise=proxy(diagnosis.noise, cfg["tuner"]["proxy_side"]),
                              snr=proxy(diagnosis.snr, cfg["tuner"]["proxy_side"]),
                              # cv2.resize preserves two dimensions for a one-channel
                              # mask.  Adding then indexing a channel made large proxy
                              # masks one-dimensional and crashed native_noise.
                              valid=cv2.resize(diagnosis.valid.astype(np.uint8), (small.shape[1], small.shape[0]),
                                               interpolation=cv2.INTER_NEAREST).astype(bool))
    candidates: list[Candidate] = []
    visuals: list[Visual] = []
    outputs: list[np.ndarray] = []
    totals: list[float] = []
    c = cfg["tuner"]
    for index, params in enumerate(_params(diagnosis, cfg, seed)):
        out = enhance_classical(small, small_diagnosis, cfg, params=params).image
        values, _ = custom_values(uint8(small), uint8(out), cfg, small_diagnosis.valid)
        clip = float(values["clip_pct_out"] or 0.0)
        chroma = float(values["chroma_delta_e"] or 0.0)
        edge = values["edge_retention"]
        # Fixed constants make scores comparable across requests.  No per-request
        # min/max scaling is used.
        score = (-clip / c["clip_norm"] - chroma / c["chroma_norm"]
                 + (float(edge) if edge is not None else 0.0) * c["edge_weight"])
        feasible = clip <= c["max_clip_pct"]
        reasons = [] if feasible else ["This version clipped too many bright pixels."]
        key = f"cand_{index:02d}"
        candidates.append(Candidate(index, asdict(params), {"clip_pct": clip, "chroma_delta_e": chroma,
                          "edge_retention": edge, "norm_custom": score, "clipiqa": None}, score if feasible else None,
                          feasible, reasons, key, False))
        visuals.append(Visual(key, uint8(proxy(out, 128)), "thumb", 128, False))
        outputs.append(uint8(out))
        totals.append(score if feasible else -np.inf)
    clip_scores = clipiqa_batch(outputs)
    if clip_scores is not None:
        for index, (candidate, clip_score) in enumerate(zip(candidates, clip_scores)):
            scores = {**candidate.scores, "clipiqa": clip_score, "norm_clipiqa": clip_score * c["clipiqa_weight"]}
            total = candidate.total + clip_score * c["clipiqa_weight"] if candidate.total is not None else None
            candidates[index] = replace(candidate, scores=scores, total=total)
            totals[index] = total if total is not None else -np.inf
    winner = int(np.argmax(totals)) if np.isfinite(totals).any() else 10
    chosen = candidates[winner]
    candidates[winner] = replace(chosen, chosen=True)
    for index, candidate in enumerate(candidates):
        if index != winner and candidate.feasible:
            candidates[index] = replace(candidate, reasons=["A more conservative quality score ranked higher."])
    recipe = "clipiqa+custom" if clip_scores is not None else "custom_no_clipiqa"
    return candidates, visuals, EnhanceParams(**candidates[winner].params), recipe
