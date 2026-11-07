"""Deterministic twelve-candidate proxy tuner for hybrid still images."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .config import EngineConfig
from .contracts import Candidate, Visual
from .diagnose import proxy
from .metrics import custom_values
from .metrics import clipiqa_batch
from .render import uint8
from .types import DiagnosisResult, EnhanceParams


def _params() -> list[dict[str, float]]:
    """The fixed twelve-way local-correction search; candidate zero is the anchor."""
    values = [{"highlight_repair": 1.0, "denoise_k": 0.0, "color_strength": 0.0, "exposure_trim": 1.0}]
    for repair, denoise_k, colour, trim in ((.5,.15,.05,.95), (.7,.25,.10,.98), (.8,.35,.15,1.02),
                                             (1,.25,.05,1.05), (.6,.45,.10,.97), (.9,.15,.20,1.03),
                                             (1,.35,.10,.99), (.7,.20,.15,1.06), (.5,.30,.05,1.01),
                                             (.8,.40,.20,.96), (1,.10,.10,1.04)):
        values.append({"highlight_repair": repair, "denoise_k": denoise_k,
                       "color_strength": colour, "exposure_trim": trim})
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
    anchor_mean = float(np.mean(small))
    for index, params in enumerate(_params()):
        # The real engine applies these to CIDNet after its event-ordered search;
        # this proxy pass scores only bounded local corrections.
        out = np.clip(small * params["exposure_trim"], 0, 1)
        values, _ = custom_values(uint8(small), uint8(out), cfg, small_diagnosis.valid)
        clip = float(values["clip_pct_out"] or 0.0)
        chroma = float(values["chroma_delta_e"] or 0.0)
        edge = values["edge_retention"]
        # Fixed constants make scores comparable across requests.  No per-request
        # min/max scaling is used.
        score = (-clip / c["clip_norm"] - chroma / c["chroma_norm"]
                 + (float(edge) if edge is not None else 0.0) * c["edge_weight"])
        darker = float(np.mean(out)) < anchor_mean * .85
        feasible = clip <= 1.0 and not darker
        reasons = [] if feasible else (["This version clipped more than 1% of pixels."] if clip > 1 else ["This version is over 15% darker than CIDNet."])
        key = f"cand_{index:02d}"
        candidates.append(Candidate(index, params, {"clip_pct": clip, "chroma_delta_e": chroma,
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
    # Stable argmax deliberately chooses the least-correction candidate on ties.
    winner = int(np.argmax(totals)) if np.isfinite(totals).any() else 0
    chosen = candidates[winner]
    candidates[winner] = replace(chosen, chosen=True)
    for index, candidate in enumerate(candidates):
        if index != winner and candidate.feasible:
            candidates[index] = replace(candidate, reasons=["A more conservative quality score ranked higher."])
    recipe = "clipiqa+custom" if clip_scores is not None else "custom_no_clipiqa"
    # Existing engine wiring expects an EnhanceParams object; candidate parameters
    # are retained in Candidate.params and applied after CIDNet in the next stage.
    return candidates, visuals, EnhanceParams(1.0, 0.0, 0.0), recipe
