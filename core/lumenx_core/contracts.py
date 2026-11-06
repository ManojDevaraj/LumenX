from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol, Union
import numpy as np

CONTRACT_VERSION = "1.0"

Mode = Literal["hybrid", "classical", "dl", "zerodce"]   # "zerodce" = evaluation baseline; the server API exposes only the first three
ABLATION_FLAGS = frozenset({
    "no_tuner", "no_fusion", "no_classical", "no_dl", "no_gain_denoise",
    "no_recoverability", "no_guardrails", "no_color_correction",
})
VisualRole = Literal["stage", "map", "thumb", "final"]
Severity = Literal["high", "med", "low"]

# ---------- errors ----------
class LumenXError(Exception): ...
class InvalidRequest(LumenXError):      # server maps to HTTP 422
    def __init__(self, code: str, message: str):
        super().__init__(message); self.code = code
class Cancelled(LumenXError): ...        # raised when request.cancel is set (checked at stage/tile boundaries)
class EngineUnavailable(LumenXError): ...  # server maps to HTTP 503
class EngineError(LumenXError): ...      # any other internal failure (server maps to 500 / SSE error event)

# ---------- request ----------
@dataclass(frozen=True)
class EnhanceRequest:
    image: np.ndarray                  # float32, H×W×3, RGB, [0,1], already oriented, ≤ 12 MP, C-contiguous
    original_size: tuple[int, int]     # (width, height) before any server-side downscale
    alpha: np.ndarray | None = None    # float32 H×W in [0,1], or None
    mode: Mode = "hybrid"
    ablate: frozenset[str] = frozenset()
    seed: int = 0
    cancel: threading.Event | None = None

# ---------- visuals (arrays the server encodes & publishes) ----------
@dataclass
class Visual:
    key: str                 # unique per run; becomes filename stem; [a-z0-9_]+
    image: np.ndarray        # uint8: H×W (gray), H×W×3 (RGB) or H×W×4 (RGBA)
    role: VisualRole
    max_side: int | None     # server downsizes (INTER_AREA) to this long side before encoding; None = keep
    lossless: bool           # True → PNG, False → JPEG quality 90

# ---------- streamed events ----------
@dataclass
class StageEvent:
    kind: Literal["stage", "candidate", "warning"]
    name: str                # see §5 for allowed names and order
    title: str               # ≤ 6 words, plain English
    caption: str             # ≤ 20 words, plain English, no jargon
    tech_note: str           # technical detail (may be "")
    visuals: list[Visual]
    primary_key: str | None  # key of the visual that represents this stage, or None
    data: dict[str, Any]     # JSON-safe (see §4)
    t_ms: float              # duration of this stage (0 for candidate/warning)

# ---------- final result ----------
@dataclass
class Chip:
    label: str
    value: float | None
    severity: Severity

@dataclass
class Candidate:
    id: int
    params: dict[str, float]
    scores: dict[str, float | None]   # raw subscores + "norm_*" normalized contributions
    total: float | None
    feasible: bool
    reasons: list[str]                # why infeasible / why rejected (plain English)
    thumb_key: str                    # e.g. "cand_03"
    chosen: bool

@dataclass
class GuardrailCheck:
    check: Literal["overexposure", "color_shift", "detail_loss", "halo"]
    value: float | None
    threshold: float
    passed: bool | None               # None = not applicable / unavailable
    attempt: int                      # 1 or 2
    action: str | None                # repair action taken, if any
    reason: str | None                # why unavailable / not applicable

@dataclass
class StoryStep:
    title: str
    caption: str                      # ≤ 20 words, no jargon
    visual_key: str
    highlight_key: str | None = None

@dataclass
class EngineWarning:
    code: str                         # e.g. "cidnet_unavailable", "fp16_nonfinite", "iqa_unavailable", "guardrail_unresolved"
    message: str

@dataclass
class EnhanceResult:
    headline: str
    chips: list[Chip]
    diagnosis_raw: dict[str, Any]
    hotspots: list[tuple[int, int, int, int]]          # (x, y, w, h) in processed-image pixels
    recoverability: dict[str, Any]   # recovered_pct, estimated_pct, unrecoverable_pct, sentence, estimator_confidence
    candidates: list[Candidate]      # [] when no tuner ran
    search_ms: float | None
    scoring_recipe: str              # e.g. "clipiqa+niqe+custom" or degraded recipe name
    guardrails: list[GuardrailCheck]
    decisions: list[str]             # 4–7 plain-English bullets
    story: list[StoryStep]
    metrics: dict[str, Any]          # see §6
    relight: dict[str, Any]          # base_key, L_key, gamma_default, exposure_default, gamma_range, exposure_range
    timings_ms: dict[str, float]     # core stage timings + "core_total"
    warnings: list[EngineWarning]
    provenance: dict[str, Any]       # see §7
    visuals: list[Visual]            # finalize-time visuals; MUST include key "final"

RunItem = Union[StageEvent, EnhanceResult]

# ---------- metrics ----------
class MetricSuite(Protocol):
    def paired(self, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]: ...
    #   uint8 RGB same size → keys: psnr, ssim, lpips, psnr_gt_mean
    def unpaired(self, img: np.ndarray) -> dict[str, float | None]: ...
    #   uint8 RGB → keys: niqe, brisque, clipiqa, musiq
    def custom(self, inp: np.ndarray, out: np.ndarray) -> dict[str, float | None]: ...
    #   uint8 RGB same size → keys: clip_pct_in, clip_pct_out, chroma_delta_e, edge_retention, noise_sigma_in, noise_sigma_out
    def confidence_errors(self, inp: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]: ...
    #   → mean abs error per recoverability class: err_recovered, err_estimated, err_unrecoverable, plus area pcts
    def directions(self) -> dict[str, Literal["higher", "lower"]]: ...
    def unavailable(self) -> dict[str, str]: ...       # metric name → reason (empty if all available)

# ---------- engine ----------
class Engine(Protocol):
    def warmup(self) -> None: ...                       # blocks until models are warm; idempotent
    def health(self) -> dict[str, Any]: ...             # see §8
    def public_config(self) -> dict[str, Any]: ...      # JSON-safe, no paths/secrets
    def run(self, req: EnhanceRequest) -> Iterator[RunItem]: ...
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray: ...   # live: uint8 RGB H×W×3 (long side ≤ 1280) → same shape
    def metric_suite(self) -> MetricSuite: ...

def create_engine(config_path: str | None = None, device: str | None = None) -> Engine: ...   # in lumenx_core/__init__.py
def create_fake_engine(stage_delay_ms: int = 0, seed: int = 0) -> Engine: ...               # in lumenx_core/__init__.py
