"""Public, versioned types shared by LumenX core and its server."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, Union

import numpy as np

CONTRACT_VERSION = "1.0"

Mode = Literal["hybrid", "classical", "dl", "zerodce"]
ABLATION_FLAGS = frozenset(
    {
        "no_tuner",
        "no_fusion",
        "no_classical",
        "no_dl",
        "no_gain_denoise",
        "no_recoverability",
        "no_guardrails",
        "no_color_correction",
    }
)
VisualRole = Literal["stage", "map", "thumb", "final"]
Severity = Literal["high", "med", "low"]


class LumenXError(Exception):
    """Base class for errors exposed by the LumenX engine."""


class InvalidRequest(LumenXError):
    """A request violates the public engine contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Cancelled(LumenXError):
    """Raised when a request's cancellation event is set."""


class EngineUnavailable(LumenXError):
    """Raised when a requested real processing path is not available."""


class EngineError(LumenXError):
    """Raised for an unexpected internal processing failure."""


@dataclass(frozen=True)
class EnhanceRequest:
    """Normalized RGB image and options supplied to an engine run."""

    image: np.ndarray
    original_size: tuple[int, int]
    alpha: np.ndarray | None = None
    mode: Mode = "hybrid"
    ablate: frozenset[str] = frozenset()
    seed: int = 0
    cancel: threading.Event | None = None


@dataclass
class Visual:
    """A rendered uint8 image or map emitted by the engine."""

    key: str
    image: np.ndarray
    role: VisualRole
    max_side: int | None
    lossless: bool


@dataclass
class StageEvent:
    """One ordered stage, candidate, or warning from a streamed run."""

    kind: Literal["stage", "candidate", "warning"]
    name: str
    title: str
    caption: str
    tech_note: str
    visuals: list[Visual]
    primary_key: str | None
    data: dict[str, Any]
    t_ms: float


@dataclass
class Chip:
    label: str
    value: float | None
    severity: Severity


@dataclass
class Candidate:
    id: int
    params: dict[str, float]
    scores: dict[str, float | None]
    total: float | None
    feasible: bool
    reasons: list[str]
    thumb_key: str
    chosen: bool


@dataclass
class GuardrailCheck:
    check: Literal["overexposure", "color_shift", "detail_loss", "halo"]
    value: float | None
    threshold: float
    passed: bool | None
    attempt: int
    action: str | None
    reason: str | None


@dataclass
class StoryStep:
    title: str
    caption: str
    visual_key: str
    highlight_key: str | None = None


@dataclass
class EngineWarning:
    code: str
    message: str


@dataclass
class EnhanceResult:
    headline: str
    chips: list[Chip]
    diagnosis_raw: dict[str, Any]
    hotspots: list[tuple[int, int, int, int]]
    recoverability: dict[str, Any]
    candidates: list[Candidate]
    search_ms: float | None
    scoring_recipe: str
    guardrails: list[GuardrailCheck]
    decisions: list[str]
    story: list[StoryStep]
    metrics: dict[str, Any]
    relight: dict[str, Any]
    timings_ms: dict[str, float]
    warnings: list[EngineWarning]
    provenance: dict[str, Any]
    visuals: list[Visual]


RunItem = Union[StageEvent, EnhanceResult]


class MetricSuite(Protocol):
    """Metric interface shared with evaluation orchestration."""

    def paired(self, pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]: ...

    def unpaired(self, img: np.ndarray) -> dict[str, float | None]: ...

    def custom(self, inp: np.ndarray, out: np.ndarray) -> dict[str, float | None]: ...

    def confidence_errors(
        self, inp: np.ndarray, pred: np.ndarray, gt: np.ndarray
    ) -> dict[str, float | None]: ...

    def directions(self) -> dict[str, Literal["higher", "lower"]]: ...

    def unavailable(self) -> dict[str, str]: ...


class Engine(Protocol):
    """Public processing interface consumed by the server."""

    def warmup(self) -> None: ...

    def health(self) -> dict[str, Any]: ...

    def public_config(self) -> dict[str, Any]: ...

    def run(self, req: EnhanceRequest) -> Iterator[RunItem]: ...

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray: ...

    def metric_suite(self) -> MetricSuite: ...
