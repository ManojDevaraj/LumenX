# LumenX Interface Contract — v1.0 (FROZEN)

> This file lives at the root of the single shared repository, unchanged. BOTH agents read it.
> It is the only coupling between the two parts of the monorepo:
>
> - **`core/`** (Codex): the image-processing engine, a pip-installable Python package `lumenx-core`.
> - **`server/`** (Antigravity): the service, I/O, scheduling, live mode, evaluation harness and basic frontend.
>
> Neither agent may change this contract. If a change is unavoidable, the agent stops, writes
> `CONTRACT_CHANGE_REQUEST.md` inside its own folder (`core/` or `server/`): what, why, additive or breaking. Then it waits for the human.
> Approved changes are additive only and bump `CONTRACT_VERSION` (1.0 → 1.1).

---

## 1. Ownership boundary

| Concern | Owner |
|---|---|
| Algorithms: diagnosis, recoverability, classical, CIDNet, Zero-DCE++, fusion, denoise, tuner, guardrails | core |
| Model weights, loading, GPU placement, FP16/FP32, tiling, VRAM management, warmup | core |
| Metrics (pyiqa wrappers + custom metrics) and the `MetricSuite` | core |
| Plain-English report text (headline, chips, decisions, story captions) | core |
| Rendering maps to display arrays (colormaps, RGBA overlays, heightmap, thumbnails as uint8 arrays) | core |
| `contracts.py`, `contract_check.py`, `FakeEngine` | core |
| Upload decoding, EXIF orientation, alpha split, 12 MP downscale, decompression-bomb limits | server |
| Encoding arrays to PNG/JPEG, resizing to `max_side`, atomic file publication, URLs | server |
| HTTP API, Pydantic schemas, SSE, WebSocket live, backpressure | server |
| Concurrency: single GPU lane, request queue, busy responses, cancellation tokens | server |
| Run storage, `GET /runs`, gallery caching, mock mode | server |
| Evaluation / ablation / failure mining orchestration, dataset loaders, result records, charts | server |
| Basic frontend | server |

Core never does file I/O, except for reading its own weights/config and a dev-only CLI.
Server never implements image-processing or metric algorithms; it only calls the `Engine` and the `MetricSuite`.

---

## 2. Repository, branches and packaging

**One GitHub repository (monorepo):**
```
lumenx/
├── CONTRACT.md      # this file, frozen; nobody edits
├── docs/reference/  # MASTER_PROMPT.md, FINAL_IMPLEMENTATION_PLAN.md: read-only for both agents (human-owned)
├── docs/prompts/    # the two agent prompts: read-only for both agents (human-owned)
├── README.md        # root: how to install and run both parts (owner: Antigravity)
├── .gitignore       # root (owner: Antigravity; Codex requests additions via HANDOFF.md)
├── core/            # Codex ONLY: package "lumenx-core", import "lumenx_core"
└── server/          # Antigravity ONLY: package "lumenx-server", import "lumenx_server"
```

**Folder rule (hard):** Codex creates and edits files only under `core/`. Antigravity creates and edits files only under `server/`, plus the root `README.md` and `.gitignore`. Nobody edits `CONTRACT.md`. Because the write-sets never intersect, merges never conflict.

**Branches:**
- Codex works on branch `core`. Antigravity works on branch `server`. Nobody commits directly to `main`, force-pushes, or touches the other agent's branch.
- Codex releases by merging `core` → `main` and tagging `main` with `core-v0.1.0`, `core-v0.2.0`, … (only after that milestone's gate passes).
- Antigravity picks up a new core release with `git merge core-vX.Y.Z` (or `git merge main`) into `server`, then merges `server` → `main` at its own milestones, tagging `server-sN`.
- Merging the latest `main` into your branch before merging back to `main` is mandatory.

**Packaging:**
- Python 3.10/3.11 for both parts.
- `core/pyproject.toml`: distribution `lumenx-core`, package `lumenx_core`.
- The server installs core from the same checkout: `pip install -e ../core` (documented in the root README and in `server/pyproject.toml`). The core version actually in use is whatever release was last merged into `server`. The server reports `lumenx_core.__version__` in `/health`.
- Core releases: `core-v0.1.0` = contract + FakeEngine (no real algorithms); `core-v0.2.0` = real classical; `core-v0.3.0` = hybrid; `core-v0.4.0` = feature-complete; later patch releases only.
- `lumenx_core.__version__` and `lumenx_core.contracts.CONTRACT_VERSION = "1.0"`.
- Each part declares its own dependencies in its own `pyproject.toml`. One shared venv at the repo root (`python -m venv .venv`) is used for running both: install core first (`pip install -e core`), then the server (`pip install -e server`).

---

## 3. `lumenx_core/contracts.py` (core implements exactly this)

```python
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
```

---

## 4. Behavioural rules

1. `Engine.run` is a generator. It yields `StageEvent`s in order and then **exactly one `EnhanceResult`, as the final item**. It yields nothing after that.
2. It raises `InvalidRequest` **before yielding anything** for invalid input: wrong dtype/shape/range, unknown ablation flag, contradictory flags (`no_dl`+`no_fusion`, `no_dl`+`no_classical`), or ablation flags on a non-`hybrid` mode.
3. **Engine is not thread-safe.** The server must serialise every call to `run`, `enhance_frame`, `warmup` and `MetricSuite` methods through a single lane. Core performs no locking or scheduling.
4. The generator must not be consumed concurrently. If the consumer stops iterating early, core releases GPU memory on `GeneratorExit` (use `try/finally`).
5. Cancellation: core checks `req.cancel.is_set()` between stages and between CIDNet tiles, and raises `Cancelled`.
6. JSON-safe means: only `dict`/`list`/`str`/`int`/`float`/`bool`/`None`; no NumPy scalars; **no NaN or ±Inf** (use `None` and give a reason elsewhere).
7. Every `Visual.key` is unique within a run. Every key referenced by `primary_key`, `StoryStep.visual_key`/`highlight_key`, `Candidate.thumb_key` or `relight.*_key` must appear in some emitted `Visual` (in an earlier event or in `EnhanceResult.visuals`).
8. The server publishes the input itself under key `input`. Core never emits a visual with key `input`.
9. Determinism: same image + mode + ablate + seed + config + checkpoints on the same machine → identical `final` array.
10. All hotspot coordinates and maps refer to the processed image (`req.image`) geometry.
11. `lumenx_core.contract_check.validate_run(items: list[RunItem], req: EnhanceRequest) -> list[str]` returns a list of violations (empty = valid). Both parts use it in their tests.

---

## 5. Stage names, order and required visuals

The order is fixed. Stages that a mode skips are simply absent.

| # | `name` (kind=stage) | hybrid | classical | dl | zerodce | Required visual keys (role, max_side, lossless) |
|---|---|---|---|---|---|---|
| 1 | `diagnose` | ✓ | ✓ | ✓ | ✓ | `illumination` (stage, 768, T) — primary; `L_gray` (map, 512, T, gray); `noise_map` (map, 512, T); `snr` (map, 512, T); `heightmap` (map, None, T, 128×128 gray) |
| 2 | `recoverability` | ✓ | ✓ | ✓ | ✓ | `recoverability_overlay` (map, 512, T, RGBA) — primary |
| 3 | `tuner` | ✓ (unless `no_tuner`) | — | — | — | primary `cand_XX` of the winner; emitted after all candidate events |
| — | `cand_00`…`cand_11` (kind=candidate) | ✓ | — | — | — | `cand_XX` (thumb, 128, F); `data` = the `Candidate` as a dict |
| 4 | `classical` | ✓ (unless `no_classical`) | ✓ | — | — | `classical` (stage, 768, F) — primary; `reflectance` (stage, 768, F) |
| 5 | `deep` | ✓ (unless `no_dl`) | — | ✓ | ✓ | `deep` (stage, 768, F) — primary |
| 6 | `fusion` | ✓ (unless `no_fusion`) | — | — | — | `fused` (stage, 768, F) — primary; `fusion_map` (map, 512, T) |
| 7 | `denoise` | ✓ | ✓ | — | — | `denoised` (stage, 768, F) — primary; `gain` (map, 512, T); `denoise_strength` (map, 512, T) |
| 8 | `guardrails` | ✓ | ✓ | ✓ (audit only) | ✓ (audit only) | none required (primary may be None) |
| — | (EnhanceResult.visuals) | ✓ | ✓ | ✓ | ✓ | `final` (final, None, T, RGB or RGBA if input had alpha); `final_preview` (stage, 1280, F) |

- Exactly 12 candidate events when the tuner runs, ids 0–11, exactly one with `chosen=True`.
- `kind="warning"` events may appear anywhere after `diagnose`; `name` = warning code, `visuals=[]`.
- `relight` = `{"base_key": "final_preview", "L_key": "L_gray", "gamma_default": 1.0, "exposure_default": 1.0, "gamma_range": [0.4, 2.5], "exposure_range": [0.25, 4.0]}`.

---

## 6. `EnhanceResult.metrics` shape

Request-time metrics are computed on a **512-px long-side proxy** of input and final (stated in `metrics["resolution"]`). Full-resolution metrics are reserved for evaluation.

```json
{
  "resolution": 512,
  "before": {"niqe": 6.8, "clipiqa": 0.22, "brisque": 48.1},
  "after":  {"niqe": 3.6, "clipiqa": 0.67, "brisque": 21.4},
  "edge_retention": 0.93,
  "noise_sigma": {"before": 0.041, "after": 0.012},
  "chroma_delta_e": 3.1,
  "delta_e": 3.1,
  "clip_pct": {"before": 0.4, "after": 0.6},
  "directions": {"niqe": "lower", "clipiqa": "higher", "brisque": "lower", "edge_retention": "higher", "chroma_delta_e": "lower"},
  "unavailable": {"musiq": "not computed at request time"}
}
```

Any value may be `null`; its reason goes in `unavailable`.

---

## 7. `EnhanceResult.provenance` shape

```json
{
  "requested_mode": "hybrid", "actual_method": "hybrid|classical|cidnet|zerodce|classical_fallback",
  "fallback_chain": ["cidnet:missing_weights", "zerodce"],
  "checkpoint": "cidnet_lolv2_real|cidnet_sid|zerodce_pp|null",
  "precision": "fp16|fp32", "device": "cuda:0|cpu", "placement_policy": "resident|swap",
  "ablate": [], "seed": 0,
  "config_hash": "sha256-12", "input_hash": "sha256-12", "checkpoint_hash": "sha256-12|null",
  "core_version": "0.4.0", "contract_version": "1.0",
  "repair_attempted": false, "guardrails_unresolved": []
}
```

---

## 8. `Engine.health()` shape

```json
{
  "fake": false, "ready": true, "hybrid_ready": true, "warm": true,
  "device": "cuda:0", "placement_policy": "resident",
  "models": {
    "cidnet": {"available": true, "checkpoints": ["cidnet_lolv2_real", "cidnet_sid"]},
    "zerodce": {"available": true},
    "iqa": {"clipiqa": true, "niqe": true, "brisque": true, "musiq": true, "lpips": true}
  },
  "vram_mb": {"total": 4096, "allocated": 900, "reserved": 1200, "peak": 2100},
  "degraded": [],
  "core_version": "0.4.0", "contract_version": "1.0"
}
```

---

## 9. Latency split for the 1.5 s target (warm, uncached, 1080p, RTX 2050, hybrid)

| Side | Budget | Measured as |
|---|---|---|
| core | ≤ 1 100 ms | `timings_ms["core_total"]`: first item requested → `EnhanceResult` yielded, including all visual array rendering |
| server | ≤ 400 ms | decode + encode/publish of all visuals (pipelined with core, overlapping where possible) + `result.json` + response serialisation |
| end-to-end | ≤ 1 500 ms p95 | server wall clock from request body received → `done` event / JSON response sent (queue time reported separately) |

Larger images, CPU runs, cached runs and queue time are reported, not held to this target.

---

## 10. FakeEngine (in core, shipped at `v0.1.0`)

`create_fake_engine()` returns an `Engine` that:
- runs on CPU without weights, with deterministic output for a given seed;
- produces every stage, candidate, visual key and result field in §5–§8 with correct shapes, roles and dtypes, using cheap real operations (e.g. gamma lift, simple colormaps), so its output looks plausible;
- honours `stage_delay_ms` (a sleep after each stage) so streaming UIs can be demonstrated;
- reports `health()["fake"] = true`;
- supports `enhance_frame` (simple gamma) and a `MetricSuite` returning plausible values with `unavailable` filled in honestly;
- passes `validate_run`.
