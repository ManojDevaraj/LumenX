# PROMPT FOR CODEX — Build `core/` (the LumenX processing engine) in the shared LumenX monorepo

> Repository: https://github.com/ManojDevaraj/LumenX.git. `CONTRACT.md` is at the repo root (frozen).
> Reference documents (read fully): `MASTER_PROMPT.md`, `FINAL_IMPLEMENTATION_PLAN.md` (committed in the repo under `docs/reference/`; read them from there).
> Precedence when documents disagree: **CONTRACT.md > this prompt > FINAL_IMPLEMENTATION_PLAN.md > MASTER_PROMPT.md**.
> Sections 7 (file ownership) and 8.2–8.3 (public API/artifacts) of the FINAL plan are **obsolete**: the API, files and artifacts now belong to `server/`.

---

## 0. Your role

You are the sole author of the **`core/`** folder of the shared LumenX repository: a pip-installable Python package (`lumenx-core`) containing the entire LumenX image-processing engine.

Another agent (Antigravity) is building **`server/`** in the **same repository, at the same time**, on its own branch. It handles HTTP, file I/O, scheduling, live streaming transport, evaluation orchestration and the frontend. You never edit its files. The **only** link between you is `CONTRACT.md`. Implement it exactly. The human merges and reviews everything together at the end.

Your success criterion: the server can `pip install -e ../core` from a released `main`, call `create_engine()`, and get correct, fast, honest results that pass `validate_run`, without needing to know anything about your internals.

---

## 1. Scope

**You own:**
- `lumenx_core/contracts.py` and `contract_check.py` (exactly as in CONTRACT.md §3–§4).
- `FakeEngine` (CONTRACT.md §10).
- All algorithms: diagnosis, recoverability, classical, CIDNet wrapper, Zero-DCE++ wrapper, fusion, gain-aware denoise, tuner, guardrails, report/story text, map rendering to uint8 arrays.
- Model weights acquisition script, checkpoint hashes, model registry, GPU placement policy, FP16/FP32 handling, tiling, warmup.
- `MetricSuite` (pyiqa wrappers + custom metrics), used both at request time and by the server's evaluation harness.
- Engine configuration (`lumenx_core/config/default.yaml`).
- Unit, contract and performance tests for the engine.
- A dev-only CLI (`python -m lumenx_core.cli enhance <img> <outdir> --mode hybrid`) that writes visuals with plain `cv2.imwrite` for your own debugging. It is not used in production.

**You must NOT build:** any HTTP/FastAPI/SSE/WebSocket code, upload decoding, atomic file publication, run storage, request queues or locks, dataset loaders, evaluation/ablation scripts, charts, gallery or frontend. Those belong to the server. If you find yourself wanting one of them, you are crossing the boundary.

---

## 2. Folder layout (everything you write lives under `core/`)

```
core/
├── pyproject.toml                 # name = "lumenx-core", pinned deps after validation
├── README.md                      # install, weights, CLI, health, performance table, licences
├── CHANGELOG.md                   # one entry per tag
├── lumenx_core/
│   ├── __init__.py                # __version__, create_engine, create_fake_engine
│   ├── contracts.py               # verbatim from CONTRACT.md §3
│   ├── contract_check.py          # validate_run(items, req) -> list[str]
│   ├── fake.py                    # FakeEngine
│   ├── engine.py                  # RealEngine: implements Engine; owns the run() generator
│   ├── config/default.yaml        # every threshold/weight/range, commented
│   ├── config.py                  # frozen config object + hash
│   ├── types.py                   # internal dataclasses (DiagnosisResult, RecoveryResult, EnhanceParams, …)
│   ├── diagnose.py
│   ├── recoverability.py
│   ├── classical.py
│   ├── fusion.py
│   ├── denoise.py
│   ├── tuner.py
│   ├── guardrails.py
│   ├── report.py                  # template text; captions ≤ 20 words; tech_note separate
│   ├── render.py                  # colormaps, RGBA overlay, heightmap, thumbs → uint8 arrays
│   ├── metrics.py                 # MetricSuite implementation
│   ├── models/
│   │   ├── registry.py            # load once, placement policy, warmup, health
│   │   ├── cidnet.py              # official arch wrapper, FP16, tiling 512/32 cosine blend, padding
│   │   ├── cidnet_arch/           # vendored official HVI-CIDNet architecture code (+ licence notice)
│   │   ├── zerodce.py             # Zero-DCE++ wrapper (live + still fallback + "zerodce" mode)
│   │   └── zerodce_arch/          # vendored official architecture code (+ licence notice)
│   ├── timing.py                  # CUDA-event/synchronized stage timers
│   └── cli.py                     # dev-only
├── scripts/
│   ├── download_weights.py        # downloads what it can; prints manual steps; verifies sha256
│   └── bench.py                   # latency/VRAM benchmark → bench_results.md
├── weights/                       # contents gitignored via core/.gitignore; README.md lists sources, checkpoints, sha256, licences
└── tests/
    ├── test_contract.py           # FakeEngine + RealEngine pass validate_run for every mode
    ├── test_invalid_requests.py   # InvalidRequest before first yield; flag rules
    ├── test_shapes_edgecases.py   # 1×1, odd sizes, black, flat, bright, noisy, gray-as-RGB, alpha
    ├── test_tiling.py             # coverage, padding, seams on edges crossing tile borders
    ├── test_fusion_denoise.py     # weights sum to 1 (1e-5); gain cap actually enforced on output
    ├── test_guardrails.py         # injected failures → one repair; deep not re-run; unresolved disclosed
    ├── test_tuner.py              # 12 candidates, exactly one chosen, deterministic, feasibility-first
    ├── test_determinism.py
    ├── test_cancel.py
    └── test_metrics.py            # directions, None + reason, no NaN
```

Weights directory: environment variable `LUMENX_WEIGHTS_DIR`, defaulting to `core/weights/` (resolved relative to the package, not the current working directory).

Use a `core/.gitignore` for your own ignores (weights, caches, `bench_results/` scratch). Do not edit the root `.gitignore`. If you need something there, ask in `core/HANDOFF.md`.

---

## 3. Algorithm specification

Implement the algorithms of **MASTER_PROMPT.md §3.1–§3.10** with these binding corrections from **FINAL_IMPLEMENTATION_PLAN.md §2, §3.1–§3.2, §4, §4.1 and §5.1** (read all of them). Summary of what must be true:

1. **Processing order (hybrid):** diagnose → recoverability → proxy deep (256 px, once) → tuner (12 candidates, reusing the proxy deep output) → full-resolution classical with the **winning** params → full-resolution tiled deep → fusion → gain-aware denoise + final gain enforcement → guardrails (≤ 1 repair, reusing cached deep output) → metrics (512 px proxy) → report → `EnhanceResult`. This order matches the stage order in CONTRACT.md §5.
2. **Noise** is estimated from up to 100 deterministic flat 32×32 patches at **native resolution**, with an explicit estimator-confidence value. Tiny images have a fallback estimator; having no valid patches gives low confidence, not a crash.
3. **Recoverability** = SNR soft score, penalised by *joint* near-black luminance with low local variance and by low estimator confidence. A single dark colour channel alone never forces `unrecoverable`. A flat black image never receives high confidence. Keep the class names `recovered` / `estimated` / `unrecoverable`. In report text, describe **measured evidence and the action taken** (e.g. "too little light was recorded here, so LumenX limited brightening and smoothed instead of sharpening"). Never claim that detail is guaranteed real or impossible to recover.
4. **Recoverability is a control signal:** it limits local gain (enforced on the *actual output*), caps the deep model's contribution, and raises denoise strength. `no_recoverability` disables all three while still rendering the overlay.
5. **Fusion:** compute the deep/classical/safe weights, smooth them with a guided filter guided by input Y, clamp ≥ 0, renormalise, and fall back to the safe path if the total weight ≈ 0.
6. **Denoise:** blend three guided-filter strengths per pixel by the strength map; suppress chroma noise; protect edges only where the structure is reliable. No non-local means.
7. **Tuner:** 12 candidates at 256 px: 8 prior-focused LHS samples, 2 broader LHS samples, 1 conservative, 1 identity-like. Selection: first exclude hard guardrail violations (feasibility), then rank by the configured normalised score `CLIP-IQA − NIQE − clip% − chroma ΔE + edge retention`, using **fixed normalisation constants** (never per-request min/max). Near-ties go to the more conservative candidate; final ties are broken deterministically. If every candidate is infeasible, choose the conservative one and disclose it. Batch CLIP-IQA over all 12 thumbnails in one forward pass.
8. **Guardrail metrics:** colour shift = chroma-only ΔE with lightness equalised, on valid mid-tones, excluding saturated and very dark pixels (`chroma_delta_e`; `delta_e` = same value). Edge retention only on reliable input structure; return `None` + reason when undefined. Overexposure is incremental vs. input clipping. Halo = overshoot in rings around hotspots.
9. **Repair:** exactly one attempt, using actual spatial controls (stronger highlight-protection weights in hotspot rings, lower fusion α, lower gain cap where needed), re-running only fusion/denoise/recompose. If still failing → conservative classical/safe result plus warning `guardrail_unresolved`.
10. **Modes:** `classical`, `dl` (direct CIDNet; guardrails audit-only, nothing modifies its output), `zerodce` (direct Zero-DCE++, eval baseline, audit-only), `hybrid`. Ablation semantics are in FINAL plan §3.2.
11. **Checkpoint routing:** CIDNet LOLv2-real for mild/moderate/severe, SID for extreme. Record the checkpoint in provenance. Fallback chain: CIDNet → Zero-DCE++ → classical, each step reported in `fallback_chain` + warnings.
12. **Report text:** template-based from recorded decisions only (no LLM). Headline, chips, 4–7 decisions, story steps, recoverability sentence, rejected-candidate reasons ("a brighter version was rejected because it blew out the lamps"). Captions ≤ 20 words, plain English; jargon goes in `tech_note`.
13. **Rendering:** produce every visual key in CONTRACT.md §5 as uint8 arrays at their natural resolution; the server handles resizing to `max_side`. Colormaps must be perceptually uniform (e.g. inferno/viridis). Recoverability overlay = green/amber/red RGBA, alpha 0.45. Heightmap = 128×128 gray of illumination.
14. **Live:** `enhance_frame` uses Zero-DCE++ at native frame size, FP16, ≤ 40 ms for 640×480 on the RTX 2050.

---

## 4. Models, memory and devices

Follow FINAL plan §5.1:
- Load and verify models at `create_engine()`; never download or rebuild inside a request.
- At startup, select a placement policy from **measured** free memory: `resident` (all online weights on GPU, activations serialised) if the worst-case tile + IQA batch fits with the configured reserve, otherwise `swap` (keep the CPU object and move only the active model to GPU per phase, with transfer time included in timings). Report it in `health()`.
- FP16 on CUDA, FP32 on CPU. If an FP16 output is non-finite, retry once in FP32 within the wrapper, then use the fallback chain; every step is timed and warned.
- `torch.no_grad()` / `inference_mode`, `model.eval()`. Use `empty_cache()` only after real placement transitions or OOM recovery, never as the routine strategy.
- Rule 4 of the contract: release references in `finally` when the generator is closed early.
- Timing: CUDA events or synchronised stage boundaries (not per tile). `timings_ms` keys: `diagnose, recoverability, proxy_deep, tuner, classical, deep, fusion, denoise, guardrails, repair, metrics, report, render, core_total` (omit stages that didn't run).

---

## 5. Performance

Core budget: **`core_total` ≤ 1 100 ms p95**, warm, uncached, 1080p, hybrid, RTX 2050 (CONTRACT.md §9). Suggested allocation to validate: diagnose 30 · recoverability 15 · proxy deep + tuner 380 · deep full-res 480 · classical 40 · fusion 25 · denoise 70 · guardrails 30 · metrics 50 · render/report 30 · reserve ≈ −50 (it is tight; profile from day one).

`scripts/bench.py`: 20 representative 1080p images × 3 warm passes. Report mean/p95/max per stage, peak VRAM and process RAM, cold-start time, and placement policy → `bench_results.md`.

If p95 exceeds the budget after real optimisation (buffer reuse, batching, cv2 over skimage, cached derived inputs), apply these **pre-approved levers, in this order, stopping as soon as the budget is met**. Each lever must be recorded in `provenance["perf_levers"]` and listed in the README:
1. Tuner scoring proxy 256 → 192 px.
2. Compute NIQE only for the top 6 candidates by CLIP-IQA (CLIP-IQA still scores all 12; record the scoring recipe as `clipiqa+niqe_top6+custom`).
3. Request-time before/after metrics: CLIP-IQA + NIQE only (BRISQUE marked unavailable with reason).
No other cuts. Candidate count stays 12 and full resolution is preserved up to 12 MP. If the budget is still missed, report the measured numbers honestly in `bench_results.md` and the README.

---

## 6. Milestones and releases (the server agent merges these, so release on time)

| Release tag (on `main`) | Target hour | Content | Gate (must pass before tagging) |
|---|---|---|---|
| **core-v0.1.0** | **≤ 1.0 h** (top priority) | `contracts.py`, `contract_check.py`, `FakeEngine` (all modes, all keys, `enhance_frame`, `MetricSuite`, `health`), pyproject, README install line | `pytest tests/test_contract.py tests/test_invalid_requests.py` green on FakeEngine; `pip install -e core` works in a clean venv from the repo root |
| core-v0.2.0 | ≤ 6 h | `RealEngine` with diagnose, recoverability, classical, denoise, guardrails (audit), report, render; `classical` mode real; `hybrid`/`dl` raise `EngineUnavailable` until core-v0.3.0 | Edge-case tests green; classical numerical pipeline < 100 ms at 1080p; diagnosis < 30 ms; class percentages sum to 100 ± 0.1 |
| core-v0.3.0 | ≤ 12 h | CIDNet (both checkpoints, FP16, tiling), Zero-DCE++ (`enhance_frame`, `zerodce` mode, still fallback), registry/placement/warmup, fusion, gain enforcement, repair; `dl` + `hybrid` real (tuner may be `no_tuner` defaults) | No OOM at 1080p and 12 MP; tiling coverage/seam tests; live ≥ 20 FPS-equivalent (≤ 40 ms/frame at 640×480) over 60 s; fusion sums to 1; one bounded repair |
| core-v0.4.0 | ≤ 14 h | Tuner (12 candidates, constrained ranking, thumbnails, rejection reasons), full `MetricSuite` incl. `confidence_errors`, request-time metrics, final report/story | Every contract test green for all 4 modes; determinism; `bench.py` run and `bench_results.md` committed with real numbers |
| core-v0.4.x | 14–20 h | Bug fixes and performance work only; respond to issues the server agent files in `server/NOTES_FOR_CORE.md` | No contract change; CHANGELOG updated |

After core-v0.4.0, nothing new: fixes only. Freeze at hour 20.

---

## 6a. Git workflow (mandatory)

- Clone the repo. Work **only** on branch `core` (create it from `main` if it doesn't exist). Never commit to `main` directly, never force-push, never touch branch `server`.
- **Write-set rule:** every file you create or modify must be under `core/`. Before each commit, run `git diff --name-only --cached` and abort if any path does not start with `core/`.
- **Release procedure** (only when that milestone's gate passes):
  1. `git checkout core && git merge origin/main` (to pick up the server's merges; this never conflicts because the paths are disjoint).
  2. Run the gate tests.
  3. `git checkout main && git pull && git merge --no-ff core -m "core: release core-vX.Y.Z"`
  4. `git tag core-vX.Y.Z && git push origin main core core-vX.Y.Z`
- Between releases, push branch `core` often so the human can follow your progress.
- **Communication with the server agent goes through files only:** you write `core/NOTES_FOR_SERVER.md` (what changed, what to rely on, known bugs); you read, but never edit, `server/NOTES_FOR_CORE.md` (their bug reports and requests). Check it after every `git merge origin/main`.
- If a merge ever reports a conflict, a write-set rule was broken. Stop and report it in `core/HANDOFF.md`; do not resolve it by editing `server/` files.

---

## 7. Engineering rules

- Every threshold, range and weight lives in `default.yaml` with a comment; the config hash goes into provenance.
- Pure functions per module (`diagnose(img, cfg) -> DiagnosisResult`, …); only `engine.py` and `models/registry.py` hold state.
- Type hints and docstrings everywhere. `ruff` clean. `pytest -q` green on CPU without weights (GPU/weight tests are marked and reported separately, never silently skipped as a "pass").
- Never fabricate numbers: unavailable → `None` + reason. No NaN/Inf anywhere in `data`, `metrics` or `provenance`.
- Pin dependency versions after you validate them (torch/CUDA, opencv-contrib-python, pyiqa, scikit-image, numpy). These pins are the single source of truth for the shared venv: the server installs into the same environment and will not pin these libraries itself. Use exactly one OpenCV wheel (`opencv-contrib-python`). Fail with a clear message at import if `cv2.ximgproc` is missing.
- Vendored model code: keep the original licence headers. README lists Zero-DCE++'s academic/non-commercial licence and the HVI-CIDNet licence.
- Commit small and often with clear messages, prefixed `core:`. Every release gets a CHANGELOG entry stating what the server can now rely on.

---

## 8. Handoff report (update `core/HANDOFF.md` at every release)

Completed features · contract version · the `health()` output on the target laptop · bench numbers · known limitations · open issues · any `CONTRACT_CHANGE_REQUEST.md`.

## 9. Do NOT

- Do not add enhancement models, training or LLMs.
- Do not touch I/O, HTTP, queues, datasets or the frontend.
- Do not create, edit, move or delete anything outside `core/` (including the root README, root `.gitignore` and `CONTRACT.md`).
- Do not change `CONTRACT.md` or the field names/keys it defines.
- Do not silently reduce candidates, resolution or metrics beyond the pre-approved levers in §5.
- Do not tag a version whose gate fails.
