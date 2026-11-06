# PROMPT FOR ANTIGRAVITY — Build `server/` (service, live mode, evaluation harness, basic frontend) in the shared LumenX monorepo

> Repository: https://github.com/ManojDevaraj/LumenX.git. `CONTRACT.md` is at the repo root (frozen).
> Reference documents (read fully): `MASTER_PROMPT.md`, `FINAL_IMPLEMENTATION_PLAN.md` (committed in the repo under `docs/reference/`; read them from there).
> Precedence when documents disagree: **CONTRACT.md > this prompt > FINAL_IMPLEMENTATION_PLAN.md > MASTER_PROMPT.md**.
> Section 7 (file ownership) of the FINAL plan is **obsolete**. All image-processing algorithms now belong to `core/`.

---

## 0. Your role

You are the sole author of the **`server/`** folder of the shared LumenX repository (plus the root `README.md` and root `.gitignore`). It turns the LumenX engine into a fast, reliable, demo-ready service: HTTP API, live streaming, storage, scheduling, evaluation harness and a basic, replaceable frontend.

Another agent (Codex) is building **`core/`** in the **same repository, at the same time**, on its own branch. It contains every algorithm, model and metric, and exposes them through the `Engine` / `MetricSuite` interfaces in `CONTRACT.md`. You never write image-processing or metric math, and you never edit anything under `core/`. You consume core by installing it from the same checkout (`pip install -e ../core`) at the release you last merged. The human merges and reviews everything together at the end.

Your success criterion: swapping `create_fake_engine()` for `create_engine()` requires **zero code changes** beyond a setting, and the whole system meets the end-to-end targets.

---

## 1. Scope

**You own:**
- Upload validation and decoding: type sniffing, EXIF orientation, 8/16-bit → float32 [0,1], grayscale → RGB, alpha split, palette/CMYK conversion, decoded-size limits (decompression-bomb guard), downscale above 12 MP with disclosure.
- Visual encoding (resize to `max_side`, PNG/JPEG per `lossless`), **atomic per-file publication** (temp sibling → close → `os.replace`), URLs, `result.json` written last as the completion marker.
- FastAPI app, Pydantic schemas (the public contract for frontend designers), all endpoints, SSE, WebSocket live, error mapping.
- The single GPU lane, queue limits, busy responses, cancellation on client disconnect, live-frame backpressure and priority.
- Run storage and retention, `GET /runs/{id}`, gallery curation metadata and caching, mock mode.
- Evaluation, ablation, failure mining, confidence-validation harness, dataset loaders, resumable result records, charts, `/benchmarks`.
- The basic frontend (MASTER_PROMPT §10).
- `docs/API.md` (with "Ideas for creative frontends"), `docs/ARCHITECTURE.md`, `docs/FAILURE_ANALYSIS.md` (generated + human notes), README with the 3-minute demo script.

**You must NOT build:** any enhancement, diagnosis, colormap, metric or model code, or weights handling. Anything that computes on pixels beyond decode/resize/encode belongs to core. The only exception is the WebGL relight shader in the browser (MASTER_PROMPT §9.3).

---

## 2. Folder layout (everything you write lives under `server/`, plus root README/.gitignore)

```
README.md                           # ROOT (you own): one venv, `pip install -e core`, `pip install -e server`, run, demo script, troubleshooting
.gitignore                          # ROOT (you own): Python/venv basics; folder-specific ignores live in core/.gitignore and server/.gitignore
server/
├── pyproject.toml                  # name = "lumenx-server"; does NOT list lumenx-core as a URL dependency; README says install core first
├── .gitignore                      # runs/, results/, eval caches
├── HANDOFF.md  NOTES_FOR_CORE.md
├── server.yaml                     # server settings (limits, paths, retention, engine=fake|real, delays)
├── lumenx_server/
│   ├── settings.py
│   ├── app.py                      # FastAPI factory, lifespan: create engine, warmup in lane, static mounts
│   ├── lane.py                     # single GPU lane: one worker thread, bounded queue, priorities, cancel tokens
│   ├── decode.py                   # upload → EnhanceRequest pieces (image, alpha, original_size, notes)
│   ├── publish.py                  # Visual → resized/encoded file, atomic write, URL; encoder thread pool
│   ├── assemble.py                 # StageEvents + EnhanceResult + URLs → EnhanceResponse (pydantic)
│   ├── schemas.py                  # public response/event models (versioned: schema_version="1.0")
│   ├── sse.py                      # SSE framing helpers
│   ├── runs.py                     # run dirs, status (running/done/cancelled/error), retention
│   ├── gallery.py                  # gallery listing + persistent versioned cache
│   ├── routes/
│   │   ├── enhance.py              # POST /enhance, POST /enhance/stream, GET /runs/{id}
│   │   ├── live.py                 # WS /live
│   │   └── meta.py                 # /health /config /gallery /benchmarks
│   └── mock.py                     # mock mode = FakeEngine with stage delays (no separate fake payloads)
├── eval/
│   ├── datasets.py                 # LOL-v1 test, LOL-v2 Real test, DICM, LIME, MEF, NPE, VV, campus folder
│   ├── manifest.py                 # (dataset, image, method, flags) job lists
│   ├── records.py                  # one atomic JSON record per job, keyed by identity hash → resumable
│   ├── run_eval.py                 # methods: classical, zerodce, dl, hybrid
│   ├── run_ablation.py             # hybrid + each flag individually on LOL-v2 Real + DICM
│   ├── run_confidence.py           # MetricSuite.confidence_errors on paired sets (stretch, after required)
│   ├── failure_mining.py           # reads records only; never re-runs enhancement
│   ├── aggregate.py                # records → CSV / MD tables
│   └── charts.py                   # matplotlib bar/radar charts, 6 showcase grids
├── frontend/                       # index.html, app.js, api.js, relight.js, theme.css (no build step)
├── gallery/                        # images + metadata.json (title, why hard, attribution)
├── runs/  results/                 # gitignored, generated
├── docs/API.md  docs/ARCHITECTURE.md  docs/FAILURE_ANALYSIS.md
└── tests/
    ├── test_decode.py              # every format/edge case, bomb limit, >12 MP downscale, alpha
    ├── test_publish.py             # atomicity, max_side, PNG vs JPEG, no partial files visible
    ├── test_api_contract.py        # all endpoints vs schemas using FakeEngine; validate_run on every run
    ├── test_sse.py                 # ordering, ids, exactly one terminal event, chunk-split parsing, assets fetchable when announced
    ├── test_lane.py                # serialization, busy 503, cancel on disconnect, live priority, no deadlock
    ├── test_live.py                # framing, backpressure (stale frames dropped), pause status
    ├── test_eval_resume.py         # killed run resumes; identity hashing; aggregates match records
    └── test_gallery_cache.py       # invalidation on input/config/core version/checkpoint change
```

---

## 3. Engine integration rules

- Create exactly one engine at startup: `create_fake_engine(stage_delay_ms=…)` when `engine: fake` in `server.yaml` or `LUMENX_MOCK=1`, otherwise `create_engine()`. Run `warmup()` **inside the lane** before reporting ready.
- **Every** engine call (`run` iteration, `enhance_frame`, `warmup`, all `MetricSuite` calls) executes on the lane's single worker thread (CONTRACT.md §4.3). The event loop never touches the engine.
- Bridge the generator to async: the worker iterates `engine.run(req)` and puts each item on an `asyncio.Queue` via `loop.call_soon_threadsafe`. Visual encoding goes to a **separate encoder thread pool**, so the GPU lane keeps working while earlier visuals encode (pipelining keeps the server inside its 400 ms budget).
- Map exceptions: `InvalidRequest` → 422 (`code` in body), `EngineUnavailable` → 503, `Cancelled` → run status `cancelled`, any other exception → 500 / SSE `error`. Never leak tracebacks to clients; log them.
- Run `validate_run` (from `lumenx_core.contract_check`) on every run in tests, and in production when `debug_validate: true`. Log violations as errors. Treat them as core bugs to report to the human, not something to patch around.

---

## 4. Public API (MASTER_PROMPT §5 + FINAL plan §8.2 decisions)

All responses are Pydantic models with `schema_version: "1.0"` and no NaN/Inf.

| Endpoint | Behaviour |
|---|---|
| `POST /enhance` | multipart: `file`, `mode` ∈ {hybrid, classical, dl} (default hybrid), `ablate` = JSON array string, `mock` bool. Returns `EnhanceResponse` once the run finishes. |
| `POST /enhance/stream` | Same input; `text/event-stream`. Events: `stage`, `candidate`, `warning`, then exactly one terminal `done` (full `EnhanceResponse`) or `error`. SSE `id:` = monotonic sequence number; `event:` name equals the JSON `event` field. A stage/candidate event is sent **only after its visuals are published**. Clients consume it with streaming `fetch` (not `EventSource`). |
| `GET /runs/{run_id}` | Completed `EnhanceResponse`; otherwise `{status: running|cancelled|error}`; 404 if unknown. Artifacts are served from `/files/runs/{run_id}/…` so they can never collide with this route. |
| `GET /health` | Server status (lane busy, queue depth, live sessions, uptime) + `engine.health()` verbatim under `engine`. |
| `GET /config` | Server public settings + `engine.public_config()`. |
| `GET /gallery` | Items: id, title, why_hard, attribution, image URL, cached run_id (if cache is valid). |
| `GET /benchmarks` | Tables (JSON) + chart URLs from `results/`, or `{status: "not_ready", progress}`. |
| `WS /live` | Client sends binary JPEG frames. Server replies per processed frame with a JSON text message `{seq, fps, proc_ms, queue_ms, dropped, status}` followed by one binary JPEG. When paused (a still request holds the lane), it sends status messages only. |

**`EnhanceResponse`** = every minimum field in MASTER_PROMPT §5, plus: `schema_version`, `status`, `warnings`, `provenance` (engine provenance + server fields `original_size`, `processed_size`, `downscaled`, `cache_hit`), `timings_ms` (engine timings + `decode`, `publish`, `queue`, `server_total`, `total`), `metrics` (verbatim engine metrics), `tuner` (candidates with `thumb_url`, `feasible`, `reasons`), `guardrails`, `decisions`, `story` (with URLs), `stages` (one entry per stage event: name, title, caption, tech_note, `image_url` of `primary_key`, `t_ms`), `maps` (illumination_gray←`L_gray`, snr, recoverability_overlay, gain, fusion←`fusion_map`, denoise_strength, noise←`noise_map`, heightmap; missing keys → `null`), `relight` (URLs from the keys), `input`/`final` URLs (+ `final.preview_url`).

**Status codes:** 413 decoded size above the hard limit, 415 unsupported media, 422 invalid args / undecodable / engine `InvalidRequest`, 503 queue full or engine unavailable (with `Retry-After`). After SSE headers are sent, errors only arrive as a terminal `error` event.

---

## 5. Scheduling, live and cancellation (FINAL plan §5.2)

- One worker thread = one GPU lane. Bounded still-request queue (`max_pending` in settings, default 2); beyond it return 503.
- A still request owns the lane for its entire run. Live frames have lower priority: while a still request runs, live sessions receive `status: "paused"`.
- Live: one **latest-frame slot** per session (a new frame replaces the pending one; increment `dropped`). Limit the number of live sessions (default 1). Measure the delivered FPS over 60 s for live-only operation.
- Client disconnect (SSE or plain POST) sets the request's `cancel` event. Core stops at the next stage/tile boundary. Run status becomes `cancelled`, and the lane is released in `finally`.
- Queue time is measured separately and never hidden inside processing time.

---

## 6. Storage, gallery and mock

- `runs/{run_id}/` with core visual keys as filenames (`{key}.png|jpg`), `input.png` (normalised processed input, alpha preserved), `final.png` full-res. `result.json` is written last. Retention policy by count/age in settings; cleanup never deletes a running run.
- Gallery: 8–10 curated hard images (MASTER_PROMPT §11; a human teammate provides photos + attribution). Cache key = sha256(input bytes) + engine `config_hash` + `core_version` + `checkpoint_hash` + mode. Cached runs are reused across restarts and invalidated when any part of the key changes. Responses mark `cache_hit: true`. Cached playback is never reported as processing latency.
- **Mock mode** is simply the FakeEngine through the same code path (with stage delays so streaming is visible). Responses carry `engine.fake = true`, and the UI shows a visible "MOCK" badge. Mock is never presented as real inference.

---

## 7. Evaluation harness (MASTER_PROMPT §7 + FINAL plan §11.3)

- Datasets: LOL-v1 test, LOL-v2 Real test (paired); DICM, LIME, MEF, NPE, VV (unpaired); `campus/` (20–50 own photos, unpaired). Loaders verify file counts and report missing datasets as explicit gaps.
- Methods = engine modes `classical`, `zerodce`, `dl`, `hybrid`. Ablation = `hybrid` with each of the 8 flags individually (skip contradictory ones), on LOL-v2 Real + DICM.
- Per job: run the engine on the lane, save the raw `final` output, then call `MetricSuite.paired` (if GT exists), `unpaired` and `custom` at **full resolution**. Record latency from engine `timings_ms` and provenance. Report raw PSNR and `psnr_gt_mean` as separate, clearly labelled columns. Ground truth never influences enhancement.
- Records: one atomic JSON file per job, keyed by hash(dataset, image, method, flags, core_version, config_hash, checkpoint_hash, metric versions). Resume skips matching records. Aggregates are always rebuilt from records.
- Run sequentially with exclusive engine use (never during a live demo). Show progress + ETA from pilot throughput; partial completion is published with counts and is never implied to be full coverage.
- Outputs: `results/benchmark.csv|md`, `results/ablation.csv|md`, charts (PNG), 6 showcase side-by-side grids, `docs/FAILURE_ANALYSIS.md` (10 worst per method per metric, direction-aware, tagged using recorded guardrail data, with a "Human notes:" line under each).
- Stretch, only after the required runs: `run_confidence.py` → error by recoverability class, with `no_recoverability` on vs. off. If low-confidence regions show higher error, that is the key evidence slide for the honesty layer.

---

## 8. Basic frontend (MASTER_PROMPT §10, all items)

Plain HTML + vanilla JS, no build step, served at `/`. It consumes **only** the documented API. All styling lives in `theme.css` CSS variables; every section has a `data-role`; JS is split into `api.js` (fetch + streaming-SSE parser that handles split chunks and multiple events per chunk) and small `render*` functions. Required: upload/drag-drop + gallery, live stage list with thumbnails, before/after slider, stage strip, overlay toggles (recoverability, gain, fusion, noise), WebGL relight (`out = base · clamp(L^(γ−1) · exposure)`, epsilon-protected, bounded sliders, fallback message if WebGL is unavailable; γ=1, exposure=1 reproduces the base image), chips/headline/decisions, candidate grid with the winner and rejection reasons, metrics table with better/worse arrows, timing bar, mode comparison (classical/dl/hybrid side by side), live webcam tab, benchmarks tab, MOCK badge. Keep it clean and simple; the human's team redesigns it later.

---

## 9. Milestones (aligned to core tags)

| # | Hour | Deliverable | Gate |
|---|---|---|---|
| S0 | 0–1 | Work that doesn't need core: `server/` scaffold, root README/.gitignore, `settings`, `decode.py`, `publish.py`, `runs.py`, `eval/datasets.py`, frontend skeleton + `api.js` SSE parser, `docs/API.md` draft from CONTRACT.md | `test_decode`, `test_publish` green |
| S1 | 1–4 | Merge release **core-v0.1.0** into `server`; `pip install -e core`. Lane, routes, SSE, assemble, schemas, mock mode end-to-end on FakeEngine | `test_api_contract`, `test_sse`, `test_lane` green; the frontend shows a full streamed run on FakeEngine |
| S2 | 4–8 | Live WS on FakeEngine, `/health` `/config` `/gallery`, run retention, cancellation, all frontend sections | `test_live` green; every frontend item works on mock |
| S3 | 6–10 | Merge **core-v0.2.0** (classical real); eval harness end-to-end in `classical` mode on a 5-image smoke set; records/resume/aggregate/charts | `test_eval_resume` green; smoke tables generated |
| S4 | 12–14 | Merge **core-v0.3.0 → core-v0.4.0**; real hybrid/dl through the API and live; end-to-end latency measurement (§10) | All tests green against the real engine; numbers recorded |
| S5 | 14–16 | Launch full eval → ablation → failure mining sequentially (overnight) | Progress/ETA visible; partial results served by `/benchmarks` |
| S6 | 16–20 | Gallery curation + cache, benchmarks tab on real data, README demo script, `docs/*` complete, robustness pass | `pytest -q` green; demo script runs twice end-to-end |
| S7 | 20–21 | **Freeze**: fixes only; record the backup demo video | — |

If core misses a release, keep building on the latest release + FakeEngine, write the blocker in `server/HANDOFF.md` and the request in `server/NOTES_FOR_CORE.md`. Never reimplement core functionality locally and never edit `core/` to unblock yourself.

---

## 10. End-to-end performance (CONTRACT.md §9)

`scripts/e2e_bench.py`: 20 representative 1080p images × 3 warm, uncached passes via local HTTP `POST /enhance` (hybrid). Report p50/p95/max of `total`, and separately `decode`, `publish`, `queue`, engine `core_total`, `server_total` (= total − core_total), plus SSE time-to-first-stage and time-to-`final_preview`. Targets: `server_total` ≤ 400 ms, `total` p95 ≤ 1 500 ms. Server-side levers you may use freely (no approval needed): pipelined encoding, PNG compression level 1, JPEG for non-lossless keys, resize before encode, orjson responses, avoiding copies. Commit `e2e_results.md` with real numbers. Never exclude encoding or disk writes from `total`.

---

## 11. Engineering rules

- Python 3.10/3.11, FastAPI, uvicorn (1 process), pydantic v2, python-multipart, pillow (+ pillow-heif optional), orjson, httpx/pytest for tests, matplotlib/pandas for eval. Pin versions.
- **Shared venv rule:** both parts install into one venv. Do NOT declare any `opencv-*`, `torch`, `numpy` or `pyiqa` version in `server/pyproject.toml` (multiple OpenCV wheels in one venv break `cv2`). numpy comes in via core. Use Pillow for decode/resize/encode (PNG `compress_level=1`, JPEG quality 90). If you need a library that core also uses, take whatever version core pins.
- Type hints, docstrings, `ruff` clean. Tests run on CPU with FakeEngine. Real-engine tests are marked and reported separately.
- Structured logs: run_id, stage, ms, lane wait, encode ms.
- Security basics: size limits before full decode, content sniffing, no path traversal in run IDs (UUID4 only), CORS configurable, no stack traces in responses.
- `server/HANDOFF.md` updated at each milestone: what works, the core release in use, numbers, blockers, any `CONTRACT_CHANGE_REQUEST.md`.

## 11a. Git workflow (mandatory)

- Clone the repo. Work **only** on branch `server` (create it from `main` if it doesn't exist). Never commit to `main` directly, never force-push, never touch branch `core`.
- **Write-set rule:** every file you create or modify must be under `server/`, or be the root `README.md` / root `.gitignore`. Before each commit, run `git diff --name-only --cached` and abort if any other path appears. Prefix commit messages with `server:`.
- **Picking up a core release:** `git fetch --tags && git merge core-vX.Y.Z` into `server`, re-run `pip install -e core`, read `core/NOTES_FOR_SERVER.md` and `core/CHANGELOG.md`, then run your full test suite against both FakeEngine and (if available) the real engine.
- **Merging your milestone into `main`** (only when its gate passes):
  1. `git checkout server && git merge origin/main`
  2. Run the gate tests.
  3. `git checkout main && git pull && git merge --no-ff server -m "server: milestone SN"`
  4. `git tag server-sN && git push origin main server server-sN`
- Push branch `server` often between milestones.
- **Communication with the core agent goes through files only:** you write `server/NOTES_FOR_CORE.md` (bug reports with a minimal reproduction, validate_run violations, requests); you read, but never edit, `core/NOTES_FOR_SERVER.md`.
- If a merge ever reports a conflict, a write-set rule was broken. Stop and report it in `server/HANDOFF.md`; do not resolve it by editing `core/` files.

## 12. Do NOT

- Do not compute anything on pixels beyond decode/resize/encode (except the browser relight shader).
- Do not edit `CONTRACT.md`, and do not create, edit, move or delete anything under `core/`. Do not copy core code into `server/`.
- Do not call the engine from more than one thread.
- Do not commit to `main` except through the merge procedure in §11a, and never touch branch `core`.
- Do not report cached or mock results as real processing latency.
- Do not start a milestone while the previous gate is failing.
