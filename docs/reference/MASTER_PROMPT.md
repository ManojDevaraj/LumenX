# MASTER PROMPT — "LumenX": Recoverability-Aware Adaptive Low-Light Restoration Engine

> Paste this whole file into your coding agent (Claude Code / Cursor / etc.) as the project brief.
> Then build milestone by milestone (Section 12). Do not let the agent skip ahead.

---

## 0. Your role and mission

You are a senior computer-vision + backend engineer building a hackathon system in **24 hours** with a team of 3.
Build **LumenX**, a low-light image restoration engine that:

1. **Diagnoses** each image (darkness, light sources, noise, colour cast, clipping).
2. **Adapts** its parameters per image with a quality-aware self-tuning search (no fixed parameters).
3. **Fuses** classical Retinex processing with a deep model (HVI-CIDNet), pixel by pixel.
4. **Denoises in proportion to the gain**: noise is amplified by the brightening factor, so denoise strength follows the gain map.
5. **Knows what it can't recover**: it outputs a per-pixel *Recoverability Map* (recovered / estimated / unrecoverable) and never invents detail where there was no real signal.
6. **Explains itself**: it streams every stage live and writes a plain-English report a non-technical person can understand.

**Primary principle:** the backend is the product. It must return *rich, structured, visual data* (maps, stages, decisions, candidates, metrics, timings) so that **any** frontend can turn it into a stunning experience. The included frontend is deliberately basic; the team will redesign it later, so it must be fully decoupled.

**Second principle:** a working, reliable system beats an ambitious broken one. Every milestone must end in a runnable state.

---

## 1. Hard constraints

- GPU: **NVIDIA RTX 2050, 4 GB VRAM** (laptop). CPU fallback must work (slower).
- Python 3.10/3.11, PyTorch (CUDA), OpenCV-contrib (`cv2.ximgproc`), scikit-image, numpy, pyiqa, FastAPI, uvicorn, pydantic v2, pyyaml, pandas, matplotlib.
- Everything free/open-source. No paid APIs. No LLM calls at runtime (report text is template-based).
- VRAM safety: FP16 inference, tiled inference for images > 1 MP (tile 512, overlap 32, feathered blending), `torch.no_grad()`, load models once at startup, never hold more than one large model's activations at once.
- **Latency budget (1080p, GPU): ≤ 1.5 s end-to-end** for `/enhance` (excluding first-call warmup). Measure and report per-stage timings.
- Deterministic: fixed seeds, same input gives the same output.

---

## 2. Repository structure (create exactly this)

```
lumenx/
├── README.md                      # setup, run, demo script, architecture diagram (mermaid)
├── config.yaml                    # ALL thresholds/params live here, none hard-coded
├── requirements.txt
├── weights/                       # model weights (gitignored) + README with download links
├── scripts/
│   ├── download_weights.py        # downloads what it can; prints manual instructions for the rest
│   ├── eval.py                    # benchmark: methods × datasets × metrics → CSV/MD/plots
│   ├── ablation.py                # component on/off study → CSV/MD/plots
│   ├── failure_mining.py          # auto-collect worst cases + reason tags
│   └── warmup.py
├── lumenx/
│   ├── __init__.py
│   ├── pipeline.py                # orchestrator; emits stage events; supports ablation flags
│   ├── diagnose.py                # illumination, noise σ, SNR, cast, non-uniformity, clipping
│   ├── classical.py               # Retinex decomposition/enhancement, CLAHE, colour correction
│   ├── deep/
│   │   ├── cidnet.py              # HVI-CIDNet wrapper (FP16, tiled)
│   │   └── zerodce.py             # Zero-DCE++ wrapper (live mode)
│   ├── fusion.py                  # per-pixel fusion weights + guided-filter smoothing
│   ├── denoise.py                 # gain-aware multi-strength guided/bilateral denoise
│   ├── recoverability.py          # SNR-based recoverability map + gain capping
│   ├── tuner.py                   # quality-aware parameter search on proxy image
│   ├── guardrails.py              # overexposure / colour shift / edge loss checks + retry
│   ├── metrics.py                 # pyiqa wrappers + custom edge/noise/ΔE metrics
│   ├── report.py                  # plain-English diagnosis + decision narrative + story steps
│   ├── viz.py                     # colormaps, overlays, 3D-ready heightmap export, PNG encoding
│   └── utils.py                   # io, timing context manager, tiling, color conversions
├── api/
│   ├── main.py                    # FastAPI app, static serving, CORS
│   ├── schemas.py                 # pydantic response models (the frontend contract)
│   ├── routes_enhance.py          # POST /enhance, GET /enhance/stream (SSE), GET /runs/{id}
│   ├── routes_live.py             # WS /live (webcam, Zero-DCE++)
│   ├── routes_meta.py             # /health, /gallery, /benchmarks, /config
│   └── mock.py                    # MOCK MODE: returns realistic fake payloads instantly
├── frontend/                      # BASIC, replaceable (see Section 10)
│   ├── index.html
│   ├── app.js
│   ├── relight.js                 # WebGL relighting shader
│   └── theme.css                  # all visual styling via CSS variables only
├── gallery/                       # 8–10 curated hard demo images + metadata.json
├── runs/                          # per-request outputs (gitignored), served statically
├── results/                       # eval/ablation outputs (CSV, MD, PNG charts, failure gallery)
├── docs/
│   ├── API.md                     # full frontend contract with example payloads
│   ├── ARCHITECTURE.md
│   └── FAILURE_ANALYSIS.md        # generated + hand-annotated
└── tests/
    ├── test_pipeline_shapes.py
    ├── test_api_contract.py
    └── test_guardrails.py
```

---

## 3. Pipeline: exact algorithms

All images internally are float32 RGB in [0,1]. `Y` = luminance (BT.709). Every stage records: output image (or map), duration (ms), a one-line human caption, and key numbers.

### 3.1 Diagnose (`diagnose.py`), target < 30 ms at 1080p

Compute on a 512px-long-side proxy where possible, then upsample maps.

- **Illumination map L** (LIME-style): `L0 = max(R,G,B)`, refined with `cv2.ximgproc.guidedFilter(guide=Y, src=L0, radius≈15, eps≈1e-3)`. Clip to [ε, 1].
- **Brightness level**: mean(Y), median(Y), % pixels with Y < 0.05 (“crushed shadows”).
- **Noise σ**: `skimage.restoration.estimate_sigma` on Y (or MAD of the Laplacian over low-gradient pixels). Also a **local noise map** (σ in 32×32 blocks over flat regions, interpolated).
- **SNR map**: `SNR = local_mean(Y, 15×15) / (σ_local + ε)`.
- **Light sources / non-uniformity**: threshold L > 0.85 → connected components → count + bounding boxes (“hotspots”). Non-uniformity score = std(log L) normalised.
- **Colour cast**: convert to Lab; mean a*, b* over mid-tones; cast = direction + magnitude (e.g. "warm/orange", "green", "blue").
- **Clipping**: % pixels with any channel > 0.98 (highlights) and < 0.02 (shadows).
- **Severity class**: `extreme` / `severe` / `moderate` / `mild` from thresholds in config.

Output a `Diagnosis` object (numbers + labels + severity per item + hotspot boxes).

### 3.2 Recoverability (`recoverability.py`), the main differentiator

- From the **input** SNR map, classify each pixel:
  - `recovered` (SNR ≥ τ_high): real signal; enhance fully.
  - `estimated` (τ_low ≤ SNR < τ_high): partial signal; enhance with stronger denoise.
  - `unrecoverable` (SNR < τ_low): essentially noise; **cap the gain**, denoise strongly, and do not let the DL output hallucinate texture there (fusion weight shifts toward a smooth classical result).
- Smooth class boundaries (guided filter on a soft score in [0,1]), then quantise for display.
- Output: soft map (float), 3-class map, % area per class, and an RGBA overlay PNG (green / amber / red, alpha 0.45).
- Report line example: *"18% of this image had too little light to recover real detail. LumenX smoothed those areas instead of inventing texture."*

### 3.3 Classical path (`classical.py`)

- Retinex: `R = I / L` (per channel, L broadcast), clipped. Adjusted illumination `L' = L^γ` (γ from tuner, typically 0.3–0.7), with highlight protection (where L > 0.8, blend L' back toward L).
- Recompose `E_c = R · L'`.
- CLAHE on the Lab L-channel (clipLimit and tile size from tuner).
- Colour correction: partial gray-world in Lab with strength ∝ cast magnitude (never full correction; preserve scene mood such as warm street lights).
- Keep R and L' as outputs (needed for relighting).

### 3.4 Deep path (`deep/cidnet.py`)

- **HVI-CIDNet** (official repo: `github.com/Fediory/HVI-CIDNet`). Load pretrained weights:
  - LOLv2-real weights for `moderate/severe`, **SID (Sony-Total-Dark) weights for `extreme`** darkness. Selection is automatic, based on diagnosis.
- FP16, tiled inference (tile 512, overlap 32, cosine-feathered blending), pad to a multiple of the network stride.
- If weights are missing: log a clear warning and fall back to Zero-DCE++ so the system never crashes.

### 3.5 Live path (`deep/zerodce.py`)

- **Zero-DCE++** (official Zero-DCE extension repo weights). Used ONLY for `/live` webcam streaming. Target ≥ 20 FPS at 640×480 on RTX 2050.

### 3.6 Fusion (`fusion.py`)

Per-pixel weights (sum to 1) between `E_dl` (deep), `E_c` (classical), and `I_safe` (input, gamma-lifted gently):
- `w_dl` high in dark regions with recoverable signal: `sigmoid((t_dark − L)/s) · rec_soft`.
- `w_c` high in mid-tones and in `unrecoverable` zones (smooth, no hallucination).
- `w_safe` high in highlights / hotspots (L > 0.8), to prevent blow-out and halos around lamps.
- **Smooth all weight maps with a guided filter guided by input Y** (not Gaussian), to avoid seams and halos.
- Global fusion bias `α` comes from the tuner.
- Output: fused image + the 3 weight maps (export as one RGB "fusion map": R=dl, G=classical, B=safe).

### 3.7 Gain-aware denoising (`denoise.py`)

- Gain map `G = Y_fused / (Y_in + ε)`, clipped.
- Denoise strength map `S = clip(k · G · σ_local / (σ_ref), 0, 1)`, boosted in `estimated`/`unrecoverable` zones.
- Efficient trick: compute 3 denoised versions (light / medium / strong guided-filter or bilateral; optionally fastNlMeans on the strong one at reduced resolution) and **blend per pixel by S**. Denoise luminance more than chroma, but also suppress chroma noise.
- Edge protection: reduce S where the input gradient magnitude is high and SNR is good.
- Output: denoised image, gain map, strength map.

### 3.8 Self-tuning search (`tuner.py`), the adaptive strategy

- Work on a 256px proxy. Search space (from config): γ, CLAHE clip, denoise k, fusion α, colour-correction strength.
- Generate ~12–16 candidates (Latin-hypercube sample seeded by diagnosis-based priors, e.g. extreme darkness → lower γ prior).
- Run the cheap proxy pipeline per candidate (deep output computed ONCE on the proxy and reused; only cheap steps vary).
- Score = `w1·CLIP-IQA + w2·(−NIQE_norm) − w3·clip_pct − w4·ΔE_shift + w5·edge_retention` (weights in config).
- Pick the best; run the full-resolution pipeline with it.
- Return **all candidates** (params, sub-scores, total, 128px thumbnail, chosen flag) for the frontend to visualise. Target < 400 ms total.

### 3.9 Guardrails (`guardrails.py`)

Check final output:
- Overexposure: highlight clip % > threshold (and > input's clip % + margin).
- Colour shift: mean ΔE2000 between input chroma direction and output (on mid-tones) > threshold.
- Detail loss: edge retention (correlation of Sobel magnitudes, input-normalised) < threshold.
- Halo check: overshoot around hotspot boundaries.
If any check fails, retry once with the next-best candidate or safer params (higher γ, lower α, stronger highlight protection). Log `{check, value, threshold, action}` and include it in the response and report. Never fail silently.

### 3.10 Report and story (`report.py`)

Template-based, no LLM. Produces:
- `headline`: one sentence (e.g. *"Very dark street scene with 3 bright lamps and heavy sensor noise, restored with shadows lifted 6.2× and lamps protected."*).
- `diagnosis_chips`: short labels with severity (`high/med/low`) and the value.
- `decisions`: 4–7 plain-English bullet points explaining WHAT was done and WHY.
- `story`: ordered list of steps `{title, caption (≤ 20 words, no jargon), image_url, highlight_map_url?}` so any frontend can animate the "watch it think" reveal.
- `recoverability_summary`: % per class + one honest sentence.
- `jargon_free`: true. Every caption must be understandable by a non-technical person; technical names go in a separate `tech_note` field.

---

## 4. Outputs per run

Saved to `runs/<run_id>/` and served at `/runs/<run_id>/...`:
- `input.png`, `final.png` (full-res), `final_preview.jpg` (≤ 1280px)
- Stage images (≤ 768px): `illumination.png` (colormapped), `reflectance.png`, `classical.png`, `deep.png`, `fused.png`, `denoised.png`, `final.png`
- Maps (≤ 512px): `L_gray.png` (8-bit, for WebGL relight), `snr.png`, `recoverability_overlay.png` (RGBA), `gain.png`, `denoise_strength.png`, `fusion_map.png`, `noise_map.png`
- `heightmap.json` or `L_height.png` (downsampled 128×128 illumination for 3D terrain visualisations)
- `candidates/*.jpg` thumbnails
- `result.json` (full response)

---

## 5. API (`api/`): the frontend contract

All responses are typed with pydantic models in `schemas.py` and documented in `docs/API.md` with full example payloads.

- `POST /enhance` (multipart `file`, optional form: `mode=hybrid|classical|dl`, `ablate=[...]` component flags, `mock=bool`) → `EnhanceResponse`
- `POST /enhance/stream` → **Server-Sent Events**. Emit an event as each stage finishes: `{"event":"stage","name":"diagnose","caption":"...","image_url":"...","data":{...},"t_ms":12}`, then `{"event":"candidate",...}` for each tuner candidate, then `{"event":"done","result":EnhanceResponse}`. This drives live "watch it think" UIs. **Required.**
- `GET /runs/{run_id}` → stored result.
- `GET /gallery` → curated demo images with titles and difficulty tags.
- `GET /benchmarks` → eval + ablation tables (JSON) and chart URLs from `results/`.
- `GET /config` → current tunable params (read-only).
- `GET /health` → device, VRAM, models loaded, warm status.
- `WS /live` → client sends JPEG frames, server returns enhanced JPEG frames + fps + latency.

`EnhanceResponse` (minimum fields):
```json
{
  "run_id": "str",
  "mode": "hybrid",
  "input": {"url": "...", "width": 1920, "height": 1080},
  "final": {"url": "...", "preview_url": "..."},
  "headline": "str",
  "diagnosis": {"chips": [{"label":"Very dark","value":0.07,"severity":"high"}], "raw": {...}, "hotspots":[[x,y,w,h]]},
  "stages": [{"name":"illumination","title":"Where the light is","caption":"...","tech_note":"...","image_url":"...","t_ms":9}],
  "maps": {"illumination_gray":"...","snr":"...","recoverability_overlay":"...","gain":"...","fusion":"...","denoise_strength":"...","noise":"...","heightmap":"..."},
  "relight": {"base_url":"final.png","L_url":"L_gray.png","gamma_default":1.0,"exposure_default":1.0},
  "recoverability": {"recovered_pct":71.2,"estimated_pct":17.5,"unrecoverable_pct":11.3,"sentence":"..."},
  "tuner": {"candidates":[{"id":0,"params":{},"scores":{},"total":0.71,"thumb_url":"...","chosen":true}],"search_ms":310},
  "guardrails": [{"check":"overexposure","value":0.4,"threshold":2.0,"passed":true,"action":null}],
  "decisions": ["..."],
  "story": [{"title":"...","caption":"...","image_url":"...","highlight_map_url":null}],
  "metrics": {"before":{"niqe":6.8,"clipiqa":0.22,"brisque":48.1},"after":{"niqe":3.6,"clipiqa":0.67,"brisque":21.4},"edge_retention":0.93,"noise_sigma":{"before":0.041,"after":0.012},"delta_e":3.1},
  "timings_ms": {"diagnose":14,"tuner":310,"deep":420,"classical":35,"fusion":22,"denoise":60,"guardrails":15,"total":910},
  "device": "cuda:0"
}
```

**Mock mode** (`mock=true` or env `LUMENX_MOCK=1`): returns a realistic payload instantly using pre-generated assets, including SSE streaming with artificial delays. The frontend team must be able to build everything before the backend is finished.

---

## 6. Ablation and modes

`pipeline.py` accepts flags to disable each component: `no_tuner` (fixed default params), `no_fusion` (DL only), `no_classical`, `no_dl`, `no_gain_denoise` (uniform denoise), `no_recoverability`, `no_guardrails`, `no_color_correction`. Modes: `classical` (classical path only), `dl` (CIDNet only), `hybrid` (full).

---

## 7. Evaluation (`scripts/eval.py`, `scripts/ablation.py`)

- Datasets: **LOL-v2 Real (test)** (paired), **LOL-v1 test** (paired), **DICM, LIME, MEF, NPE, VV** (unpaired, used as "unseen"), plus 20–50 **own campus night photos** (unseen, real-world).
- Methods: Classical-only, Zero-DCE++, HVI-CIDNet, **LumenX-Hybrid**. (Optional: plain CLAHE and plain gamma as naive baselines.)
- Metrics: PSNR, SSIM, LPIPS (paired, report both normal and GT-mean-adjusted PSNR and state it clearly); NIQE, BRISQUE, CLIP-IQA, MUSIQ (unpaired); edge retention; noise σ before/after; ΔE colour shift; highlight clip %; latency (mean, p95) on the RTX 2050.
- Outputs: `results/benchmark.csv`, `results/benchmark.md`, bar/radar charts (PNG), and per-image side-by-side grids for 6 showcase images.
- Ablation: run hybrid with each flag off on LOL-v2 Real + DICM → `results/ablation.csv/.md` + chart showing how much each component contributes.
- Failure mining (`failure_mining.py`): for each method, save the 10 worst images by metric, auto-tag the reason (overexposure / colour shift / noise / halo / detail loss) via guardrail checks, and write `docs/FAILURE_ANALYSIS.md` with images and tags (leave a "Human notes" line under each).
- All eval scripts must be resumable and runnable overnight unattended.

---

## 8. Performance and engineering rules

- `utils.timer("stage")` context manager on every stage → timings go into the response.
- Models load once at startup; `scripts/warmup.py` runs 2 dummy inferences.
- All thresholds/weights in `config.yaml`, with comments explaining each.
- Structured logging (stage, ms, VRAM used).
- Validate uploads (type, max 12 MP; downscale larger images and say so in the response).
- Graceful degradation: missing CIDNet weights → Zero-DCE++; no GPU → CPU with a warning field in the response.
- Type hints and docstrings everywhere; tests must pass: `pytest -q`.

---

## 9. "Wow" requirements (non-negotiable)

These are what make a non-technical viewer surprised. The backend must supply the data for each:

1. **Progressive reveal**: SSE stream lets the UI show the image "being understood" step by step.
2. **Honesty layer**: the recoverability overlay + sentence.
3. **Live relighting**: `L_gray.png` + base image let the browser relight in real time with a WebGL shader: `out = base · clamp(L^(γ−1) · exposure)` (both sliders, 60 fps, no server call).
4. **Visible adaptivity**: tuner candidates with scores; the winner is flagged.
5. **Plain-English explanations**: headline, chips, decisions, story captions without jargon.
6. **Proof**: before/after metrics with direction (better/worse) and per-stage timing.
7. **Live webcam** via `/live` that visibly brightens a dark room in real time.

---

## 10. Basic frontend (replaceable by design)

Build a **minimal, clean, functional** single-page UI in plain HTML + vanilla JS (no build step), served by FastAPI at `/`. It exists to prove the API works and to give designers a reference.

Must include:
- Upload / drag-drop + gallery picker.
- SSE-driven progress list showing each stage caption and thumbnail as it arrives.
- Before/after comparison slider.
- Stage strip (click to view full size).
- Overlay toggles: recoverability, gain, fusion map, noise.
- Relight panel (WebGL canvas, γ + exposure sliders) in `relight.js`.
- Diagnosis chips, headline, decisions list.
- Tuner candidates grid (winner highlighted).
- Metrics table (before/after with arrows) and timing bar.
- Mode switch (classical / dl / hybrid) for side-by-side comparison.
- Live webcam tab.
- Benchmarks tab reading `/benchmarks`.

**Designed for redesign:**
- ALL styling in `theme.css` via CSS variables (`--bg`, `--fg`, `--accent`, `--radius`, `--font`…). No inline styles.
- JS split into small pure functions: `api.js`-style fetch helpers, `render*` functions that take data and return DOM. No framework lock-in.
- Every element has semantic `data-role="..."` attributes so designers can restyle or replace sections.
- The frontend must ONLY use the documented API: no hidden coupling.
- Add a `docs/API.md` section "Ideas for creative frontends" listing which fields can power: a flashlight/torch reveal (final vs input + recoverability), a 3D illumination terrain (heightmap), a story-mode scroll (story[]), and a candidate tournament animation (tuner.candidates).

---

## 11. Demo assets

- `gallery/`: 8–10 hard images (street at night, indoor with one lamp, backlit window, near-black scene, coloured neon cast, high-noise phone shot, a face in dim light, a campus photo). `metadata.json` with title + why it's hard.
- Pre-run all gallery images at startup (cache results) so the demo never waits.
- README includes a **3-minute demo script** in order: hardest image → reveal → recoverability → relight → candidates → ablation chart → lights off + webcam.

---

## 12. Build milestones (stop and verify at each)

| # | Hours | Deliverable | Verify |
|---|---|---|---|
| M1 | 0–2 | Repo skeleton, config, schemas, **mock API + mock SSE**, basic frontend wired to mock | Frontend works fully on mock data |
| M2 | 2–6 | `diagnose`, `classical`, `recoverability` + real `/enhance` (classical mode) | Real maps render in UI; < 100 ms |
| M3 | 4–8 | CIDNet (FP16, tiled, weight selection) + Zero-DCE++ + `/live` | No OOM at 1080p; live ≥ 20 FPS |
| M4 | 8–12 | `fusion`, gain-aware `denoise`, `guardrails`, hybrid mode | Visibly fewer halos/noise than DL-only |
| M5 | 10–14 | `tuner` + candidates + `report`/story + SSE real streaming | Full response matches schema; total ≤ 1.5 s |
| M6 | 14–16 | `eval.py`, `ablation.py`, `failure_mining.py` → **run overnight** | CSV/MD/charts generated |
| M7 | 16–20 | Gallery caching, `/benchmarks`, frontend completeness, tests | `pytest` passes; all tabs work |
| M8 | 20–21 | **Feature freeze.** Bug fixes only. Record backup demo video. | Demo script runs cleanly 2× |

---

## 13. Definition of done

- [ ] `uvicorn api.main:app` starts, warms up, and serves the UI at `/`.
- [ ] Any dark image → final result + all maps + story + metrics in ≤ 1.5 s on RTX 2050.
- [ ] SSE streaming works; mock mode works with no GPU/weights.
- [ ] Recoverability overlay and live relight work.
- [ ] Classical vs DL vs Hybrid comparison available in UI and in `results/benchmark.md`.
- [ ] Ablation table shows each component's contribution.
- [ ] Failure analysis doc generated with real failure cases.
- [ ] Never crashes on: tiny image, huge image, already-bright image (should change it minimally and say so), grayscale image, PNG with alpha.
- [ ] README lets a stranger run it in 10 minutes.

---

## 14. Do NOT

- Do not add more deep models (no Retinexformer, diffusion, GANs): one strong model plus the fusion logic is the design.
- Do not hard-code thresholds in code.
- Do not let the frontend compute anything the backend should own (except the WebGL relight).
- Do not use jargon in user-facing captions.
- Do not silently swallow errors; every fallback is reported in the response.
- Do not start a new milestone while the current one is broken.
