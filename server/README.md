# LumenX Server

This folder contains the local FastAPI application, browser interface, demo gallery, and dataset evaluation tools. The default configuration uses `FakeEngine` for a working UI preview. The real CPU/GPU pipeline is enabled with `LUMENX_ENGINE=real`.

## Install and launch

From the repository root in PowerShell:

```powershell
core\.venv\Scripts\python.exe -m pip install -e "core[vision]" -e server
core\.venv\Scripts\python.exe server\scripts\create_demo_gallery.py
$env:LUMENX_MOCK = "1"
core\.venv\Scripts\python.exe -m uvicorn lumenx_server.main:app --app-dir server --host 127.0.0.1 --port 8000
```

For a real engine, install the optional model runtime, download weights, clear mock mode, and set `LUMENX_ENGINE=real` as described in the root README. By default the app runs locally at port 8000. Set `LUMENX_SERVER_CONFIG` to a YAML file to override upload limits, paths, queue depth, and retention. The engine's vision and model thresholds are configured separately in `core/lumenx_core/config/default.yaml`.

## What the app includes

- Drag/drop upload with orientation correction, alpha preservation, 16-bit grayscale conversion, upload-size checks, and disclosed downscaling.
- Progressive SSE events with maps, candidate thumbnails, diagnostics, final result, and run retrieval.
- Before/after comparison, map overlays, a WebGL relight preview, candidate cards, and quality/timing readouts.
- Classical/model/hybrid comparison, one bounded live-camera session, synthetic local gallery, and benchmark summary view.
- Bounded single-worker processing lane, one active model instance, atomic run artifacts, and JSON error responses.

Synthetic gallery scenes are illustrative UI assets only. They are not benchmark or training data.

## Evaluation workflow

Download the datasets from the source links in [`datasets/README.md`](datasets/README.md). Do not place third-party data into version control. The evaluator discovers common `Low/High` and `Low/Normal` layouts or accepts explicit folders.

Paired LOL example:

```powershell
core\.venv\Scripts\python.exe server\eval\evaluate.py --dataset LOL-v1 --root D:\datasets\LOL --paired --modes classical,dl,hybrid --ablations no_tuner,no_fusion,no_classical,no_dl,no_gain_denoise,no_recoverability,no_guardrails,no_color_correction
core\.venv\Scripts\python.exe server\eval\aggregate.py
core\.venv\Scripts\python.exe server\eval\mine_failures.py
```

For a dataset with no references, omit `--paired`. Use `--input-dir` and `--reference-dir` if automatic discovery does not match its layout. The evaluator writes resumable per-image rows, progress, paired PSNR/SSIM/LPIPS where available, no-reference IQA where available, clipping/colour/edge/noise metrics, confidence-class errors, guardrail failures, and timings under `server/results/`. It validates each core run before recording it. Re-run the same command to resume; existing keys are skipped. Add `--save-images` to retain final outputs under `server/results/previews/` for manual review. Run the aggregate command again after each evaluation batch.

Measure actual API latency separately with a running server:

```powershell
core\.venv\Scripts\python.exe server\eval\benchmark_server.py --images D:\datasets\LOL\eval\Low --mode hybrid --warmup 1
```

The first request is excluded from timing by default. Measurements include image upload, server processing, artifact publication, and response serialization. Record device, Python version, image dimensions, warm-up policy, and checkpoint hash with any reported latency.

## Dataset path and split discipline

Keep training, validation, and test paths separate. Use the published test split unchanged for final reporting; do not fine-tune on benchmark references. The training script expects unpaired low-light training images and is not invoked by the evaluator. For previously unseen sets such as DICM, LIME, MEF, NPE, and VV, report no-reference and custom measurements only. Record input filenames and any resize policy.

## Operational notes

- `GET /health` reports service readiness, engine readiness, loaded model, and lane state.
- `GET /config` returns the public-safe configuration; absolute paths and secrets are omitted.
- `GET /runs/{id}` returns the complete result or current run status.
- `server/runs/` and `server/results/` are local, ignored runtime data. Use a local disk with enough space for uploaded originals and intermediate images.
- Browser camera access requires localhost or HTTPS and explicit browser permission.

See [`docs/API.md`](docs/API.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md).
