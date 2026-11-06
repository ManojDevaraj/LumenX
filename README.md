# LumenX

LumenX is a hackathon project for adaptive low-light image restoration. It combines illumination diagnosis, a Retinex/CLAHE classical path, optional Zero-DCE++ inference, per-pixel hybrid fusion, signal-aware denoising, safety checks, and a browser demo.

## Quick start

Requires Python 3.10 or later. From PowerShell at the repository root:

```powershell
python -m venv core\.venv
core\.venv\Scripts\python.exe -m pip install --upgrade pip
core\.venv\Scripts\python.exe -m pip install -e "core[vision,dev]" -e "server[test]"
core\.venv\Scripts\python.exe server\scripts\create_demo_gallery.py
$env:LUMENX_MOCK = "1"
core\.venv\Scripts\python.exe -m uvicorn lumenx_server.main:app --app-dir server --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Fake mode exercises uploads, progress, gallery, comparisons, and UI interactions without claiming real restoration quality.

## Enable the real engine

Install the optional Zero-DCE++ runtime and fetch its upstream checkpoint:

```powershell
core\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
core\.venv\Scripts\python.exe -m pip install -e "core[models]"
core\.venv\Scripts\python.exe -m lumenx_core.download_weights
Remove-Item Env:LUMENX_MOCK -ErrorAction SilentlyContinue
$env:LUMENX_ENGINE = "real"
core\.venv\Scripts\python.exe -m uvicorn lumenx_server.main:app --app-dir server --host 127.0.0.1 --port 8000
```

The first command installs the CPU-only PyTorch wheel. For CUDA inference, install the matching PyTorch build for the target machine before installing `core[models]`. The checkpoint is licensed CC BY-NC 4.0 by its authors and is downloaded locally; it is not bundled. Use it only for non-commercial research or demonstrations unless separately licensed. Without model weights, the classical and safe paths remain usable and the hybrid run reports the missing model. See [core model notices](core/LICENSE_NOTICES.md).

## Project map

- [`core/`](core/README.md): processing engine, CLI, metrics, adaptive search, model adapter, and training entry point.
- [`server/`](server/README.md): FastAPI service, streaming interface, local gallery, evaluation scripts, and runbook.
- [`server/docs/API.md`](server/docs/API.md): HTTP and WebSocket interfaces.
- [`server/docs/ARCHITECTURE.md`](server/docs/ARCHITECTURE.md): processing and runtime design.
- [`server/datasets/README.md`](server/datasets/README.md): dataset acquisition and expected layouts.
- [`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md): tests, browser check, LOL-v1 scores, fine-tuning, ablations, LIME/NPE unseen-scene evaluation, and measured limitations.

## Tests

Run the core contract, Zero-DCE++ checkpoint, and HTTP API checks from the repository root:

```powershell
core\.venv\Scripts\python.exe -m pytest core\tests server\tests -q
```

## Three-minute demo

1. Start in mock mode and open the local page. Run a synthetic gallery image to show the progressive diagnosis, candidate search, restoration stages, report, and before/after slider.
2. Switch among classical, Zero-DCE++, and hybrid modes. If weights are not installed, the page labels the disclosed fallback.
3. Use the live-camera tab if a camera is available, then open Benchmarks to view evaluation results after the dataset scripts have been run.

The synthetic gallery is for UI demonstration only. It is not benchmark data. Quality scores and latency must be measured on the target dataset and machine before making performance claims.

## Current release limits

- Zero-DCE++ is the supported learned model. CIDNet is unavailable in this build.
- Learned no-reference metrics require the optional `core[iqa]` extra and may download metric weights on first use.
- The optional LOL-v1 dataset, evaluation JSONL, and fine-tuned checkpoint stay in ignored local data folders; acquire them with the runbook if they are not present on your machine.
- CPU timings are hardware-specific. The live-camera feature still needs a camera-enabled browser check, and the GPU latency target has not been measured on a CUDA device.
