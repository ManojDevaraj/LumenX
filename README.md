---
title: LumenX Low-Light Restoration
emoji: 🌃
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
---

# LumenX: Adaptive Low-Light Image Restoration

**Team Blackbox** · Problem **P1: Robust Low-Light Image Restoration and Enhancement**
Manoj D (1RUA24CSE0242) · Mohit D (1RUA24CSE0280) · Nikhil Shridhara (1RUA24CSE0283)

LumenX restores dark, noisy, unevenly lit photos. A classical diagnosis stage measures illumination, noise, signal confidence, light sources and colour cast. **HVI-CIDNet** produces the base enhancement, and classical, edge-preserving corrections repair only what the network gets wrong, mainly blown-out highlights around lamps and windows. Every stage is streamed to the browser with plain-English explanations, maps and metrics.

| Document | What it contains |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture diagram and module map |
| [`docs/DESIGN_DOCUMENT.md`](docs/DESIGN_DOCUMENT.md) ([PDF](docs/Hackathon_Technical_Solution_Report_LumenX.pdf)) | Design document / technical solution report |
| [`eval_P1_Blackbox.md`](eval_P1_Blackbox.md) | Comparative evaluation, ablation study, failure-case analysis |
| [`server/docs/API.md`](server/docs/API.md) | HTTP / streaming / WebSocket API |
| [`CONTRACT.md`](CONTRACT.md) | Engine ↔ server interface contract |

## Results (LOL-v1 eval15, held out, RTX 2050)

| Approach | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Blown-out px ↓ | Latency |
|---|---|---|---|---|---|
| Classical only | 13.93 | 0.711 | 0.328 | 0.007% | 537 ms |
| Deep learning only (HVI-CIDNet) | 19.65 | 0.812 | 0.195 | 11.25% | 646 ms |
| **LumenX hybrid** | **19.79** | **0.816** | 0.196 | **0.26%** | 1140 ms |

The hybrid matches the deep model's quality while cutting blown-out highlights 43×. Full tables, ablation and failure cases: [`eval_P1_Blackbox.md`](eval_P1_Blackbox.md).

## Run it

### Option A: one command (Docker, CPU)
```bash
docker build -t lumenx . && docker run -p 7860:7860 lumenx
```
Open http://localhost:7860. The image installs everything and downloads the model weights at build time. The same Dockerfile deploys as a Hugging Face Space (Docker SDK).

### Option B: local Python (uses the NVIDIA GPU automatically if present)
Python 3.11, from the repository root (Windows PowerShell shown; use `source .venv/bin/activate` on Linux/macOS):
```powershell
python -m venv .venv
.venv\Scripts\activate
# GPU: CUDA 12.4 build. For CPU only, use --index-url https://download.pytorch.org/whl/cpu
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e "core[models,iqa]" -e server
pip uninstall -y opencv-python opencv-python-headless
pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86
python core/scripts/download_weights.py
$env:LUMENX_ENGINE = "real"
python -m uvicorn lumenx_server.main:app --app-dir server --port 8000
```
Open http://localhost:8000. Check `GET /health` → `"device": "cuda:0"` (GPU) or `"cpu"`.

The two OpenCV lines matter: `pyiqa` installs plain `opencv-python`, which hides the `opencv-contrib` guided filter that LumenX needs.

**No GPU or weights?** Set `$env:LUMENX_MOCK = "1"` to run the UI against the built-in mock engine (it's clearly labelled MOCK).

### Command line
```powershell
python -m lumenx_core.cli enhance path\to\dark.png out_dir --mode hybrid   # or classical | dl | zerodce
```

## Reproduce the evaluation
1. Download LOL-v1 (and optionally LOL-v2, LIME, DICM, or your own unseen images). Datasets are not stored in this repo.
2. Set the dataset paths at the top of `server/eval/run_final_eval.ps1`.
3. `powershell -ExecutionPolicy Bypass -File server/eval/run_final_eval.ps1`

This produces `server/results/benchmark_summary.{csv,json,svg}` and `failure_review.csv`, which the web UI serves through `GET /benchmarks`. Runs are resumable. CPU runs give the same quality metrics, just with slower latency.

## Repository layout
- `core/`: `lumenx-core`, the processing engine (diagnosis, classical CV, HVI-CIDNet, Zero-DCE++, fusion, tuner, guardrails, metrics). Config: `core/lumenx_core/config/default.yaml`. Tests: `pytest core/tests -q`.
- `server/`: `lumenx-server`, the FastAPI app, web UI (`server/lumenx_server/frontend`), evaluation scripts (`server/eval`), results (`server/results`).
- `docs/`: architecture, design document, reference brief.

## Limitations
- The adaptive search kept the minimal-correction candidate on all LOL-v1 test images: it acts as a safety net and costs about 0.5 s per image.
- Residual detail loss in very dark, noisy regions (flagged on 5/15 test images) and a halo near one sharp lamp edge.
- No-reference NIQE was unavailable in the installed toolkit; CLIP-IQA and MUSIQ are reported instead.
- Fine-tuning Zero-DCE++ on LOL `our485` lowered held-out PSNR (13.43 vs 15.14 dB), so pretrained weights are used.
- On CPU-only hosts, hybrid processing takes several seconds per image.

## Ethics and responsible use
- Outputs are enhancements, not evidence: the model may smooth or alter fine detail, and regions with too little recorded light are flagged rather than invented. Don't use LumenX for forensic, surveillance or identification decisions.
- Uploaded images stay on the server in a bounded run history and are never sent to third-party services. No secrets or credentials are stored in this repository.
- Licences: HVI-CIDNet per its upstream repository; Zero-DCE++ weights are academic/non-commercial. Datasets are used under their research terms and are not redistributed.

## Credits
HVI-CIDNet (Yan et al.), Zero-DCE++ (Li et al.), pyiqa (Chen et al.). Zero-DCE++ fine-tuning script originally by Mohit D.
