# Hackathon Technical Solution Report (Design Document)

_Also available as PDF: [`Hackathon_Technical_Solution_Report_LumenX.pdf`](Hackathon_Technical_Solution_Report_LumenX.pdf). Metrics: [`eval_P1_Blackbox.md`](../eval_P1_Blackbox.md). Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)._

| Field | Details |
|---|---|
| Problem ID | ProbID - P1 |
| Problem Title | Robust Low-Light Image Restoration and Enhancement |
| Team Name | Blackbox |
| Team Members | Manoj D — 1RUA24CSE0242; Mohit D — 1RUA24CSE0280; Nikhil Shridhara — 1RUA24CSE0283 |
| GitHub Repository | https://github.com/ManojDevaraj/LumenX — branch main, commit b096195 (engine tag core-v0.4.3) |
| Live Demo / Deployment | Local FastAPI web app (uvicorn, RTX 2050 laptop); Video demo: [link to be added] |

## 1. Problem Understanding & Solution Strategy

- **Core Technical Challenge:** Low-light photos combine several failure modes in one frame: severe underexposure (most pixels below 5% brightness), sensor noise that is amplified by the same factor as the brightening, colour casts from mixed street/indoor lighting, and non-uniform illumination where lamps and windows are already bright. A single global enhancement cannot satisfy all regions: in our measurements the strongest deep model alone (HVI-CIDNet) reaches 19.65 dB PSNR on LOL-v1 but blows out 11.2% of pixels around light sources. Constraints: a single 4 GB RTX 2050 laptop GPU, free/open-source tools only, a 24-hour build, and per-image processing in a few seconds for an interactive demo.
- **Proposed Methodology:** LumenX is a deep-anchored hybrid: a classical diagnosis stage measures illumination, native-resolution noise, signal-to-noise confidence, light-source hotspots and colour cast; HVI-CIDNet then produces the base enhancement, and classical, edge-preserving corrections are applied only where the model fails (clipped highlights, halo rings, weak-signal texture). The correction strengths are chosen per image by a 12-candidate, quality-aware search (CLIP-IQA plus clipping, colour-shift and edge checks) instead of fixed parameters, because static pipelines either under-brighten dark regions or over-expose bright ones.

## 2. Technical Architecture & Implementation

> Note: Do not paste raw code files in this document. Reference repository modules and explain system design.

### 2.1 Classical Algorithmic Baseline

- **Algorithms Utilized:** LIME-style Retinex decomposition (max-RGB illumination map refined with a guided filter, reflectance = image / illumination), highlight-protected illumination gamma, CLAHE on the Lab L-channel, partial gray-world colour correction in Lab, and gain-aware denoising that blends three guided-filter strengths per pixel. Diagnostics include a native-resolution noise estimate (median absolute deviation over up to 100 flat 32×32 patches), an SNR-based recoverability map (recovered / estimated / unrecoverable), hotspot detection and clipping statistics (module `core/lumenx_core/`: diagnose, recoverability, classical, denoise).
- **Adaptive Selection Logic:** All thresholds live in `core/lumenx_core/config/default.yaml`; per-image values are derived from the diagnosis. Gamma and CLAHE clip limits follow the measured brightness percentiles and severity class; denoise strength is proportional to local gain × local noise σ and is raised in low-confidence regions; colour-correction strength is proportional to the measured Lab cast and capped so that warm lamps keep their mood; highlight protection is driven by hotspots (illumination > 0.85). Pure-black regions with zero local variance are kept black.

### 2.2 Deep Learning Pipeline

- **Model Architecture:** HVI-CIDNet (CVPR 2025, 1.88 M parameters, about 7.6 GFLOPs), which separates intensity from chromaticity in a trainable HVI colour space and so reduces the colour bias and brightness artifacts of RGB-space models. The LOL-v2-real checkpoint is used for mild/moderate/severe darkness and the SID (Sony-Total-Dark) checkpoint for extreme darkness, both run in FP16 with 512-px tiles, 32-px overlap and cosine-feathered blending (peak GPU allocation 407 MiB at 1080p and 12 MP). Zero-DCE++ is used for the live webcam path and as an automatic fallback. CIDNet was chosen over Zero-DCE++ because it scored 19.65 dB vs 15.14 dB on LOL-v1.
- **Loss Functions & Training:** CIDNet uses the authors' pretrained weights (trained on LOL-v2 / SID with combined pixel, structural/edge and perceptual losses in HVI and sRGB space). We fine-tuned Zero-DCE++ on the 485 LOL-v1 `our485/low` training images (10 epochs on the RTX 2050, no references) with its no-reference composite loss: spatial consistency, exposure control, colour constancy and illumination smoothness (loss 2.84 → 2.27). The held-out LOL-v1 eval15 split was never used for training or calibration. Fine-tuning lowered eval15 PSNR (13.43 dB vs 15.14 dB pretrained), so the pretrained weights remain the default and the fine-tuned checkpoint is kept as a documented experiment.

### 2.3 Hybrid Integration & Adaptive Mechanism

- **Fusion Strategy:** The classical priors steer where the network's output is trusted rather than averaging two images. The illumination map and hotspot boxes define a highlight-repair mask (pixels the network clipped at ≥ 0.98 plus overshoot rings around light sources), smoothed with a guided filter on the input luminance so that it follows real edges. Inside the mask the output blends toward a hue-preserving, soft-knee compressed version of the CIDNet result; outside it, the output equals CIDNet exactly. The recoverability map limits texture sharpening in near-black, low-confidence regions so that detail is not invented where too little light was recorded.
- **Decision Logic:** Brightening multiplies noise by the local gain, so the system computes a gain map (output / input luminance) and applies denoising in proportion to gain × local noise, restricted to repaired and low-confidence regions because CIDNet already denoises the rest. A 12-candidate search on a 256-px proxy (candidate 0 = CIDNet + highlight repair only) varies repair strength, denoise strength and colour correction; candidates with more than 1% clipped pixels or more than 15% darker than CIDNet are rejected, and the remaining ones are ranked by CLIP-IQA plus clipping, chroma ΔE and edge-retention terms, with ties going to the least-correction candidate. Guardrails then audit overexposure, colour shift, detail loss and halos, and every decision is reported in plain English.

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks

| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual Metric (LPIPS / NIQE) ↓ | Latency (ms / frame) |
|---|---|---|---|---|
| Classical Baseline | 13.93 | 0.711 | 0.328 (LPIPS) | 537 ms |
| Deep Learning Only | 19.65 | 0.812 | 0.195 (LPIPS) | 646 ms |
| Proposed Hybrid System | 19.79 | 0.816 | 0.196 (LPIPS) | 1140 ms |

- **Analysis:** On the held-out LOL-v1 eval15 set (15 paired images, 600×400, RTX 2050), the hybrid slightly exceeds the deep model on structural fidelity (PSNR +0.14 dB, SSIM +0.004) with equal perceptual quality (LPIPS 0.196 vs 0.195), while cutting blown-out pixels from 11.24% to 0.26% (43× fewer), lowering chroma shift (ΔE 3.18 vs 3.36) and improving edge retention (0.811 vs 0.792). The classical baseline is the safest (0.007% clipped) but 5.7 dB less accurate. The cost is speed: the quality-aware candidate search and guardrail checks add about 0.5 s per image over the deep model alone (1140 vs 646 ms; latency includes request-time quality metrics). NIQE was not computed by the installed toolkit; the no-reference scores recorded instead are CLIP-IQA (0.501 hybrid vs 0.506 DL) and MUSIQ (64.8 vs 65.1).

### 3.2 Visual Outputs & Intermediate Representations

Figures are produced by the repository: every enhancement run writes its stage images and maps to `server/runs/<run_id>/` (shown in the web app's pipeline view), and the evaluation saves outputs to `server/results/previews/` with `--save-images`. They demonstrate:

1. **Extreme low-light:** LOL-v1 eval15 inputs (for example `1.png`, `146.png`) next to the classical, CIDNet and hybrid outputs, with the plain-English diagnosis of each image.
2. **Non-uniform / backlit scenes:** On `1.png` the deep model alone clips 12.8% of pixels around the light sources; the hybrid reduces this to 0.38% at the same brightness (mean luminance 0.476 vs 0.472) and the same PSNR (23.42 vs 23.48 dB).
3. **Intermediate states:** Illumination map, reflectance map, SNR / recoverability overlay (green recovered, amber estimated, red unrecoverable), fusion-weight map, gain map, denoise-strength map, a 128×128 illumination height-map and the 12 tuner candidates with their scores and rejection reasons.

### 3.3 Ablation Study

- Each module was disabled individually on LOL-v1 eval15 in the final configuration (full system: 19.79 dB PSNR / 0.816 SSIM / 0.196 LPIPS / 0.26% clipped / 1332 ms):
- **Deep model removed (no_dl):** 7.76 dB / 0.189 / 0.634 LPIPS: the network is the backbone of reconstruction quality (−12.0 dB).
- **Highlight-repair fusion removed (no_fusion, = DL only):** 19.65 dB / 0.812 with 11.25% clipped pixels: the classical-prior-guided repair cuts blow-outs 43× and also improves PSNR (+0.14 dB), edge retention (0.792 → 0.811) and chroma shift (ΔE 3.38 → 3.18).
- **Gain-aware denoise removed:** 19.80 dB / 0.816 / 0.12% clipped, edge retention 0.800 vs 0.811: a small effect on LOL because denoising is confined to repaired regions and CIDNet already denoises.
- **Adaptive tuner removed (fixed parameters):** identical image quality; on all 15 images the quality-aware search kept the minimal-correction candidate, so it acted as a safety net (rejecting candidates with >1% clipping or darker exposure) at a cost of about 0.5 s per image (839 vs 1332 ms).
- **Classical pixel path, colour correction, recoverability, guardrails removed:** no measurable change (recoverability: 0.258% vs 0.260% clipped). The classical stage contributes through its priors (illumination map, hotspots, noise and SNR maps that define where to repair) rather than through blended pixels; LOL-v1 has no strong colour cast or near-black regions, and guardrails are audit-only by design.
- **Design decision from an earlier ablation:** an automatic guardrail re-repair pass lowered SSIM from 0.816 to 0.721 and LPIPS from 0.196 to 0.247 by over-smoothing, so it was removed and guardrails now only audit and report.

## 4. Failure Modes & Limitations

- **Observed Failure Case:** Three failures were observed. (1) The deep model alone saturates light sources: 11.24% of pixels clipped on eval15 and 12.8% on `1.png`. (2) Our first hybrid averaged CIDNet with the classical path and was too dark: 12.21 dB PSNR, worse than either component. (3) In the final system the guardrail audit still flags detail loss on 5 of 15 eval15 images (493, 55, 748, 778, 780) and a halo ring around a lamp on `748.png`; the deep model alone is flagged on 6 of 15. Fine-tuned Zero-DCE++ also regressed (13.43 vs 15.14 dB).
- **Root Cause & Mitigation:** (1) CIDNet's training targets reward overall brightness, with no explicit penalty on saturating already-bright regions; this is fixed by the masked soft-knee highlight repair. (2) Averaging a strong output with a darker one pulls every pixel toward the darker estimate, and the recoverability and tuner caps compounded it; fixed by anchoring on CIDNet and correcting only where it fails (12.21 → 19.79 dB). (3) Residual detail loss comes from very dark, noisy regions where the network smooths fine texture, and the halo from compressing local contrast at a sharp lamp boundary; mitigation: an edge-aware (smaller-radius) repair mask at high-gradient lamp edges, detail transfer from the classical reflectance inside repaired areas, and a highlight-aware loss term if the model is retrained. Zero-DCE++ fine-tuning regressed because its no-reference exposure target does not match LOL's reference brightness; paired fine-tuning or a tuned exposure target is the next step.
- **Ethics & Responsible Use:** Enhanced images are reconstructions, not evidence: the network can smooth or alter fine detail, and regions with too little recorded light are flagged rather than invented, so LumenX should not be used for forensic or identification decisions. Uploads stay on the server in a bounded run history and are not sent to third parties; Zero-DCE++ weights are academic/non-commercial and datasets are not redistributed.

## 5. Deployment & Quickstart Guide

### Quickstart Commands

```bash
# 1. Clone the repository and install dependencies
git clone https://github.com/ManojDevaraj/LumenX.git
cd LumenX
python -m venv .venv
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e "core[models,iqa]" -e server
python core/scripts/download_weights.py

# 2. Run inference on a sample image
python -m lumenx_core.cli enhance ./samples/night.png ./results/ --mode hybrid

# 3. Launch the web app (http://localhost:8000)
uvicorn lumenx_server.main:app --app-dir server --port 8000

# 4. Reproduce the benchmarks and ablations (set dataset paths at the top of the script)
powershell -ExecutionPolicy Bypass -File server/eval/run_final_eval.ps1
```

### Deployment Format

- FastAPI web application served by uvicorn: REST `POST /enhance`, server-sent-event streaming of every pipeline stage (`POST /enhance/stream`), a WebSocket live-webcam mode (Zero-DCE++) and `GET /benchmarks`, with a vanilla HTML/JS frontend including WebGL relighting. The engine is an installable Python package (`lumenx-core`) running PyTorch 2.6 FP16 on CUDA (tested on an RTX 2050, 4 GB) with automatic CPU fallback; a CLI (`python -m lumenx_core.cli`) and resumable evaluation scripts are included. A mock engine lets the frontend run without a GPU or model weights.

### Student Submission Checklist

- [ ] Working GitHub repository link included with an updated `README.md` .
- [x] No raw code pasted into this report.
- [x] All table metrics match test runs reproducible via the repository.
- [x] All team member details accurately recorded.
