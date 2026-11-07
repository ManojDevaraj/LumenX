1RUA24CSE0242,1RUA24CSE0280,1RUA24CSE0283

# Evaluation: P1 Robust Low-Light Image Restoration and Enhancement

Team **Black Rock** · Nikhil Sridhara (1RUA24CSE0283), Manoj D (1RUA24CSE0242), Mohit D (1RUA24CSE0280)
Repository: https://github.com/ManojDevaraj/LumenX · engine tag `core-v0.4.3` · results commit `b096195`

## Setup

- **Test set:** LOL-v1 `eval15`, 15 paired low/normal-light images (600×400), held out: never used for training, calibration or threshold tuning (previously unseen).
- **Training data:** LOL-v1 `our485/low` (485 images) used only for the Zero-DCE++ fine-tuning experiment.
- **Hardware:** NVIDIA RTX 2050 laptop GPU (4 GB), PyTorch 2.6 FP16. Latency is mean wall time per image including request-time quality metrics.
- **Reproduce:** `server/eval/run_final_eval.ps1` (or `server/eval/evaluate.py` + `aggregate.py`); raw summary in `server/results/benchmark_summary.csv`.

## 1. Comparative results (metrics in the order of the problem statement)

| Approach | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | No-reference: CLIP-IQA ↑ / MUSIQ ↑ | Latency (ms/image) ↓ | Noise suppression: residual noise σ ↓ | Edge preservation: edge retention ↑ |
|---|---|---|---|---|---|---|---|
| Classical only | 13.93 | 0.711 | 0.328 | 0.479 / 59.13 | 537 | 0.0032 | 0.928 |
| Deep learning only (HVI-CIDNet) | 19.65 | 0.812 | **0.195** | 0.507 / **65.11** | 646 | 0.0050 | 0.792 |
| **Hybrid (LumenX)** | **19.79** | **0.816** | 0.196 | 0.503 / 64.84 | 1140 | 0.0043 | 0.810 |
| Zero-DCE++ (pretrained, extra baseline) | 15.14 | 0.668 | 0.342 | 0.607 / 56.42 | 508 | 0.0154 | 0.919 |

NIQE was not available in the installed toolkit; CLIP-IQA and MUSIQ are the no-reference quality metrics used instead.

### Overexposure, colour shift and detail loss (advanced requirement 4)

| Approach | Blown-out pixels (%) ↓ | Chroma shift ΔE ↓ | Images flagged by guardrail audit (of 15) ↓ |
|---|---|---|---|
| Classical only | 0.007 | 3.96 | 0 |
| Deep learning only | 11.25 | 3.38 | 6 |
| **Hybrid (LumenX)** | **0.26** | **3.18** | 5 |
| Zero-DCE++ | 0.009 | 7.73 | 1 |

**Summary:** the hybrid matches or slightly exceeds the deep model on fidelity (PSNR +0.14 dB, SSIM +0.004, LPIPS equal) while reducing blown-out pixels 43× (11.25% → 0.26%), lowering chroma shift and improving edge retention. The cost is about 0.5 s extra per image for the adaptive search and audit.

## 2. Ablation study (hybrid, one component disabled at a time, LOL-v1 eval15)

| Configuration | PSNR | SSIM | LPIPS | Blown-out % | Edge retention | Latency (ms) |
|---|---|---|---|---|---|---|
| Full hybrid | 19.79 | 0.816 | 0.196 | 0.26 | 0.811 | 1332 |
| − deep model (`no_dl`) | 7.76 | 0.189 | 0.634 | 0.006 | 0.972 | 1151 |
| − highlight-repair fusion (`no_fusion`) | 19.65 | 0.812 | 0.195 | 11.25 | 0.792 | 1271 |
| − gain-aware denoise (`no_gain_denoise`) | 19.80 | 0.816 | 0.197 | 0.12 | 0.800 | 1346 |
| − adaptive tuner (`no_tuner`) | 19.79 | 0.816 | 0.196 | 0.26 | 0.811 | 839 |
| − classical pixel path (`no_classical`) | 19.79 | 0.816 | 0.196 | 0.26 | 0.811 | 1444 |
| − recoverability (`no_recoverability`) | 19.79 | 0.816 | 0.196 | 0.26 | 0.811 | 1398 |
| − colour correction (`no_color_correction`) | 19.79 | 0.816 | 0.196 | 0.26 | 0.811 | 1371 |
| − guardrails (`no_guardrails`) | 19.79 | 0.816 | 0.196 | 0.26 | 0.811 | 1401 |
| Earlier design: guardrail auto re-repair enabled | 19.57 | 0.721 | 0.247 | 0.09 | 0.930 | n/a |

Findings:
- The **deep model** is the backbone of reconstruction quality (−12 dB without it).
- **Classical-prior-guided highlight repair** removes 97.7% of blow-outs and improves PSNR, edge retention and colour.
- **Gain-aware denoise** has a small effect on LOL because it is confined to repaired regions and CIDNet already denoises.
- The **adaptive tuner** kept the minimal-correction candidate on all 15 images. It acts as a safety net (rejecting candidates with more than 1% clipping or darker exposure) at a cost of about 0.5 s per image.
- **Recoverability and colour correction** do not trigger on LOL-v1 (no near-black regions or strong casts). The classical stage contributes through priors (illumination, hotspots, noise/SNR) rather than blended pixels.
- **Guardrails** are audit-only. An automatic re-repair pass was removed because it over-smoothed (SSIM 0.816 → 0.721).

## 3. Fine-tuning experiment (Zero-DCE++)

| Checkpoint | PSNR | SSIM | LPIPS | Blown-out % | Chroma ΔE | Flagged |
|---|---|---|---|---|---|---|
| Pretrained (publisher) | 15.14 | 0.668 | 0.342 | 0.009 | 7.73 | 1/15 |
| Fine-tuned on `our485/low` (10 epochs, no-reference losses) | 13.43 | n/a | n/a | 0.016 | 5.03 | 9/15 |

Fine-tuning reduced training loss (2.84 → 2.27) but lowered held-out PSNR, so the pretrained weights are the default. The fine-tuned checkpoint is kept selectable for transparency.

## 4. Failure-case analysis

| Image | Mode | Audit flags | Cause | Mitigation |
|---|---|---|---|---|
| eval15 (aggregate) | DL only | 11.25% blown-out pixels | CIDNet brightens without a penalty on saturating bright regions | Masked soft-knee highlight repair (adopted: 0.26%) |
| first hybrid version | hybrid | too dark (12.21 dB) | averaging CIDNet with a darker classical image, plus exposure caps | Deep-anchored fusion: correct only where CIDNet fails (adopted: 19.79 dB) |
| 493, 55, 778, 780 | hybrid | detail loss | network smoothing of fine texture in very dark, noisy regions (DL-only is flagged on 6/15) | Detail transfer from classical reflectance inside low-confidence regions |
| 748 | hybrid | detail loss, halo | contrast compression at a sharp lamp boundary | Edge-aware, smaller-radius repair mask at high-gradient lamp edges |
| eval15 | Zero-DCE++ fine-tuned | lower PSNR | no-reference exposure target mismatched to reference brightness | Paired fine-tuning or a tuned exposure target |

The per-image list is in `server/results/failure_review.csv`.
