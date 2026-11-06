# LumenX validation report

Validation run: 6 November 2026. This report records local test, browser, held-out benchmark, fine-tuning, ablation, and unseen-scene results. It is specific to the machine and dataset subsets below; it is not a state-of-the-art claim.

## Test and application checks

- `pytest core/tests server/tests -q`: **5 passed**. Coverage includes the FakeEngine hybrid stream contract, real classical contract, Zero-DCE++ checkpoint loading and geometry preservation, health/gallery endpoints, a successful multipart enhancement request, and malformed-image errors.
- `python -m compileall -q core/lumenx_core server/lumenx_server server/eval core/scripts`: passed.
- Browser walkthrough: real engine, synthetic night-street gallery image, adaptive hybrid mode, 12 candidates, 8 displayed stages, and before/after comparison. The health endpoint reported the real engine ready with Zero-DCE++ available.
- That browser run processed a 768 x 480 image in **3.443 s total server time** (**3.305 s core**). It ran on CPU; this is not a GPU or 1080p measurement.
- The Benchmarks page was also opened against the local server after aggregate refresh; its published summary contains 264 runs across 20 groups.

Two defects found during these checks were fixed: the stream validator now accepts the contract's required `L_gray` visual key, and the report reads the configured `weak_area_medium_pct` field. The fake metric path also now returns an unavailable edge score for a constant edge map without raising NumPy warnings.

## Benchmark setup

- Dataset: authors' LOL-v1 archive; `eval15` contains 15 paired low/high test images. The separate `our485` training set was used for fine-tuning only. No `eval15` images or references were used for training.
- Unpaired generalization subset: 10 LIME and 8 NPE low-light images from the linked public collection. They have no aligned references in this run and were not used for fine-tuning.
- Input images are evaluated at their decoded size. The evaluator aligns reference geometry to the low-light input before PSNR/SSIM calculation.
- Machine/runtime: Windows 11, Python 3.13.5, 12 logical CPU threads, PyTorch 2.15.0+cpu, OpenCV 4.14.0, NumPy 2.5.4. CUDA was unavailable.
- The pretrained model was the local upstream Zero-DCE++ checkpoint (`sha256-ca8855b90df9`). The fine-tuned checkpoint is a separate ignored local file.
- All measured run sets completed without evaluator errors: 45 baseline runs, 30 fine-tuned runs, 135 ablation runs, and 54 unpaired runs. There are 12 saved failure-review previews.

Mean latency is the warmed engine-run time measured by `server/eval/evaluate.py`; it excludes HTTP upload, server startup, and model warmup. `Delta E` is the measured chroma change between the low-light input and the output, not a reference-image colour error. Noise sigma is an image-derived estimate. Guardrail failures count runs with at least one failed audit check.

## Main comparison

| Approach | PSNR mean +/- SD / median (dB) | SSIM mean +/- SD / median | Chroma Delta E vs input (lower) | Edge retention (higher) | Noise sigma out (lower) | Clipped pixels out (lower, %) | Latency mean / P95 (s) | Guardrail failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical | 14.855 +/- 3.552 / 14.053 | 0.5465 +/- 0.0762 / 0.5522 | 4.201 | 0.785 | 0.0219 | 0.0053 | 0.648 / 0.775 | 3/15 |
| Zero-DCE++ | 15.357 +/- 4.501 / 14.903 | 0.5726 +/- 0.1227 / 0.6073 | 9.700 | 0.894 | 0.0162 | 0.0104 | 0.614 / 0.724 | 15/15 |
| Hybrid | 14.772 +/- 3.779 / 13.934 | 0.5477 +/- 0.1030 / 0.5556 | 4.055 | 0.818 | 0.0194 | 0.0054 | 1.907 / 2.419 | 3/15 |
| Zero-DCE++ fine-tuned 1 epoch | 16.930 +/- 3.596 / 17.650 | 0.5437 +/- 0.1101 / 0.5206 | 8.402 | 0.727 | 0.0230 | 0.0569 | 0.467 / 0.556 | 12/15 |
| Hybrid with fine-tuned model | 14.866 +/- 3.725 / 13.498 | 0.5860 +/- 0.1090 / 0.6379 | 3.446 | 0.840 | 0.0155 | 0.0014 | 1.965 / 2.185 | 1/15 |

Classical and learned-only are faster than hybrid on this CPU. Learned-only had the best baseline PSNR and SSIM but failed the colour-shift guardrail on every held-out image. Hybrid gave up some paired PSNR/SSIM versus learned-only while substantially lowering the mean chroma change and guardrail failures. The fine-tuned checkpoint improved hybrid SSIM and lowered the hybrid guardrail failures to 1/15, but its learned-only path traded lower SSIM and edge retention for higher PSNR and still failed 12/15 safety audits. The local fine-tuned checkpoint is therefore **not** the runtime default.

## Unpaired unseen-scene check

These no-reference runs do not have PSNR or SSIM values. The listed measurements are means over each subset; latency is warmed engine time per image.

| Dataset | Method | Images | Chroma Delta E vs input | Edge retention | Noise sigma out | Clipped pixels out (%) | Latency mean (s) | Guardrail failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LIME | Classical | 10 | 5.275 | 0.820 | 0.0136 | 0.543 | 1.725 | 6/10 |
| LIME | Zero-DCE++ | 10 | 14.922 | 0.737 | 0.0167 | 2.400 | 1.267 | 10/10 |
| LIME | Hybrid | 10 | 4.652 | 0.871 | 0.0112 | 0.460 | 3.664 | 2/10 |
| NPE | Classical | 8 | 2.529 | 0.867 | 0.0100 | 1.157 | 0.787 | 3/8 |
| NPE | Zero-DCE++ | 8 | 8.538 | 0.783 | 0.0072 | 4.104 | 0.681 | 8/8 |
| NPE | Hybrid | 8 | 1.693 | 0.943 | 0.0061 | 2.330 | 2.141 | 1/8 |

All 54 unpaired runs completed without evaluator errors; 30 had at least one guardrail failure. The failed checks were colour shift (22), halo (20), overexposure (8), and detail loss (3); checks can overlap within a run. Hybrid had fewer guardrail failures and lower chroma change than learned-only on both subsets, with longer CPU latency. These are small subsets, and the measurements without references do not establish perceptual quality against a ground truth.

## Hybrid ablation results

Each row is the hybrid pipeline with one component disabled, evaluated on the same 15 test pairs. The default row is repeated to make these ablations directly comparable.

| Configuration | PSNR mean +/- SD / median | SSIM mean +/- SD / median | Chroma Delta E vs input (lower) | Edge retention (higher) | Noise sigma out (lower) | Mean latency (s) | Guardrail failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 14.772 +/- 3.779 / 13.934 | 0.5477 +/- 0.1030 / 0.5556 | 4.055 | 0.818 | 0.0194 | 1.787 | 3/15 |
| No tuner | 14.685 +/- 4.257 / 14.179 | 0.5602 +/- 0.1027 / 0.5788 | 4.426 | 0.851 | 0.0169 | 0.882 | 2/15 |
| No fusion | 14.429 +/- 3.340 / 13.348 | 0.5320 +/- 0.0863 / 0.5529 | 3.815 | 0.774 | 0.0226 | 1.444 | 3/15 |
| No classical path | 11.141 +/- 3.153 / 10.957 | 0.4722 +/- 0.1421 / 0.5243 | 5.561 | 0.938 | 0.0078 | 1.793 | 7/15 |
| No deep path | 14.383 +/- 3.346 / 13.181 | 0.5233 +/- 0.0954 / 0.5285 | 3.779 | 0.780 | 0.0213 | 1.580 | 3/15 |
| No gain-aware denoise | 14.796 +/- 3.796 / 13.939 | 0.5523 +/- 0.1002 / 0.5580 | 4.036 | 0.820 | 0.0198 | 1.804 | 2/15 |
| No recoverability control | 14.760 +/- 3.581 / 14.418 | 0.5616 +/- 0.1085 / 0.5686 | 4.092 | 0.812 | 0.0224 | 1.766 | 1/15 |
| No guardrail repair | 14.987 +/- 3.794 / 13.934 | 0.5400 +/- 0.0910 / 0.5556 | 4.524 | 0.805 | 0.0208 | 1.758 | 3/15 |
| No colour correction | 14.575 +/- 3.801 / 13.951 | 0.5494 +/- 0.1059 / 0.5557 | 3.844 | 0.825 | 0.0188 | 1.744 | 2/15 |

The ablations show tradeoffs rather than a uniform gain from every component. Disabling the classical path caused the largest paired-quality drop and raised hybrid guardrail failures from 3 to 7. Disabling fusion lowered SSIM and edge retention. Disabling the tuner almost halved mean runtime and slightly improved SSIM, while lowering PSNR modestly. These results support keeping the tuner configurable and treating its ranking metrics as an area for further work.

## Failure analysis and limits

- **Learned colour shifts:** all 15 baseline Zero-DCE++ outputs exceeded the configured chroma-shift threshold. In the four-image visual review set, `665.png` had Delta E 13.93 for learned-only and 8.50 for hybrid; the hybrid still failed on this very dark image. In `79.png`, learned-only triggered both colour-shift and halo checks, while hybrid passed the audit.
- **Worst paired samples:** `55.png` had the lowest learned-only PSNR (8.26 dB); `23.png` had the lowest hybrid PSNR (9.35 dB); `79.png` had the lowest classical PSNR (9.35 dB). Mean hybrid latency's slowest sample was `1.png` at 2.44 s; the evaluator P95 was 2.42 s.
- **Extremely dark inputs:** the model may not recover visible detail when the captured signal is weak. For `665.png`, all modes remained far from the bright reference; conservative recovery cannot reconstruct information absent from the input.
- **The one-epoch fine-tune is mixed:** learned-only clipping rose to 0.0569% and edge retention fell to 0.727, despite the PSNR increase. More training, alternative losses, and a larger unseen set are needed before promoting it.
- **Unpaired samples:** hybrid still tripped at least one guardrail on 2/10 LIME and 1/8 NPE images. Broader unpaired data and human review are needed; the LIME/NPE subsets do not have references in this run.
- **Learned no-reference IQA unavailable:** NIQE, BRISQUE, CLIP-IQA, MUSIQ, and LPIPS were not installed, so those values are null. The benchmark uses paired PSNR/SSIM and the implemented chroma, edge, noise, clipping, confidence, and guardrail measurements.
- **Unmeasured paths:** no CUDA device was available. GPU latency, 1080p latency targets, camera permission/live capture, LOL-v2, MIT-Adobe FiveK, and other unpaired datasets were not validated in this run.

Representative failure previews and per-image metrics are local under `server/results/failure_audit/`. Raw benchmark JSONL, the LOL archive, unpaired source images, and fine-tuned checkpoint remain ignored by Git; the aggregate CSV, JSON, and chart are included so the hosted Benchmarks panel has results. Reproduction commands and split guidance are in [the dataset runbook](../server/datasets/README.md).
