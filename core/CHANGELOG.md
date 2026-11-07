# Changelog

## 0.4.1 — 2026-11-07

- Fixed hybrid tuner proxy validity mask shape for HD/FHD inputs; request noise
  metrics use native-resolution patch estimates.
- Added regression coverage for 720p/1080p non-black flat/noisy hybrid on CPU and GPU.
- Ported GPU Zero-DCE++ no-reference training, with a selectable fine-tuned weight.
- Training credit: Mohith-D05 (PR #1), ported from the read-only snapshot.
- GPU fine-tuning: 10 epochs total on the 485 images in LOL-v1 `our485/low`,
  initialized from the publisher checkpoint. The first 3 epochs were resumed from
  the checkpoint after the command session ended; subsequent seven epochs completed.
- Fine-tuned SHA256: `90d375b5b00a2d413d0baf0e382d9dbf8128ba9e8ff30b84b797965a8a5ea65f`.
- CPU suite: 359 passed, 13 GPU cases deselected; Ruff clean. User-confirmed prior
  GPU run: 39 passed across hybrid, metrics and model tests.

## 0.4.0 — 2026-11-06

- Fixed native noise support accounting and checkerboard flatness detection; synthetic
  flat and textured noise is measured within 20%, and denoise cannot increase it.
- Added a deterministic 12-candidate tuner, candidate events/thumbnails, fixed-score
  feasibility ranking, request-time metrics and learned CLIP-IQA integration.
- Tuner recipe is `clipiqa+custom` when the optional pinned `core[iqa]` PyIQA
  dependency is present; it records `custom_no_clipiqa` with an unavailable reason otherwise.
- Latency is reported, not a release gate. Confidence-validation and live/webcam
  tuning are deferred by approved scope.

## 0.3.0 — 2026-11-06

- GPU release gates verified on the RTX 2050: all four mode contracts, both CIDNet
  checkpoints in FP16 with no fallback, and hybrid at 1080p and 12 MP without OOM.
- Verified cosine tiling, fusion and the bounded single repair reusing cached deep
  output. Resident peak VRAM was 407 MiB.
- Latency is retained as a reported metric rather than a v0.3 release gate.

## 0.3.0.dev0 — superseded by 0.3.0

- Official CIDNet/Zero-DCE++ architectures with provenance/licence notices.
- SHA256 manifest/downloader and verified official Zero-DCE++ checkpoint.
- Cosine tiling, stride padding, native-size output and cancellation.
- Once-loaded registry with measured resident/swap preflight and original CPU
  weight snapshots. Finite-output checking allows one FP32 retry; precision-
  sensitive HVI/attention normalization use FP32 without changing layers/state names.
- Direct baselines, confidence-bounded fusion, hybrid denoise and a single repair
  using cached deep output. Until v0.4, hybrid requires explicit no_tuner.
- 349 CPU tests pass; nine real-GPU gates explicitly deselected by default.
  --run-gpu opts in; missing prerequisites FAIL rather than silently skip.
- Torch/einops and both CIDNet files remain absent in the visible setup. No
  core-v0.3.0 tag, trained-output or GPU performance claim is made.
- Separate top-five CPU function profile appended to bench_results.md; the
  required final hybrid v0.4 profile/1100 ms p95 budget remain unchanged.

## 0.2.0 — 2026-11-06, temporary performance exception approved

- Real CPU classical Engine with contract-valid ordered stages and report.
- Native-patch MAD noise with confidence, clipping-aware recovery and full-size maps.
- Retinex, partial Lab CLAHE/cast correction, highlights and bright-input identity.
- Three-strength guided denoising with actual gain caps and alpha isolation.
- Real custom metrics, CIEDE2000, audit guardrails and explicit learned-IQA nulls.
- Frozen validated config, deterministic CPU behavior, cancellation and cleanup.
- Developer CLI, benchmark, reference colour tests and real edge-case coverage.
- 325 CPU tests pass; Ruff clean. The user explicitly approved tagging core-v0.2.0
  with a temporary exception for the two failed CPU performance gates.
- Measured 60-run synthetic 1080p p95: diagnosis 51.28 ms (target <30), numerical
  pipeline 582.47 ms (target <100), complete core_total 830.55 ms; peak RAM
  501.63 MiB. These timings remain failures of the original limits, not passes.
- Exception expires before core-v0.4.0: full hybrid core_total <=1100 ms p95 at
  1080p remains binding, including CPU diagnosis/classical work. GPU inference
  does not remove CPU cost. No waiver of resolution, quality, or later GPU gates.
- Required v0.4 work: proxy guided filtering then upsample; float32/OpenCV hot
  path; cv2.setNumThreads(0); shared Lab/luminance/box statistics; vectorized
  native patch extraction bounded by config; top-five function hotspots recorded
  in bench_results.md. Start v0.3.0 immediately after this tag.
- Neural/live modes remain unavailable until v0.3.0; no silent mock fallback.

## 0.1.0 — 2026-11-06

- Frozen contract v1.0 Python definitions copied verbatim.
- CPU FakeEngine for all four modes and every valid ablation combination.
- Ordered stages, twelve mock candidates, unique uint8 visual keys, result/story,
  alpha preservation, live-frame gamma, honest metrics, health and provenance.
- Independent output contract checker plus positive and deliberately broken runs.
- Strict request rejection, deterministic output, cancellation, and early close.
- Editable package and package-owned commented configuration.

Release gates: 243 CPU tests passed, Ruff passed, isolated Python 3.11.9 editable
installation verified, and pip check passed. Tag: `core-v0.1.0` on branch `core`,
per the user's branch restriction; no merge to main. This milestone supplies mock
integration only; `create_engine()` raises EngineUnavailable. Review required
before starting v0.2.0.


## 0.4.2 - 2026-11-07

- Hybrid restoration is anchored to full-resolution CIDNet, with guided local highlight repair. Output outside repair masks remains CIDNet.
- Denoise is localized to repair areas; recoverability smooths texture only in near-black flat areas and does not cap exposure. Pure black input remains black.
- The deterministic tuner evaluates twelve bounded candidates varying highlight repair, denoise strength, colour correction, and exposure trim. Candidate 0 is the least-correction tie-break.
- MetricSuite provides paired SSIM/LPIPS and unpaired PyIQA metrics, rejects non-finite outputs, and reports current-call unavailable reasons. The optional iqa extra pins setuptools==80.9.0 for CLIP-IQA.
- User-reported LOL-v1 eval15, three images: PSNR dl/hybrid 23.48/23.42, 21.56/21.56, 25.33/25.33 dB; SSIM delta 0.001; LPIPS equal or better for hybrid; clipped pixels 12.8/0.38%, 0.36/0.09%, and 0/0%.
- User-confirmed full suite: 359 passed, 13 GPU tests deselected. All Step 5 targets passed.
