# Core handoff — v0.4.0 released

## core-v0.4.1 — release verified

## core-v0.4.2 development — Step 1

- Kept the reviewed publisher Zero-DCE++ default. Replaced the partial hybrid
  wiring with a CIDNet-anchored full-resolution path: only clipped (>=0.98) or
  hotspot-overshoot pixels are softly guided toward CIDNet's own tanh highlight
  shoulder. Classical output is not a highlight-fusion source.
- Removed recoverability gain enforcement from denoise, including its early
  bypass, so denoise is a texture correction and does not darken CIDNet output.
- The `no_dl` ablation retains its existing classical-only fusion fallback to
  preserve the frozen stage schedule. Step 1 regression gate: 22 passed.

## core-v0.4.2 development — metrics work in progress

- MetricSuite now attempts SSIM/LPIPS (paired) and NIQE/BRISQUE/CLIP-IQA/MUSIQ
  (unpaired) through pinned PyIQA, selecting CUDA whenever it is available and
  preserving a reason when a model cannot run. PSNR and `psnr_gt_mean` remain
  direct numerical measurements.

External verification found the tuned-hybrid proxy mask crash at 720p/1080p.
The candidate replaces the one-channel proxy operation with a direct two-dimensional
nearest-neighbour resize and adds CPU/GPU tests for flat and noisy non-black 720p
and 1080p inputs. Request noise values now use native-resolution patch estimates,
while perceptual metrics remain on the 512px proxy.

Training script is `core/scripts/train_zerodce.py`; it used only
`C:\Users\Manoj D\Desktop\CV_hackathon\LOLdataset\our485\low` (485 images),
initialized from the publisher Zero-DCE++ checkpoint, on CUDA FP32 for 10 epochs.
Loss decreased from 2.840631 at epoch 1 to 2.272758 at epoch 10. The first command
session ended after epoch 3; training resumed from that saved state for seven more
epochs. Checkpoint: `core/weights/zerodce_pp_finetuned.pth`; SHA256:
`90d375b5b00a2d413d0baf0e382d9dbf8128ba9e8ff30b84b797965a8a5ea65f`.
The package default selects `models.zerodce_checkpoint: finetuned`; publisher weights
remain available by setting it to `publisher`. Registry verified all four checkpoint
files and CUDA placement (`cuda:0`).

Validation: CPU suite 359 passed, 13 GPU-marked cases deselected; Ruff clean.
The user confirmed 39 GPU tests passed (149.18s) across test_hybrid,
test_metrics and test_models_gpu after restoring Torch 2.6.0+cu124 and contrib
OpenCV 4.11.0.86. PyIQA was verified by the user. Release tag core-v0.4.1 is created.

## Final backend completion inventory (2026-11-07)

Read-only dataset discovery found LOL-v1 at
`C:\Users\Manoj D\Desktop\CV_hackathon\LOLdataset\our485\low` (training only)
and `C:\Users\Manoj D\Desktop\CV_hackathon\LOLdataset\eval15\low` (evaluation only),
with paired `high` sibling directories. LOL-v2 Real is at
`C:\Users\Manoj D\Desktop\CV_hackathon\LOL-v2\Real_captured\{Train,Test}\{Low,Normal}`
and Synthetic is at `C:\Users\Manoj D\Desktop\CV_hackathon\LOL-v2\Synthetic\{Train,Test}\{Low,Normal}`.
No LIME, NPE, DICM or college dataset directory exists under the discovered
`C:\Users\Manoj D\Desktop\CV_hackathon` tree. They must be located before their
required real-image check and final evaluation; no datasets were copied into the repo.

The core virtual environment is currently missing Torch/OpenCV after the interrupted
CUDA restoration transfer, so no CPU or GPU runtime/test/training/evaluation claim
has been made after v0.4.1 candidate `c7cc01a`. Restore it using `constraints.txt`,
then verify CUDA, ximgproc and the metric models before release work resumes.

## Current development status

core-v0.2.0 was tagged on core at cd572df with the user's temporary performance
exception. Exact measurements/exception are preserved below and in CHANGELOG.md.
Work proceeded immediately into v0.3.0, only under core/ and on branch core.

Implemented: official architectures/notices, publisher SHA256 manifest/downloader,
once-loaded registry, measured placement preflight, precision handling, tiled
adapters, live adapter, confidence fusion, ablations and one repair reusing cached
deep output. Direct baselines remain unmodified; no_fusion is a postprocess bypass;
no_recoverability removes all confidence control signals. No trained data invented.

Until v0.4, hybrid explicitly requires no_tuner; otherwise EngineUnavailable before
output. No input/contract field is changed. Full hybrid_ready remains false;
hybrid_without_tuner_ready reflects actual loaded models. No random weights count
as a verified model. Registry checks bytes/state dictionaries before inference.

GPU gates were verified in the preceding session on the target RTX 2050: all four
mode contracts passed, both CIDNet checkpoints ran in FP16 without fallback, and
hybrid completed at native 1080p and 12 MP without OOM. Tiling, fusion and the
single repair pass were exercised, with cached deep output reused. Measured peak
resident VRAM was 407 MiB. This verifies the v0.3.0 GPU gate; latency remains a
reported metric rather than a release gate by user-approved scope reduction.

The v0.3.0 release tag is created on `core`. v0.4 fixes estimator support,
checkerboard flatness, denoise noise monotonicity and conservative brightness,
with regression coverage. It adds a deterministic 12-candidate proxy tuner,
candidate events, fixed-score feasibility selection and request-time MetricSuite
metrics. PyIQA is pinned in the optional `core[iqa]` extra; if unavailable at runtime, CLIP-IQA is
null with a reason and the tuner identifies the degraded `custom_no_clipiqa`
recipe. Confidence-validation is intentionally out of scope; webcam and
`enhance_frame` tuning comes after the v0.4 still-image path.

Top-five separate CPU profile is in bench_results.md: guidedFilter, denoise, max,
enhance_classical and multiply. It is not the required final hybrid v0.4 profile
and does not overwrite the original v0.2 benchmark sample or claim a speedup.

---

# Released core-v0.2.0 (approved temporary performance exception)

## Current status and review decision

The user explicitly approved tagging core-v0.2.0 with a documented temporary CPU
performance exception and instructed v0.3.0 to start immediately afterward. Work
remains on core and inside core/; frozen contract, server, main and references
are unchanged. Functional gates pass; the original CPU timing gates still fail.

Completed: native-patch noise/confidence, illumination/SNR/cast/hotspots/clipping,
input-based recovery, adaptive Retinex/partial Lab CLAHE/colour, three-strength
guided denoising, actual gain enforcement, real custom metrics, audit checks,
full-size map/report/story/alpha output, frozen config, debug CLI and benchmark.
FakeEngine remains compatible. Learned IQA is null with reasons; no trained model
is present. Neural/live requests fail before output. Classical has no tuner and
guardrails are audit-only as required for v0.2.0.

Validation: **325 CPU tests passed; Ruff passed**. New cases cover reference colour
pairs, injected audit failures, native noise accuracy, actual gain caps, alpha
isolation, neutral RGB preservation and tiny OpenCV scalar/filter corner cases.
Developer CLI smoke test completed; synthetic input/final were visually inspected.
This is not a real-photo quality evaluation or a GPU test run.

Benchmark: core/bench_results.md; 20 synthetic 1920x1080 cases x three warm passes.
Diagnosis p95 **51.28 ms** (limit <30), numerical p95 **582.47 ms** (limit <100),
complete core p95 **830.55 ms**, peak RAM about 502 MiB. CPU only; no RTX/CUDA,
neural, live or end-to-end server timing claim.

Optimizations without resolution/stage cuts: OpenCV image/channel operations,
8-bit Lab histogram operations, fixed threading, official Fast Guided Filter
coefficients with full-resolution evaluation, a proven native-max highlight
bypass, and no-noise/bright denoise bypass retaining actual gain enforcement.
No pre-approved hybrid performance lever applied; no silent scoring cuts.

The user's latest approval supersedes the original no-failed-gate-tag rule only
for the v0.2.0 CPU timing exception. It does not reclassify measurements as passes.
Before core-v0.4.0 the full warm hybrid core_total <=1100 ms p95 at 1080p remains
mandatory, including CPU stages. GPU model integration does not fix those costs.
No contract change or quality/resolution reduction was approved. Dependency pins
are unchanged. bench_results.md embeds observed v0.2 health and timing details.

## Binding v0.4 performance work (latest user instruction)

1. Compute all guided filters at proxy resolution, then upsample (fast guided filter).
2. Keep the hot path float32, use OpenCV rather than skimage, call cv2.setNumThreads(0).
3. Compute Lab/luminance conversions and box-filter statistics once; reuse across stages.
4. Bound native noise sampling to the configured patch budget using vectorized extraction.
5. Profile per function and record the top five hotspots in bench_results.md.

These requirements must be verified before v0.4.0; the exception is temporary.
Continue v0.3.0 model registry/wrappers, tiling, live inference, fusion and one
bounded cached-output repair now. No automatic waiver for v0.3 GPU/weight gates.

---

# Archived handoff — core-v0.1.0

## Status

The contract/FakeEngine milestone is complete and release gates passed. Release
tag: `core-v0.1.0`, on branch `core`. No `main` merge or `server` branch change is
authorized by the user's current instruction. Work stops here for review.

## Completed

- Contract version 1.0 definitions copied verbatim.
- FakeEngine, all modes/flags, maps/events/results/live mock/MetricSuite/health.
- Contract validator and release-gate tests, package metadata and documentation.
- Only paths under `core/` created/modified.

## Environment and validation

- Validated Python: 3.11.9, copied by the user into ignored `core/.python311`.
- Isolated environment: `core/.venv`, include-system-site-packages=false.
- Editable installation: lumenx-core 0.1.0 points to this checkout's `core/`,
  confirmed through installed distribution metadata and isolated-mode import.
- Validated pins: numpy 2.2.6, opencv-contrib-python 4.11.0.86, PyYAML 6.0.2,
  pytest 8.3.5, Ruff 0.11.13. Exactly one OpenCV distribution is installed and
  cv2.ximgproc.guidedFilter is available. `pip check`: no broken requirements.
- Python 3.10 syntax parsing of all Python sources: passed.
- Required pytest gate: **243 passed**, zero skipped, initial measured run 4.46 s.
- Ruff: **All checks passed**.
- Network from agent terminal remains blocked to GitHub/PyPI. The initial push
  attempt failed before connection. A local tag is not evidence of a remote tag;
  remote publication must be verified separately. Never force-push.
- No GPU benchmarks: this release implements no real engine.

## Observed warm health output

```json
{
  "fake": true,
  "ready": true,
  "hybrid_ready": false,
  "warm": true,
  "device": "cpu",
  "placement_policy": "resident",
  "models": {
    "cidnet": {"available": false, "checkpoints": []},
    "zerodce": {"available": false},
    "iqa": {"clipiqa": false, "niqe": false, "brisque": false, "musiq": false, "lpips": false}
  },
  "vram_mb": {"total": 0, "allocated": 0, "reserved": 0, "peak": 0},
  "degraded": ["fake_engine", "real_engine_not_implemented"],
  "core_version": "0.1.0",
  "contract_version": "1.0"
}
```

## Known limitations and open issues

- No trained restoration, real recovery estimate, or learned IQA exists at v0.1.0.
- No model weights or vendored model code ship yet; nothing is downloaded at runtime.
- The frozen contract requires `L_gray` despite its generic lowercase key comment;
  the checker preserves the explicit required key without changing the contract.
- User says work only on `core` and stop after tagging. This supersedes the prompt's
  main-merge release steps: tag on core after gates pass; do not merge into main.
- The supported-Python editable-install, pytest and Ruff gates have passed.
  No contract change is requested.

## Review boundary

Stop at the first v0.1.0 tag. No v0.2.0 implementation before user review.
