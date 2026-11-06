# Changelog

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
