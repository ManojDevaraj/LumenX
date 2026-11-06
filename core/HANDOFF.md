# Core handoff — core-v0.1.0

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
