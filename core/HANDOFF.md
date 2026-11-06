# Core handoff — v0.1.0 candidate

## Status

Implementation of the contract/FakeEngine milestone is complete; release gates
and tag are pending. Work remains on branch `core`; no `main` or `server` branch
changes are authorized by the user's current instruction.

## Completed

- Contract version 1.0 definitions copied verbatim.
- FakeEngine, all modes/flags, maps/events/results/live mock/MetricSuite/health.
- Contract validator and release-gate tests, package metadata and documentation.
- Only paths under `core/` created/modified.

## Environment and validation

- Initially visible Python: 3.13.2, unsupported by the required 3.10/3.11 range.
- Python 3.10 syntax parsing of all Python sources: passed.
- Network from agent terminal: blocked socket access to GitHub and PyPI.
- User reported environment prepared; the visible parent `.venv` points to a
  Python311 executable not currently accessible to this runtime. Expected
  `core/.venv` is not yet visible. Awaiting the supported interpreter/environment.
- Editable install, pytest, Ruff, and runtime health: pending actual execution.
- No GPU benchmarks: this release implements no real engine.

## Known limitations and open issues

- No trained restoration, real recovery estimate, or learned IQA exists at v0.1.0.
- No model weights or vendored model code ship yet; nothing is downloaded at runtime.
- The frozen contract requires `L_gray` despite its generic lowercase key comment;
  the checker preserves the explicit required key without changing the contract.
- User says work only on `core` and stop after tagging. This supersedes the prompt's
  main-merge release steps: tag on core after gates pass; do not merge into main.
- A release must not be tagged until a clean supported-Python editable install,
  required pytest gate, and Ruff checks pass. No contract change is requested.

## Review boundary

Stop at the first v0.1.0 tag. No v0.2.0 implementation before user review.
