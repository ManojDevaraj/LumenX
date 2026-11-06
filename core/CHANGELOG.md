# Changelog

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
