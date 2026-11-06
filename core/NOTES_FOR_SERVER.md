# Notes for server integration

## core-v0.3.0 — GPU gates verified

The RTX 2050 GPU gate is complete: all four modes passed their contracts, CIDNet
LOLv2-real and SID ran FP16 with no fallback, and native hybrid processing completed
at 1080p and 12 MP without OOM. Tiling, fusion and one cached-deep-output repair
were verified; resident peak VRAM was 407 MiB. v0.4 tuner and metrics work is now
in progress. Latency remains reported rather than release-blocking by approved scope.

v0.4 removes the default-hybrid `no_tuner` requirement. Hybrid emits twelve
candidate events before its tuner event and returns the same candidates in the
result. Candidate scores use `clipiqa+custom` when the optional `core[iqa]` PyIQA
extra is installed; otherwise
the result names `custom_no_clipiqa` and includes the unavailable reason. Request
metrics now include CLIP-IQA before/after when available. No server changes are
required.

## v0.3.0.dev0 — historical development note

Latest released tag is core-v0.2.0. Branch code now wires registry/wrappers, tiled
inference, direct baselines, confidence fusion, live and one cached-output repair.
Optional model dependencies are pinned; no request/startup downloads occur.

Before v0.4, hybrid requires explicit ablate=frozenset({"no_tuner"}). Default
hybrid raises EngineUnavailable before output; do not silently change requests.
Frozen stage table is unchanged; candidates=[] when no tuner runs. Health adds
hybrid_requires_no_tuner and hybrid_without_tuner_ready; full hybrid_ready=false.
Inspect actual method/fallback_chain: missing trained weights may explicitly
degrade dl/zerodce to classical_fallback. Live never becomes a gamma mock.

349 CPU tests pass with nine actual GPU gates deselected, not claimed passing.
Torch/einops and CIDNet weights are still absent in the visible setup. No v0.3
tag or GPU claims. No contract/server changes. The temporary v0.2 exception does
not relax CPU cost accounting or the final full hybrid 1100 ms p95 target.

## core-v0.2.0 — temporary CPU-performance exception approved

create_engine(device="cpu") now returns real classical processing. Request
mode="classical" explicitly: the default hybrid mode, dl, zerodce and real live
raise EngineUnavailable before output. Health reports fake=false, ready=true,
modes=[classical], hybrid_ready=false. No dependency or frozen contract change.

325 CPU tests and Ruff pass. Real diagnosis/recovery/classical/denoise/audit stages
produce full maps and preserved final alpha. Actual output gain is capped; hidden
RGB cannot drive diagnosis. Guardrails audit only; failures remain warnings and
no repair is claimed. Custom metrics are real, learned IQA remains null with
reasons. Use MetricSuite.unavailable() and per-result metrics.unavailable.

The 60-run synthetic 1080p report is core/bench_results.md: diagnosis p95 51.28 ms,
numerical p95 582.47 ms, complete core p95 830.55 ms. Original CPU limits still
fail; the user explicitly approved the core-v0.2.0 tag with this temporary exception
and immediate v0.3.0 work. Full hybrid core_total <=1100 ms p95 before v0.4.0 is
unchanged, including CPU costs. Do not advertise CPU figures as RTX/server timings.

## v0.1.0 contract fixture

At the v0.1.0 tag: import `create_fake_engine`. `create_engine` deliberately raises
EngineUnavailable until real processing ships. Contract version is 1.0; package
version is 0.1.0. Release tag is `core-v0.1.0` on branch `core` (no main merge under
the user's current restriction). The release passes 243 CPU tests and Ruff. See
HANDOFF.md for environment, observed health, and remote publication limitations.

The fake supports hybrid, classical, dl, zerodce, and legal ablations according to
the frozen stage table. Serialize engine/MetricSuite access on the server; core
does not lock or schedule. Consume StageEvents then exactly one EnhanceResult.
Use generator.close() in finally if the consumer stops early. A cancelled run
raises Cancelled without a result.

All mandatory visual keys/transport hints are supplied, including `L_gray` with
its required case. Do not duplicate primary-referenced candidate visuals. Publish
the input yourself, and resize/encode the arrays using their Visual metadata.

Explicit mock limitations: no trained model is loaded, IQA scores and guardrail
values are None with reasons, noise/strength maps are fixtures, candidate totals
are brightness-surrogate scores, and confidence colours are brightness groups.
Keep fake=true visibly distinguished. Health hybrid_ready=false means real hybrid
processing is absent, not that the fake cannot produce a hybrid-shaped stream.

The fake result uses the contract enum's method name for its corresponding mock
path; `provenance.fake=true` and warnings identify that no neural inference ran.
No server code or frozen docs were changed. This milestone does not include
server I/O, CLI, models, actual enhancement algorithms, or evaluation orchestration.
