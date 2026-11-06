# Notes for server integration

## v0.1.0 contract fixture

Import `create_fake_engine` from `lumenx_core`. `create_engine` deliberately raises
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
