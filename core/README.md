# lumenx-core 0.1.0

CPU-only FakeEngine and executable frozen contract v1.0. This milestone enables
server integration without weights, CUDA, HTTP, storage, or the real algorithms.

## Install

Python **3.10 or 3.11** is required. From the monorepo root:

```powershell
py -3.11 -m venv core\.venv
.\core\.venv\Scripts\python.exe -m pip install -e ".\core[dev]"
```

From an activated server environment: `python -m pip install -e ../core`.
The gate environment is deliberately under `core/` to honor the current write-set
restriction; a shared environment can still install the same package.

Dependencies are pinned in `pyproject.toml`. Do not install `opencv-python`,
`opencv-python-headless`, or `opencv-contrib-python-headless` alongside the pinned
`opencv-contrib-python` wheel. Import fails clearly if guided-filter support is
absent. Real-engine dependencies (PyTorch, pyiqa, scikit-image) will be pinned when
their releases are implemented and validated; they are not required by this fake.

## Use

```python
import numpy as np
from lumenx_core import create_fake_engine
from lumenx_core.contracts import EnhanceRequest, EnhanceResult
from lumenx_core.contract_check import validate_run

engine = create_fake_engine(stage_delay_ms=0, seed=0)
engine.warmup()  # optional and idempotent
image = np.full((480, 640, 3), 0.08, dtype=np.float32)
request = EnhanceRequest(image=image, original_size=(640, 480), mode="hybrid")
items = list(engine.run(request))
assert validate_run(items, request) == []
result = items[-1]
assert isinstance(result, EnhanceResult)
final = next(v.image for v in result.visuals if v.key == "final")
print(engine.health())
```

`create_engine()` raises `EngineUnavailable` in this release; real restoration is
never silently replaced by a mock. `create_fake_engine()` supports `hybrid`,
`classical`, `dl`, and `zerodce`, all legal ablation combinations, supplied alpha,
tiny images, explicit cancellation, and early generator closure. `enhance_frame`
accepts uint8 RGB with long side at most 1280 and returns the same shape/dtype.

Only the fake engine has been implemented. Brightness curves stand in for the
classical/deep stages; brightness groups stand in for measured confidence; noise
maps are explicit empty fixtures; no trained inference, real denoising, or actual
quality checks are claimed. Every run emits a fake warning and marks provenance
`fake=true`. `provenance.actual_method` names the requested contract-shaped mock
path (e.g. `cidnet` for `dl`), not an available trained model; always inspect the
fake flag and warnings. Health reports `fake=true`, `ready=true`, and
`hybrid_ready=false` (real hybrid processing is not available).

The fake MetricSuite computes actual finite PSNR and clipping statistics. Learned
IQA, SSIM, noise, colour/edge, and confidence-error metrics return `None` with
reasons via `unavailable()`. Perfect-match PSNR is also `None` rather than infinity.
Candidate scores are explicitly named fake brightness surrogates, not CLIP-IQA or
NIQE. Never use these outputs as a restoration benchmark.

## Integration details

- Follow the root `CONTRACT.md`, not obsolete HTTP/module layouts in references.
- `contracts.py` is copied verbatim from its frozen Python block. The checker
  accepts the explicitly required `L_gray` spelling despite the generic lowercase
  key comment. No frozen field or key is changed.
- Section 5's exact stage table governs fake ablation event schedules. In
  particular, `no_fusion` omits fusion but the other required hybrid stages still
  appear. Real baseline semantics will be addressed within the frozen contract.
- Candidate events precede the tuner stage; the chosen thumbnail is reused by
  reference and never emitted twice. Exactly one final result terminates a run.
- Visuals are uint8 RGB/gray/RGBA at natural geometry, except the required 128x128
  heightmap and thumbnail-sized candidates. The server applies `max_side`, encodes
  files, assigns URLs, preserves run data, and serializes access.
- The fake engine never emits the server-owned `input` key and performs no output
  file I/O, transport, scheduling, or locking. Calls are not thread-safe.
- Check cancellation between events; stop a partially consumed generator with
  `close()` in `finally`. Returned arrays remain owned by the returned objects;
  the engine does not retain a run cache after completion/closure.
- Configuration is package-local, independent of working directory. Public config
  returns an isolated copy. Provenance includes input/config hashes and both seeds.
- `stage_delay_ms` is a demonstration delay after each stage; no delay is added to
  candidates or warnings. Fake `core_total` includes delays and consumer pauses;
  it is not a GPU benchmark.

## Tests

From the monorepo root after installation:

```powershell
.\core\.venv\Scripts\python.exe -m pytest core/tests/test_contract.py core/tests/test_invalid_requests.py -q -o cache_dir=core/.pytest_cache
.\core\.venv\Scripts\python.exe -m ruff check core --cache-dir core/.ruff_cache
```

Tests include every legal flag combination, corrupted-output rejection, exact
contract-source comparison, all modes, tiny/black/bright/alpha inputs, determinism,
input isolation, cancellation, early-close cleanup, and invalid inputs before any
yield. These are CPU fake-engine gates; GPU/real-model tests do not exist yet and
are not represented by skipped tests.

## Weights, licences, and performance

No model weights, vendored architectures, real CLI, or GPU benchmark ship in
v0.1.0. Later model integration must preserve upstream licence headers and record
checkpoint hashes. HVI-CIDNet's upstream repository identifies an MIT licence;
Zero-DCE++ states academic/noncommercial use. See the original sources before
redistributing models:

- https://github.com/Fediory/HVI-CIDNet
- https://github.com/Li-Chongyi/Zero-DCE_extension#license

| Measurement | v0.1.0 status |
|---|---|
| Real 1080p hybrid p95 / 1100 ms core target | Not applicable; real engine absent |
| GPU VRAM, trained live throughput | Not measured; fake is CPU-only |
| Pre-approved performance levers | None applied |

See `HANDOFF.md` for actual environment and gate results. No performance claims are
made from fake output. Review stops at `core-v0.1.0`; no v0.2 work starts without
the user's review.
