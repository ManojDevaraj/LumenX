# lumenx-core - v0.2.0

Real CPU classical restoration plus explicit FakeEngine, using frozen contract
v1.0. Released as core-v0.2.0 with the user's temporary CPU-performance exception.
Original CPU limits still fail. See HANDOFF.md and bench_results.md.

## Install and use

Python 3.10/3.11 is required. From the monorepo root:

```powershell
.\core\.venv\Scripts\python.exe -m pip install -e ".\core[dev]"
```

For a new environment, first run `py -3.11 -m venv core/.venv`. From an activated
server environment use `python -m pip install -e ../core`. Dependencies remain
numpy 2.2.6, opencv-contrib-python 4.11.0.86 and PyYAML 6.0.2. Install only that
OpenCV wheel; import verifies cv2.ximgproc.guidedFilter. No neural dependencies
have been added before their integration milestone.

```python
import numpy as np
from lumenx_core import create_engine
from lumenx_core.contracts import EnhanceRequest
from lumenx_core.contract_check import validate_run

engine = create_engine(device="cpu")
engine.warmup()
req = EnhanceRequest(np.full((480, 640, 3), 0.08, np.float32),
                     (640, 480), mode="classical")
items = list(engine.run(req))
assert validate_run(items, req) == []
print(items[-1].headline, items[-1].timings_ms)
```

Specify classical explicitly: EnhanceRequest defaults to hybrid. Real hybrid,
dl, zerodce and live modes raise EngineUnavailable before yielding in v0.2.0.
Unsupported explicit GPU selection also raises; default/device=cpu uses the real
CPU classical path. No requested neural inference silently becomes mock output.
create_fake_engine(stage_delay_ms=0, seed=0) remains available for all four modes
and legal ablations, with explicit fake=true, mock maps and honest surrogate/null
quality scores. Health ready=true means classical works; hybrid_ready=false means
trained hybrid restoration is absent.

## Classical algorithm and controls

- 512-long-side guided max-channel illumination, full-size input-aligned maps.
- Up to 100 deterministic native 32x32 flat-patch high-pass MAD noise estimates,
  interpolated local noise and explicit estimator confidence. Tiny/no-flat-patch
  images use a conservative low-confidence fallback, not a claim of zero noise.
- Input SNR plus estimator support and joint near-black variance controls recovery.
  One dark colour channel alone never forces unrecoverable. Required labels describe
  estimates, not guaranteed real detail or proof that recovery is impossible.
- Severity-derived Retinex, partial Lab CLAHE/cast correction, highlight protection.
  Already-bright identity uses percentile, median and crushed-shadow agreement.
  RGB math is float32; histogram/colour conversion uses an 8-bit Lab representation
  before returning to float32 RGB. Thresholds and weights are in default.yaml.
- Three-strength Fast Guided Filter denoising, gain/noise/confidence strength,
  chroma suppression and reliable-edge protection. Half-scale filter coefficients
  are evaluated at full output size; tiny dimensions use full-scale coefficients.
- Actual final RGB luminance gain caps, not just a displayed map. Fully black input
  stays black; common RGB scaling preserves hue. Uint8 output has normal rounding.
- Real clipping/chroma/edge/hotspot-ring guardrails, **audit only** in this milestone.
  Failed checks remain warnings; no repair is claimed or silently attempted.
- Measured-fact headline/chips/decisions/story, captions <=20 words, no runtime LLM.

Server supplies oriented, C-contiguous float32 RGB up to 12 MP, with separate alpha.
Core preserves processed resolution and alpha; hidden RGB cannot influence diagnosis.
All-transparent requests explicitly report no visible evidence and use a 100% low
confidence convention to retain a 100% area total, without fabricated metrics.
No upload decoding, EXIF handling, HTTP, scheduling, run storage or production image
encoding belongs to core. Only config reads and dev CLI/benchmark I/O occur.

## Configuration, metrics and contracts

create_engine(config_path=...) accepts known-key YAML overrides. Config is validated,
recursively immutable, hashed and independent of working directory. public_config()
returns an isolated JSON-safe copy without private paths. Standard mathematical
coefficients/code ranges are definitions, not tunable thresholds.

MetricSuite supplies actual PSNR/GT-mean PSNR, clipping, chroma-only CIEDE2000,
reliable-edge correlation and native-patch noise. Request metrics use a 512-long-side
proxy; native diagnosis noise is separately labelled. NIQE, BRISQUE, CLIP-IQA, MUSIQ,
LPIPS, SSIM and confidence calibration remain unavailable until later milestones.
Every null has a reason; exact-match infinite PSNR is null. Never present missing
IQA as a fake score. Confidence remains a heuristic awaiting evaluation calibration.

Formula reference/test vectors: [Sharma/Wu/Dalal CIEDE2000 notes](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf).
Filter reference: [OpenCV Guided Filter](https://docs.opencv.org/4.x/de/d73/classcv_1_1ximgproc_1_1GuidedFilter.html).

contracts.py remains verbatim. validate_run checks stage order, unique visuals,
metadata, references, candidates, percentages, JSON safety and alpha. It preserves
required L_gray despite the generic lowercase key comment. Server publishes input,
resizes/encodes visual arrays using max_side/lossless, and serializes all engine and
MetricSuite calls. Stop abandoned generators with close() in finally; cancellation
raises Cancelled between stages. No engine run cache is retained after closure.
Fake stage delay is for UI demonstration, not a GPU timing measurement.

## Tests and developer tools

```powershell
.\core\.venv\Scripts\python.exe -m pytest core/tests -q -o cache_dir=core/.pytest_cache --basetemp=core/.tmp/pytest
.\core\.venv\Scripts\python.exe -m ruff check core --cache-dir core/.ruff_cache
.\core\.venv\Scripts\python.exe -m lumenx_core.cli enhance image.png core/bench_results/debug --mode classical
.\core\.venv\Scripts\python.exe core/scripts/bench.py --images 20 --passes 3 --check-gates
```

325 CPU tests pass and Ruff is clean. Coverage includes both engines, every legal
fake flag combination, malformed output rejection, real tiny/odd/black/gray/bright/
noisy/coloured/transparent inputs, actual gain limits, native noise accuracy,
reference colour pairs, injected guardrail damage, config isolation, determinism,
cancellation and early cleanup. No GPU tests are skipped and reported as passing.
The CLI writes natural-size debug visuals through plain cv2.imwrite; it is not a
production upload, encoding or storage implementation.

## Measured performance and release blocker

Windows CPU, Python 3.11.9, four OpenCV threads, 20 deterministic synthetic 1080p
cases x three warm passes (60 runs). These are performance cases, not real photos
or a quality benchmark:

| Measurement | p95 ms | Gate |
|---|---:|---|
| Diagnosis | 51.28 | FAIL: <30 ms |
| Numerical (diagnose + recovery + classical + denoise) | 582.47 | FAIL: <100 ms |
| Complete core including maps/metrics/report | 830.55 | CPU measurement; no GPU claim |

Peak process RAM about 502 MiB. Complete details/means/max and observed health are
in bench_results.md. No RTX 2050 neural, live or end-to-end server measurements
exist. No pre-approved hybrid performance lever was applied. Resolution and stages
were preserved. The user explicitly approved the v0.2.0 exception and immediate
v0.3.0 work. Before v0.4.0 the full hybrid core_total <=1100 ms p95 target still
applies, including CPU stages. Required optimizations are recorded in HANDOFF.md.

## Weights and licences

No pretrained weights or vendored neural architecture code ship yet. Future model
integration must preserve upstream notices and checkpoint hashes. HVI-CIDNet's
repository identifies MIT; Zero-DCE++ states academic/noncommercial use:

- https://github.com/Fediory/HVI-CIDNet
- https://github.com/Li-Chongyi/Zero-DCE_extension#license
