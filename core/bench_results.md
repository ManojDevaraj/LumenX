# v0.2.0 CPU classical benchmark

Python 3.11.9; Windows-10-10.0.26200-SP0
Processor: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel
Synthetic cases: 20; warm passes: 3; samples: 60
All inputs are 1920x1080. These are deterministic performance cases, not real photos or quality benchmarks.
Cold factory + warmup: 179.57 ms
Peak process RAM: 501.62890625 MiB
GPU VRAM / hybrid inference: not applicable, absent in this milestone.
Policy: CPU resident; no perf levers applied; full input resolution preserved.

| Stage | Mean ms | p95 ms | Maximum ms |
|---|---:|---:|---:|
| diagnose | 46.87 | 51.28 | 52.37 |
| render | 105.38 | 112.49 | 115.53 |
| recoverability | 32.27 | 36.36 | 39.51 |
| classical | 129.22 | 178.13 | 181.13 |
| denoise | 107.06 | 386.33 | 409.70 |
| metrics | 98.09 | 131.56 | 133.27 |
| guardrails | 6.97 | 8.09 | 11.62 |
| report | 0.01 | 0.02 | 0.03 |
| core_total | 544.80 | 830.55 | 868.96 |
| numerical | 315.42 | 582.47 | 605.73 |

Numerical = diagnosis + recoverability + classical + denoise, excluding render, metrics and report.
Release performance gate: FAIL (numerical p95 <100 ms and diagnosis p95 <30 ms).
core_total additionally includes required full-resolution visual array generation and remaining core work.
Not an RTX 2050 GPU measurement; no end-to-end server latency claim.

## Observed health

```json
{
  "fake": false,
  "ready": true,
  "hybrid_ready": false,
  "warm": true,
  "device": "cpu",
  "placement_policy": "resident",
  "modes": [
    "classical"
  ],
  "models": {
    "cidnet": {
      "available": false,
      "checkpoints": []
    },
    "zerodce": {
      "available": false
    },
    "iqa": {
      "clipiqa": false,
      "niqe": false,
      "brisque": false,
      "musiq": false,
      "lpips": false
    }
  },
  "vram_mb": {
    "total": 0,
    "allocated": 0,
    "reserved": 0,
    "peak": 0
  },
  "degraded": [
    "neural_modes_unavailable",
    "iqa_unavailable",
    "guardrails_audit_only"
  ],
  "core_version": "0.2.0",
  "contract_version": "1.0"
}
```
