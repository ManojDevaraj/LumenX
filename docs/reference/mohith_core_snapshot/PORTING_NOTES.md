# Reference snapshot: Mohith's engine (PR #1) — READ-ONLY

Source: branch `origin/codex/lumenx-hackathon-project` (author Mohith-D05), commit 7e37cd8.
These files are a **reference for Codex**, not part of the product. Nothing imports them.
Do not edit them; port ideas/code into `core/` and credit "Mohith-D05 (PR #1)" in `core/CHANGELOG.md`.

His engine was a separate implementation (Zero-DCE++ only, CPU). The official engine is `core/` (Codex).
His server (`server/`) was merged into `main` as the official server.

## What to port into `core/` (priority order)

1. **Fine-tuning — mandatory requirement** ("fine-tune or adapt the model wherever suitable training data is available").
   - `scripts/train_zerodce.py`: no-reference Zero-DCE++ fine-tuning (exposure, spatial-consistency, colour, illumination-smoothness losses).
   - Port to `core/scripts/train_zerodce.py`, using **our** Zero-DCE++ architecture/registry (`core/lumenx_core/models/`), GPU, FP32 training.
   - Train only on LOL `our485` (or other training splits). Never on LOL-v1 `eval15`, LOL-v2 test, DICM, LIME, the college test set.
   - Output `core/weights/zerodce_pp_finetuned.pth` + SHA256 recorded; selectable via config (default stays the publisher checkpoint unless results justify switching).
   - His measured outcome (CPU, 1 epoch, LOL-v1 eval15): hybrid SSIM 0.548 → 0.586 and guardrail failures 3/15 → 1/15; learned-only PSNR 15.36 → 16.93 but more colour-shift failures. See `VALIDATION_REPORT_his_engine.md`.

2. **Working 12-candidate tuner** — `lumenx_core/tuner.py` (+ `fusion.py`, `metrics.py`, `pipeline.py` for how it is called).
   - His tuner runs hybrid at full sizes without crashing. Compare its proxy/mask handling with ours: our v0.4.0 crashes at 720p/1080p with
     `IndexError` at `diagnose.py:49 native_noise` ← `metrics.py:110 custom_values` ← `tuner.py:53 tune` because the proxy `valid` mask is 1-D.
   - Useful ideas: diagnosis-centred gamma window (±0.08 around a brightness-derived centre), LHS over the other parameters, 2 reference candidates.
   - Keep our contract semantics (12 events, exactly one chosen, feasibility-first ranking, `clipiqa+custom` recipe).

3. **Evaluation experience** — `VALIDATION_REPORT_his_engine.md` shows table formats and failure categories the server's eval scripts produce. Numbers there came from his engine and are **not** final results.

## Not to port

- His contracts/fake/validation copies (ours are canonical), his registry/weights downloader (ours verifies hashes), his CPU-only defaults, Python 3.13 assumptions.
