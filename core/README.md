# LumenX Core

`lumenx-core` 0.4.0 implements the frozen v1.0 stream contract and the real adaptive restoration pipeline. The CPU path estimates illumination and noise, decomposes illumination and reflectance, applies highlight-aware Retinex and Lab CLAHE, and performs edge-aware noise reduction. The optional Zero-DCE++ path can be fused with classical and conservative safe paths. All outputs include measured diagnostics, guardrail results, and provenance.

## Install

From the repository root with Python 3.10 or later:

```powershell
python -m venv core\.venv
core\.venv\Scripts\python.exe -m pip install -e "core[vision]"
```

The vision extra includes OpenCV contrib, scikit-image, PyYAML, SciPy, and Pillow. Zero-DCE++ is optional:

```powershell
core\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
core\.venv\Scripts\python.exe -m pip install -e "core[models]"
core\.venv\Scripts\python.exe -m lumenx_core.download_weights
```

The first command is for CPU-only inference; install the matching CUDA-enabled PyTorch build for a CUDA target. This downloads the authors' `Epoch99.pth` checkpoint into `core/weights/`. Its upstream license is CC BY-NC 4.0; the repository does not redistribute the checkpoint. Learned IQA metrics are separately optional: install `core[iqa]` to enable pyiqa metrics. The metrics load lazily and may fetch their own weights on first use.

## Run one image

```powershell
core\.venv\Scripts\python.exe -m lumenx_core.cli enhance .\input.jpg --mode hybrid -o .\output.png --device auto
```

Supported modes are `classical`, `dl`/`zerodce`, and `hybrid`. If the optional learned checkpoint is missing, direct learned mode produces a disclosed classical fallback; hybrid disables the unavailable model contribution. The output directory receives stage images and a JSON report. For server uploads, the service owns decoding, EXIF orientation, alpha handling, upload limits, and downscaling.

## Adaptive search and ablations

Hybrid mode compares twelve deterministic settings on a 256-pixel proxy. Its custom score combines exposure, highlight clipping, colour change, edge retention, and noise measurements. It does not claim to be a learned perceptual metric. The request accepts these hybrid ablations: `no_tuner`, `no_fusion`, `no_classical`, `no_dl`, `no_gain_denoise`, `no_recoverability`, `no_guardrails`, and `no_color_correction`.

The engine attempts one bounded blend toward the input if the final audit detects a guardrail failure, then records any remaining failures. PSNR/SSIM/LPIPS require paired references; NIQE/BRISQUE/CLIP-IQA/MUSIQ are nullable when their optional metric model is unavailable.

## Training

`scripts/train_zerodce.py` provides a no-reference Zero-DCE++ fine-tuning path using exposure, spatial consistency, colour constancy, and curve smoothness losses. It expects the user to provide a lawful low-light image directory. No dataset or trained checkpoint is included. Training outputs are not automatically installed as the runtime checkpoint.

## Main modules

`contracts.py` and `contract_check.py` define and inspect the stream interface; `diagnose.py`, `recoverability.py`, `classical.py`, `fusion.py`, `denoise.py`, `guardrails.py`, `metrics.py`, and `report.py` implement the pipeline; `models/` contains the optional model adapter and registry; `cli.py` runs local images; `config/default.yaml` holds tunable thresholds.
