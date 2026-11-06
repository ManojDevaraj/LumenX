"""Gradio entry point for the free Hugging Face ZeroGPU Space."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

try:
    import spaces
except ImportError:
    # Let the same UI run locally without the Hugging Face ZeroGPU package.
    class _SpacesFallback:
        @staticmethod
        def GPU(function: Any = None, **_: Any) -> Any:
            if function is None:
                return lambda wrapped: wrapped
            return function

    spaces = _SpacesFallback()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
WEIGHTS_DIR = ROOT / "core" / "weights"
CHECKPOINT = WEIGHTS_DIR / "zerodce_pp.pth"
os.environ.setdefault("LUMENX_WEIGHTS_DIR", str(WEIGHTS_DIR))

from lumenx_core import EnhanceRequest, EnhanceResult, create_engine  # noqa: E402
from lumenx_core.download_weights import download  # noqa: E402

try:
    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _device = "cpu"

_checkpoint_notice = ""
if not CHECKPOINT.is_file():
    try:
        download(CHECKPOINT)
    except (OSError, RuntimeError) as exc:
        _checkpoint_notice = f"Checkpoint download failed; learned modes will disclose fallback behavior ({exc})."

# ZeroGPU's CUDA emulation is active during module import, so load the model
# here. The decorated call below requests the real shared GPU for inference.
ENGINE = create_engine(device=_device)
ENGINE.warmup()

_MAX_PROCESSING_PIXELS = 12_000_000


def _fit_pixel_budget(width: int, height: int) -> tuple[int, int]:
    """Keep engine input within its memory-safe pixel budget."""

    pixels = width * height
    if pixels <= _MAX_PROCESSING_PIXELS:
        return width, height

    scale = (_MAX_PROCESSING_PIXELS / pixels) ** 0.5
    fitted_width = max(1, int(round(width * scale)))
    fitted_height = max(1, int(round(height * scale)))
    while fitted_width * fitted_height > _MAX_PROCESSING_PIXELS:
        if fitted_width >= fitted_height:
            fitted_width -= 1
        else:
            fitted_height -= 1
    return fitted_width, fitted_height


def _format_result(image: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    if image is None:
        raise gr.Error("Upload a low-light image first.")
    if image.ndim != 3 or image.shape[2] < 3:
        raise gr.Error("Choose an RGB or RGBA image.")

    rgb = image[:, :, :3]
    original_height, original_width = rgb.shape[:2]
    processing_width, processing_height = _fit_pixel_budget(original_width, original_height)
    was_resized = (processing_width, processing_height) != (original_width, original_height)
    if was_resized:
        rgb = cv2.resize(
            rgb,
            (processing_width, processing_height),
            interpolation=cv2.INTER_AREA,
        )
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype == np.uint8:
        normalized = rgb.astype(np.float32) / 255.0
    else:
        normalized = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    normalized = np.ascontiguousarray(normalized)

    result: EnhanceResult | None = None
    request = EnhanceRequest(
        image=normalized,
        original_size=(original_width, original_height),
        mode=mode,  # type: ignore[arg-type]
    )
    for item in ENGINE.run(request):
        if isinstance(item, EnhanceResult):
            result = item
    if result is None:
        raise gr.Error("The restoration run did not produce an output.")

    final = next((visual.image for visual in result.visuals if visual.key == "final"), None)
    if final is None:
        raise gr.Error("The engine did not return its final image.")
    if was_resized:
        final = cv2.resize(
            final,
            (original_width, original_height),
            interpolation=cv2.INTER_CUBIC,
        )

    chips = [
        f"- **{chip.label}:** {chip.value:.3f} ({chip.severity})"
        if chip.value is not None
        else f"- **{chip.label}:** unavailable"
        for chip in result.chips
    ]
    failures = [check.check for check in result.guardrails if check.passed is False]
    audit = ", ".join(failures) if failures else "No configured guardrail failures"
    duration = result.timings_ms.get("core_total", 0.0)
    details = [
        f"### {result.headline}",
        f"Method: `{result.provenance.get('actual_method', mode)}` · "
        f"Device: `{result.provenance.get('device', _device)}` · "
        f"Engine time: `{duration:.0f} ms`",
        f"**Audit:** {audit}",
    ]
    if was_resized:
        details.append(
            f"**Resolution:** processed at {processing_width}×{processing_height} "
            f"({processing_width * processing_height / 1_000_000:.1f} MP) to fit the "
            f"restoration limit, then resized back to {original_width}×{original_height}. "
            "Very fine detail above the processing resolution may not be preserved."
        )
    if chips:
        details.append("\n".join(chips))
    warnings = [warning.message for warning in result.warnings]
    if warnings:
        details.append("\n**Notes:** " + " ".join(warnings))
    if _checkpoint_notice:
        details.append(f"\n**Startup note:** {_checkpoint_notice}")
    return final, "\n\n".join(details)


@spaces.GPU(duration=60)
def _restore_on_gpu(image: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    return _format_result(image, mode)


def restore(image: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    # The classical path does not request or consume ZeroGPU time.
    if mode == "classical":
        return _format_result(image, mode)
    return _restore_on_gpu(image, mode)


def _benchmark_rows() -> list[list[Any]]:
    summary_path = ROOT / "server" / "results" / "benchmark_summary.json"
    if not summary_path.is_file():
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    rows: list[list[Any]] = []
    for table in summary.get("tables", {}).values():
        for entry in table:
            rows.append(
                [
                    entry.get("dataset"),
                    entry.get("mode"),
                    entry.get("ablation"),
                    entry.get("images"),
                    entry.get("paired_images"),
                    entry.get("psnr"),
                    entry.get("ssim"),
                    entry.get("chroma_delta_e"),
                    entry.get("edge_retention"),
                    entry.get("failed_guardrail_runs"),
                    round(entry["latency_ms"], 1) if entry.get("latency_ms") is not None else None,
                ]
            )
    return rows


with gr.Blocks(title="LumenX — Low-Light Restoration") as demo:
    gr.Markdown(
        """
        # LumenX: Low-Light Restoration
        Restore dark images with classical vision, Zero-DCE++, or an adaptive hybrid.
        The classical path runs on CPU. Learned and hybrid runs use shared ZeroGPU.
        """
    )
    with gr.Tab("Enhance"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", image_mode="RGB", label="Low-light input")
                mode = gr.Radio(
                    choices=[
                        ("Adaptive hybrid", "hybrid"),
                        ("Classical vision", "classical"),
                        ("Zero-DCE++", "dl"),
                    ],
                    value="hybrid",
                    label="Restoration path",
                )
                run_button = gr.Button("Restore image", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Enhanced output")
                report = gr.Markdown()
        gr.Examples(
            examples=[
                ["server/gallery/synthetic_night_street.png", "hybrid"],
                ["server/gallery/synthetic_indoor_hall.png", "classical"],
            ],
            inputs=[input_image, mode],
            label="Try a sample scene",
        )
        run_button.click(restore, inputs=[input_image, mode], outputs=[output_image, report])

    with gr.Tab("Benchmarks"):
        gr.Markdown(
            "Paired LOL-v1 and unpaired LIME/NPE evaluation. Unpaired entries have no PSNR/SSIM reference scores."
        )
        gr.Dataframe(
            headers=[
                "dataset",
                "method",
                "ablation",
                "images",
                "paired_images",
                "psnr",
                "ssim",
                "chroma_delta_e",
                "edge_retention",
                "guardrail_failures",
                "latency_ms",
            ],
            value=_benchmark_rows(),
            interactive=False,
            wrap=True,
        )

    gr.Markdown(
        "Zero-DCE++ is provided for non-commercial research and demonstration use. "
        "Avoid uploading sensitive photos to this public demo."
    )

demo.queue(max_size=8).launch()
