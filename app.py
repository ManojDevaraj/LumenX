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


_MODE_COPY = {
    "hybrid": ("ADAPTIVE HYBRID", "Balances learned detail with classical colour and highlight control."),
    "classical": ("CLASSICAL VISION", "Retinex, illumination-aware contrast, and edge-preserving denoising."),
    "dl": ("ZERO-DCE++", "Learned low-light curves run through the shared ZeroGPU pool."),
}


def _status_panel(image: np.ndarray | None, mode: str, phase: str = "ready") -> str:
    mode_name, mode_copy = _MODE_COPY.get(mode, _MODE_COPY["hybrid"])
    phase_text = {
        "ready": "READY",
        "processing": "RESTORING",
        "complete": "COMPLETE",
    }.get(phase, "READY")
    phase_class = "is-busy" if phase == "processing" else "is-complete" if phase == "complete" else ""
    image_text = "Upload a photo or choose a sample scene to begin."
    resolution_text = ""
    if image is not None and image.ndim >= 2:
        height, width = image.shape[:2]
        megapixels = width * height / 1_000_000
        image_text = f"{width:,} × {height:,} pixels · {megapixels:.1f} MP"
        if width * height > _MAX_PROCESSING_PIXELS:
            resolution_text = (
                "Large image: it will be reduced to the safe processing size and returned "
                "at its original dimensions."
            )
    return (
        f'<div class="lx-status {phase_class}">'
        f'<div class="lx-status-top"><span class="lx-status-dot"></span>'
        f'<span>{phase_text}</span><span class="lx-status-divider">/</span><span>{mode_name}</span></div>'
        f'<div class="lx-status-copy">{image_text}</div>'
        f'<div class="lx-status-hint">{resolution_text or mode_copy}</div>'
        "</div>"
    )


def _update_status(image: np.ndarray | None, mode: str) -> str:
    return _status_panel(image, mode)


def _processing_status(image: np.ndarray | None, mode: str) -> str:
    return _status_panel(image, mode, "processing")


def _completed_status(image: np.ndarray | None, mode: str) -> str:
    return _status_panel(image, mode, "complete")


UI_CSS = """
:root {
  --lx-black: #080808;
  --lx-surface: #111111;
  --lx-surface-raised: #181313;
  --lx-border: #332021;
  --lx-muted: #aaa09a;
  --lx-red: #f04438;
  --lx-red-deep: #9f1824;
  --lx-yellow: #ffd34f;
  --lx-gold: #f5a623;
}
html, body, .gradio-container {
  background: var(--lx-black) !important;
  color: #f7f2ed !important;
}
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
.gradio-container { padding: 22px clamp(12px, 3vw, 42px) 30px; }
.gradio-container * { box-sizing: border-box; }
.gradio-container .lx-topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 12px 2px 22px; border-bottom: 1px solid rgba(255, 211, 79, .18);
}
.gradio-container .lx-brand { display: flex; align-items: center; gap: 12px; }
.gradio-container .lx-brand-mark {
  width: 42px; height: 42px; display: grid; place-items: center; border-radius: 13px;
  color: #0a0908; font-weight: 950; letter-spacing: -.08em;
  background: linear-gradient(140deg, var(--lx-yellow), var(--lx-gold) 55%, var(--lx-red));
  box-shadow: 0 0 26px rgba(240, 68, 56, .22);
}
.gradio-container .lx-brand-name { font-size: 17px; font-weight: 900; letter-spacing: .2em; }
.gradio-container .lx-brand-sub { display: block; margin-top: 2px; color: var(--lx-muted); font-size: 10px; letter-spacing: .16em; text-transform: uppercase; }
.gradio-container .lx-live-pill {
  display: inline-flex; align-items: center; gap: 9px; padding: 9px 13px; border-radius: 99px;
  border: 1px solid rgba(255, 211, 79, .25); background: rgba(255, 211, 79, .055);
  color: #f3d98b; font: 700 10px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .12em;
}
.gradio-container .lx-pulse-dot, .gradio-container .lx-status-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--lx-yellow);
  box-shadow: 0 0 14px rgba(255, 211, 79, .8); animation: lx-pulse 1.8s infinite;
}
.gradio-container .lx-hero {
  position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between;
  gap: 24px; padding: clamp(24px, 5vw, 56px) 0 30px;
}
.gradio-container .lx-hero:after {
  content: ""; position: absolute; width: 360px; height: 260px; right: 4%; top: -55px; pointer-events: none;
  background: radial-gradient(ellipse, rgba(240, 68, 56, .17), rgba(245, 166, 35, .07) 42%, transparent 72%);
  filter: blur(8px);
}
.gradio-container .lx-eyebrow, .gradio-container .lx-kicker {
  color: var(--lx-yellow); font: 800 10px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
  letter-spacing: .2em; text-transform: uppercase;
}
.gradio-container .lx-hero h1 { max-width: 760px; margin: 10px 0 12px; color: #fff5eb; font-size: clamp(32px, 5vw, 56px); line-height: 1.02; letter-spacing: -.045em; }
.gradio-container .lx-hero h1 span { color: var(--lx-red); text-shadow: 0 0 28px rgba(240, 68, 56, .2); }
.gradio-container .lx-hero p { max-width: 740px; margin: 0; color: #b8ada6; font-size: 15px; line-height: 1.7; }
.gradio-container .lx-hero-orbit {
  z-index: 1; flex: 0 0 150px; width: 150px; height: 150px; display: grid; place-items: center;
  border: 1px solid rgba(255, 211, 79, .2); border-radius: 50%;
  background: radial-gradient(circle, rgba(240, 68, 56, .15), rgba(17, 17, 17, .1) 66%);
  box-shadow: inset 0 0 35px rgba(240, 68, 56, .08), 0 0 50px rgba(240, 68, 56, .08);
}
.gradio-container .lx-orbit-core { width: 56px; height: 56px; border: 2px solid var(--lx-yellow); border-radius: 50%; position: relative; box-shadow: 0 0 24px rgba(255, 211, 79, .35); }
.gradio-container .lx-orbit-core:before, .gradio-container .lx-orbit-core:after { content: ""; position: absolute; border: 1px solid rgba(240, 68, 56, .7); border-radius: 50%; inset: -17px 8px; transform: rotate(45deg); }
.gradio-container .lx-orbit-core:after { inset: 8px -17px; transform: rotate(-45deg); border-color: rgba(255, 211, 79, .5); }
.gradio-container .lx-feature-row { display: flex; flex-wrap: wrap; gap: 8px; padding-bottom: 22px; }
.gradio-container .lx-feature { border: 1px solid #342526; border-radius: 99px; padding: 7px 11px; color: #c6b8ae; background: #100d0d; font-size: 11px; }
.gradio-container .lx-feature b { color: var(--lx-yellow); margin-right: 5px; }
.gradio-container .lx-tabs { border: 1px solid var(--lx-border); border-radius: 18px; background: rgba(17, 17, 17, .78); padding: 8px; box-shadow: 0 22px 70px rgba(0,0,0,.28); }
.gradio-container .tab-nav { border-bottom: 1px solid #332021 !important; padding: 0 8px; }
.gradio-container .tab-nav button { color: #a99d97 !important; font-weight: 750 !important; transition: color .2s, border-color .2s; }
.gradio-container .tab-nav button.selected { color: var(--lx-yellow) !important; border-color: var(--lx-red) !important; }
.gradio-container .lx-panel {
  min-width: 0; padding: 20px; border: 1px solid #332526; border-radius: 15px;
  background: linear-gradient(145deg, rgba(28,20,20,.95), rgba(15,15,15,.98));
  box-shadow: inset 0 1px rgba(255,255,255,.025); transition: border-color .25s, transform .25s, box-shadow .25s;
}
.gradio-container .lx-panel:hover { border-color: rgba(240,68,56,.5); box-shadow: 0 14px 38px rgba(0,0,0,.22), 0 0 24px rgba(240,68,56,.055); }
.gradio-container .lx-section-title { margin: 2px 0 14px; color: #fff3e7; font-size: 16px; font-weight: 850; }
.gradio-container .lx-section-title span { margin-right: 9px; color: var(--lx-red); font: 800 10px ui-monospace, Consolas, monospace; }
.gradio-container .lx-image-input, .gradio-container .lx-image-output { border: 1px solid #3a2928 !important; border-radius: 12px !important; overflow: hidden; }
.gradio-container .lx-image-input { background: #100d0d !important; }
.gradio-container .lx-image-input [data-testid="image"] { min-height: 300px; }
.gradio-container .lx-image-input button, .gradio-container .lx-image-output button { color: var(--lx-yellow) !important; }
.gradio-container .lx-mode-select { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.gradio-container .lx-mode-select label {
  min-height: 64px; justify-content: center; border: 1px solid #3b2929 !important; border-radius: 11px !important;
  background: #100d0d !important; color: #e8ddd4 !important; transition: .2s ease;
}
.gradio-container .lx-mode-select label:hover { border-color: var(--lx-gold) !important; transform: translateY(-2px); }
.gradio-container .lx-mode-select label:has(input:checked) { border-color: var(--lx-red) !important; background: linear-gradient(135deg, rgba(159,24,36,.3), rgba(245,166,35,.08)) !important; box-shadow: 0 0 20px rgba(240,68,56,.1); }
.gradio-container .lx-mode-select input { accent-color: var(--lx-yellow) !important; }
.gradio-container .lx-status {
  margin-top: 14px; padding: 13px 14px; border: 1px solid rgba(255,211,79,.16); border-radius: 11px;
  background: linear-gradient(100deg, rgba(255,211,79,.055), rgba(240,68,56,.045));
}
.gradio-container .lx-status-top { display: flex; align-items: center; gap: 9px; color: var(--lx-yellow); font: 800 10px ui-monospace, Consolas, monospace; letter-spacing: .1em; }
.gradio-container .lx-status-divider { color: #755b52; }
.gradio-container .lx-status-copy { margin-top: 9px; color: #f4e8dc; font-size: 13px; font-weight: 700; }
.gradio-container .lx-status-hint { margin-top: 5px; color: #a99b92; font-size: 11px; line-height: 1.5; }
.gradio-container .lx-status.is-busy { border-color: rgba(240,68,56,.6); animation: lx-border 1s ease-in-out infinite alternate; }
.gradio-container .lx-status.is-busy .lx-status-dot { background: var(--lx-red); box-shadow: 0 0 16px rgba(240,68,56,.9); }
.gradio-container .lx-status.is-complete { border-color: rgba(255,211,79,.55); }
.gradio-container button.primary, .gradio-container .lx-run button {
  min-height: 52px; border: 0 !important; border-radius: 11px !important; color: #140b08 !important;
  background: linear-gradient(105deg, var(--lx-red), #ff633c 48%, var(--lx-yellow)) !important;
  font-weight: 900 !important; letter-spacing: .01em; box-shadow: 0 9px 26px rgba(240,68,56,.19);
  transition: transform .2s, filter .2s, box-shadow .2s;
}
.gradio-container button.primary:hover, .gradio-container .lx-run button:hover { transform: translateY(-2px); filter: saturate(1.12) brightness(1.08); box-shadow: 0 13px 32px rgba(240,68,56,.28); }
.gradio-container .lx-run { margin-top: 12px; }
.gradio-container .lx-report { margin-top: 14px; padding: 4px 14px; border-left: 2px solid var(--lx-gold); border-radius: 0 10px 10px 0; background: rgba(255,211,79,.035); color: #d9cec6; }
.gradio-container .lx-report h3 { color: var(--lx-yellow); }
.gradio-container .lx-examples { margin: 22px 4px 8px; padding: 18px; border: 1px solid #332526; border-radius: 14px; background: #0e0d0d; }
.gradio-container .lx-examples img { border: 1px solid #3a2928; border-radius: 10px; transition: border-color .2s, transform .2s; }
.gradio-container .lx-examples img:hover { border-color: var(--lx-yellow); transform: translateY(-2px); }
.gradio-container .lx-benchmark-intro { padding: 16px; border-radius: 12px; border: 1px solid rgba(255,211,79,.17); background: rgba(255,211,79,.045); color: #cbbfb6; }
.gradio-container .lx-benchmark-table { border: 1px solid #332526; border-radius: 12px; overflow: hidden; }
.gradio-container .lx-benchmark-table th { color: var(--lx-yellow) !important; background: #211516 !important; }
.gradio-container .lx-footer { margin: 22px 4px 0; padding-top: 16px; border-top: 1px solid rgba(255,211,79,.13); color: #8e827b; font-size: 11px; line-height: 1.7; }
.gradio-container .lx-footer strong { color: #d8b75a; }
.gradio-container .lx-workspace { gap: 16px !important; align-items: stretch !important; }
.gradio-container .lx-workspace > .lx-panel { flex: 1 1 0 !important; }
@keyframes lx-pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .45; transform: scale(.8); } }
@keyframes lx-border { from { box-shadow: 0 0 0 rgba(240,68,56,0); } to { box-shadow: 0 0 24px rgba(240,68,56,.17); } }
@media (max-width: 760px) {
  .gradio-container { padding: 12px 10px 20px; }
  .gradio-container .lx-hero { padding: 30px 2px 22px; }
  .gradio-container .lx-hero-orbit { flex-basis: 82px; width: 82px; height: 82px; transform: scale(.8); margin-right: -10px; }
  .gradio-container .lx-panel { padding: 13px; }
  .gradio-container .lx-mode-select { grid-template-columns: 1fr; }
  .gradio-container .lx-mode-select label { min-height: 48px; justify-content: flex-start; }
}
@media (prefers-reduced-motion: reduce) {
  .gradio-container *, .gradio-container *:before, .gradio-container *:after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
"""


with gr.Blocks(title="LumenX — Low-Light Restoration", elem_id="lumenx-app") as demo:
    gr.HTML(
        """
        <div class="lx-topbar">
          <div class="lx-brand">
            <span class="lx-brand-mark">LX</span>
            <div><span class="lx-brand-name">LUMENX</span><span class="lx-brand-sub">Adaptive low-light lab</span></div>
          </div>
          <div class="lx-live-pill"><span class="lx-pulse-dot"></span> LIVE · ZERO GPU</div>
        </div>
        <div class="lx-hero">
          <div>
            <div class="lx-eyebrow">COMPUTATIONAL PHOTOGRAPHY / NIGHT MODE</div>
            <h1>Bring the night<br><span>back into focus.</span></h1>
            <p>Restore dark photos with illumination-aware vision, learned enhancement, or an adaptive blend. Protect colour, edges, and the detail that matters.</p>
          </div>
          <div class="lx-hero-orbit" aria-hidden="true"><div class="lx-orbit-core"></div></div>
        </div>
        <div class="lx-feature-row">
          <span class="lx-feature"><b>01</b> illumination mapping</span>
          <span class="lx-feature"><b>02</b> noise-aware recovery</span>
          <span class="lx-feature"><b>03</b> highlight protection</span>
          <span class="lx-feature"><b>04</b> colour + edge audit</span>
        </div>
        """
    )
    with gr.Tabs(elem_classes=["lx-tabs"]):
        with gr.Tab("Enhance"):
            with gr.Row(elem_classes=["lx-workspace"]):
                with gr.Column(scale=1, min_width=320, elem_classes=["lx-panel"]):
                    gr.HTML('<div class="lx-section-title"><span>01 / SOURCE</span> Add a low-light photo</div>')
                    input_image = gr.Image(
                        type="numpy",
                        image_mode="RGB",
                        label="Low-light input",
                        height=330,
                        elem_classes=["lx-image-input"],
                    )
                    gr.HTML('<div class="lx-section-title"><span>02 / METHOD</span> Choose a restoration path</div>')
                    mode = gr.Radio(
                        choices=[
                            ("Adaptive hybrid", "hybrid"),
                            ("Classical vision", "classical"),
                            ("Zero-DCE++", "dl"),
                        ],
                        value="hybrid",
                        label="Restoration path",
                        show_label=False,
                        elem_classes=["lx-mode-select"],
                    )
                    live_status = gr.HTML(_status_panel(None, "hybrid"))
                    run_button = gr.Button("RESTORE IMAGE  →", variant="primary", elem_classes=["lx-run"])
                with gr.Column(scale=1, min_width=320, elem_classes=["lx-panel"]):
                    gr.HTML('<div class="lx-section-title"><span>03 / RESULT</span> Review the restored image</div>')
                    output_image = gr.Image(
                        label="Enhanced output",
                        height=390,
                        elem_classes=["lx-image-output"],
                    )
                    report = gr.Markdown(elem_classes=["lx-report"])

            input_image.change(_update_status, inputs=[input_image, mode], outputs=[live_status], queue=False)
            mode.change(_update_status, inputs=[input_image, mode], outputs=[live_status], queue=False)
            restore_event = run_button.click(
                _processing_status,
                inputs=[input_image, mode],
                outputs=[live_status],
                queue=False,
            ).then(
                restore,
                inputs=[input_image, mode],
                outputs=[output_image, report],
                show_progress="full",
                api_name="restore",
            )
            restore_event.then(
                _completed_status,
                inputs=[input_image, mode],
                outputs=[live_status],
                queue=False,
            )

            with gr.Column(elem_classes=["lx-examples"]):
                gr.HTML('<div class="lx-section-title"><span>QUICK START</span> Try a sample scene</div>')
                gr.Examples(
                    examples=[
                        ["server/gallery/synthetic_night_street.png", "hybrid"],
                        ["server/gallery/synthetic_indoor_hall.png", "classical"],
                    ],
                    inputs=[input_image, mode],
                    label="",
                )

        with gr.Tab("Benchmarks"):
            gr.HTML(
                '<div class="lx-benchmark-intro"><span class="lx-kicker">MEASURED RESULTS</span><br>'
                "Paired LOL-v1 and unpaired LIME/NPE evaluation. Unpaired rows have no PSNR/SSIM reference scores.</div>"
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
                elem_classes=["lx-benchmark-table"],
            )

    gr.HTML(
        '<div class="lx-footer"><strong>MODEL NOTE</strong> Zero-DCE++ is provided for non-commercial research and demonstration. '
        "Enhancement runs on the uploaded image; avoid sending sensitive photos to this public demo.</div>"
    )

demo.queue(max_size=8).launch(css=UI_CSS)
