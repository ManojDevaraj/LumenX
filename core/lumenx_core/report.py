"""Plain-English run summary built only from measured pipeline outputs."""

from __future__ import annotations

from .contracts import Chip, StoryStep
from .types import ClassicalResult, DiagnosisResult, DenoiseResult, GuardrailResult, RecoveryResult


def _severity(value: float, high: float, medium: float) -> str:
    return "high" if value >= high else "med" if value >= medium else "low"


def build_report(
    diagnosis: DiagnosisResult,
    recovery: RecoveryResult,
    classical: ClassicalResult,
    denoised: DenoiseResult,
    guardrails: GuardrailResult,
    cfg: dict[str, dict[str, float | int]],
) -> tuple[str, list[Chip], list[str], list[StoryStep]]:
    """Create a concise explanation without inferring unmeasured quality."""

    headline = (
        f"{diagnosis.severity.capitalize()} low-light scene; "
        f"brightness was lifted with {len(diagnosis.hotspot_boxes)} bright-light region(s) protected."
    )
    report_cfg = cfg["report"]
    chips = [
        Chip(
            "Mean brightness",
            diagnosis.mean_luminance,
            _severity(
                1.0 - diagnosis.mean_luminance,
                float(report_cfg["brightness_high"]),
                float(report_cfg["brightness_medium"]),
            ),
        ),
        Chip(
            "Clipped highlights",
            diagnosis.highlight_clip_pct,
            _severity(
                diagnosis.highlight_clip_pct,
                float(report_cfg["highlight_high_pct"]),
                float(report_cfg["highlight_medium_pct"]),
            ),
        ),
        Chip(
            "Noise estimate",
            diagnosis.noise_sigma,
            _severity(
                diagnosis.noise_sigma,
                float(report_cfg["noise_high"]),
                float(report_cfg["noise_medium"]),
            ),
        ),
        Chip(
            "Weak-signal area",
            recovery.unrecoverable_pct,
            _severity(
                recovery.unrecoverable_pct,
                float(report_cfg["weak_area_high_pct"]),
                float(report_cfg["weak_area_medium_pct"]),
            ),
        ),
        Chip(
            "Lighting variation",
            diagnosis.illumination_nonuniformity,
            _severity(
                diagnosis.illumination_nonuniformity,
                float(report_cfg["nonuniformity_high"]),
                float(report_cfg["nonuniformity_medium"]),
            ),
        ),
    ]

    decisions = [
        (
            f"The measured mean brightness was {diagnosis.mean_luminance:.3f}; "
            f"an adaptive gamma of {classical.gamma:.2f} lifted darker areas."
        ),
        (
            f"{len(diagnosis.hotspot_boxes)} bright-light region(s) were detected, "
            "so the brightest illumination was protected during the lift."
        ),
        (
            "Brightness varied across the scene, so the lift followed the illumination map."
        ),
        (
            f"Lightness contrast was balanced with CLAHE at clip limit {classical.clahe_clip:.1f}; "
            "a partial colour correction preserved the scene's cast."
        ),
        (
            f"The noise estimate had {diagnosis.noise_confidence:.0%} confidence; "
            f"the final local noise estimate was {denoised.noise_sigma_after:.4f}."
        ),
        recovery.sentence,
    ]
    failed = [check.check.replace("_", " ") for check in guardrails.checks if check.passed is False]
    if failed:
        decisions.append("The audit flagged " + ", ".join(failed) + "; no automatic repair is enabled in this milestone.")
    else:
        decisions.append("Safety checks are shown as an audit; this milestone does not retry or alter a failing result.")

    story = [
        StoryStep(
            "Where light falls",
            "This map shows the brighter and darker parts of the original scene.",
            "illumination",
        ),
        StoryStep(
            "Where signal looks weak",
            "The overlay marks areas with less measured signal and guides a gentler lift.",
            "recoverability_overlay",
        ),
        StoryStep(
            "Balance light and colour",
            "A brightness lift, local contrast adjustment, and partial colour correction shape the preview.",
            "classical",
        ),
        StoryStep(
            "Reduce noise",
            "Smoothing strength varies with the estimated noise and visible edge detail.",
            "denoised",
        ),
        StoryStep(
            "Final image",
            "Compare the result with the original and inspect the measured checks.",
            "final_preview",
        ),
    ]
    return headline, chips, decisions, story


def build_adaptive_report(
    diagnosis: DiagnosisResult,
    recovery: RecoveryResult,
    denoised: DenoiseResult,
    guardrails: GuardrailResult,
    cfg: dict[str, dict[str, float | int]],
    *,
    mode: str,
    params: dict[str, float],
    stages: set[str],
    deep_available: bool,
    repair_attempted: bool,
) -> tuple[str, list[Chip], list[str], list[StoryStep]]:
    """Explain the selected path using only outputs and measurements that exist."""

    report_cfg = cfg["report"]
    chips = [
        Chip("Mean brightness", diagnosis.mean_luminance, _severity(1.0 - diagnosis.mean_luminance, float(report_cfg["brightness_high"]), float(report_cfg["brightness_medium"]))),
        Chip("Clipped highlights", diagnosis.highlight_clip_pct, _severity(diagnosis.highlight_clip_pct, float(report_cfg["highlight_high_pct"]), float(report_cfg["highlight_medium_pct"]))),
        Chip("Noise estimate", diagnosis.noise_sigma, _severity(diagnosis.noise_sigma, float(report_cfg["noise_high"]), float(report_cfg["noise_medium"]))),
        Chip("Weak-signal area", recovery.unrecoverable_pct, _severity(recovery.unrecoverable_pct, float(report_cfg["weak_area_high_pct"]), float(report_cfg["weak_area_medium_pct"]))),
        Chip("Lighting variation", diagnosis.illumination_nonuniformity, _severity(diagnosis.illumination_nonuniformity, float(report_cfg["nonuniformity_high"]), float(report_cfg["nonuniformity_medium"]))),
    ]
    method = "Zero-DCE++" if mode in ("dl", "zerodce") else "adaptive hybrid restoration"
    if mode == "classical":
        method = "adaptive classical restoration"
    elif mode == "hybrid" and "fusion" not in stages:
        if "classical" in stages:
            method = "the classical path with fusion ablated"
        elif deep_available:
            method = "the Zero-DCE++ path with fusion ablated"
        else:
            method = "the classical fallback with fusion ablated"
    elif "deep" in stages and not deep_available:
        method = "classical restoration with a disclosed model fallback"
    headline = (
        f"{diagnosis.severity.capitalize()} low-light scene; {method} was guided by measured noise, "
        f"illumination, and {recovery.unrecoverable_pct:.1f}% weak-signal coverage."
    )
    decisions = [
        f"Mean brightness measured {diagnosis.mean_luminance:.3f}; {len(diagnosis.hotspot_boxes)} bright-light region(s) were protected.",
        recovery.sentence,
    ]
    if "classical" in stages:
        decisions.append(
            f"The classical path used gamma {params.get('gamma', 1.0):.2f}, local contrast {params.get('clahe_clip', 0.0):.1f}, and partial colour correction."
        )
    if "deep" in stages:
        if deep_available:
            decisions.append("Zero-DCE++ estimated a smooth set of local light curves for the image.")
        else:
            decisions.append("Trained Zero-DCE++ weights were unavailable, so the model contribution was disabled.")
    if "fusion" in stages:
        decisions.append("The fusion map balanced the model, classical result, and a conservative highlight-safe path.")
    if "denoise" in stages:
        decisions.append(
            f"Noise-aware smoothing changed the local noise estimate from {diagnosis.noise_sigma:.4f} to {denoised.noise_sigma_after:.4f}."
        )
    failed = [check.check.replace("_", " ") for check in guardrails.checks if check.passed is False]
    if failed:
        action = "one conservative repair was attempted" if repair_attempted else "no repair was applied"
        decisions.append("The final audit still flagged " + ", ".join(failed) + f"; {action}.")
    else:
        decisions.append("The reported safety checks passed or were unavailable for this image.")
    story_labels = {
        "diagnose": ("Where light falls", "This map shows brighter areas, deep shadows, and measured noise.", "illumination"),
        "recoverability": ("Where signal looks weak", "The overlay estimates where the recorded signal looks less reliable.", "recoverability_overlay"),
        "classical": ("Balance light and colour", "Retinex and local contrast shape the classical restoration path.", "classical"),
        "deep": ("Learned light curves", "Zero-DCE++ estimates image-specific curves, when trained weights are available.", "deep"),
        "fusion": ("Combine restoration paths", "The map shows how the model, classical result, and safer path were balanced.", "fused"),
        "denoise": ("Reduce amplified noise", "Smoothing follows the measured noise and visible edge structure.", "denoised"),
    }
    story: list[StoryStep] = []
    for key in ("diagnose", "recoverability", "classical", "deep", "fusion", "denoise"):
        if key in stages:
            title, caption, visual_key = story_labels[key]
            story.append(StoryStep(title, caption, visual_key))
    story.append(StoryStep("Final image", "Compare the result with the original and inspect the measured checks.", "final_preview"))
    return headline, chips, decisions, story
