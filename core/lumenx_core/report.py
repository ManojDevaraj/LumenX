"""Plain-English classical reports assembled only from actual measured decisions."""

from __future__ import annotations

from .contracts import Chip, GuardrailCheck
from .types import ClassicalResult, DiagnosisResult, RecoveryResult


def diagnosis_chips(diagnosis: DiagnosisResult, recovery: RecoveryResult) -> list[Chip]:
    """Share factual diagnosis chips across classical, direct and hybrid outputs."""
    raw = diagnosis.raw
    severity = raw["severity"]
    level = {"extreme": "high", "severe": "high", "moderate": "med", "mild": "low"}[severity]
    return [Chip(f"{severity.capitalize()} low light", raw["median_y"], level),
            Chip("Weak recorded light", recovery.summary["unrecoverable_pct"], "med"),
            Chip("Noise estimate support", raw["noise_estimator_confidence"], "low" if raw["noise_estimator_confidence"] else "high")]


def classical_report(diagnosis: DiagnosisResult, recovery: RecoveryResult, classical: ClassicalResult,
                     checks: list[GuardrailCheck]) -> tuple[str, list[Chip], list[str]]:
    """Return headline, severity chips and four to seven evidence-backed decisions."""
    raw = diagnosis.raw
    severity = raw["severity"]
    if raw["empty_visible_area"]:
        headline = "No visible pixels were supplied; transparency was preserved."
    elif raw["already_bright"]:
        headline = "This image is already well lit; its appearance was preserved."
    else:
        headline = f"{severity.capitalize()} low light with {len(diagnosis.hotspots)} bright areas; shadow brightening was limited by recorded signal."
    chips = diagnosis_chips(diagnosis, recovery)
    decisions = [
        "Used classical image processing; no trained restoration model was loaded.",
        ("Kept the already-bright appearance unchanged." if raw["already_bright"]
         else f"Chose a brightness curve from the measured {severity} lighting level."),
        "Limited actual brightening where recorded signal confidence was weak.",
        ("Applied partial colour correction to reduce the measured cast." if classical.params.color_strength
         else "Preserved scene colour because no supported correction was needed."),
        ("Checked sensor noise using flat patches at the original processing resolution." if raw["noise_patch_count"]
         else "Used a low-confidence noise estimate because suitable flat patches were unavailable."),
        "Audited quality without attempting automatic repairs in this release.",
    ]
    failed = [check.check for check in checks if check.passed is False]
    if failed:
        decisions.append("Some quality checks failed; the result includes those warnings for review.")
    return headline, chips, decisions
