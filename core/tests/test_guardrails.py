"""Real v0.2 checks audit failures without inventing a repair or mutating output."""

from __future__ import annotations

import numpy as np

from lumenx_core.config import load_config
from lumenx_core.diagnose import diagnose
from lumenx_core.guardrails import audit


def test_audit_detects_injected_failures_without_repair():
    """Clipping, colour and edge failures retain separate thresholds and audit-only actions."""
    cfg = load_config()
    image = np.full((96, 128, 3), 0.1, np.float32)
    diagnosis = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    metrics = {"clip_pct": {"before": 0, "after": 20}, "chroma_delta_e": 30,
               "edge_retention": 0.1, "unavailable": {}}
    original = image.copy()
    checks = audit(metrics, image, diagnosis, cfg)
    lookup = {check.check: check for check in checks}
    for name in ("overexposure", "color_shift", "detail_loss"):
        assert lookup[name].passed is False
        assert lookup[name].action is None and lookup[name].attempt == 1
    assert lookup["halo"].passed is None and lookup["halo"].reason
    np.testing.assert_array_equal(image, original)


def test_existing_clipping_is_not_mistaken_for_new_damage():
    """Audit compares both absolute output clipping and change from input."""
    cfg = load_config()
    image = np.full((64, 96, 3), 0.1, np.float32)
    diagnosis = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    metrics = {"clip_pct": {"before": 25, "after": 25}, "chroma_delta_e": None,
               "edge_retention": None, "unavailable": {"chroma_delta_e": "No midtones", "edge_retention": "No edges"}}
    checks = audit(metrics, image, diagnosis, cfg)
    assert checks[0].passed is True and checks[0].value == 0
    assert checks[1].passed is None and checks[1].reason


def test_halo_ring_detects_new_bright_outline():
    """A new bright rim outside a lamp should produce positive localized overshoot."""
    cfg = load_config()
    image = np.full((128, 128, 3), 0.02, np.float32)
    image[48:80, 48:80] = 1
    diagnosis = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    damaged = image.copy()
    damaged[45:83, 45:83] = 1
    metrics = {"clip_pct": {"before": 0, "after": 0}, "chroma_delta_e": 0,
               "edge_retention": None, "unavailable": {"edge_retention": "Not needed"}}
    check = audit(metrics, damaged, diagnosis, cfg)[-1]
    assert check.value is not None and check.value > 0
    assert check.passed is False
