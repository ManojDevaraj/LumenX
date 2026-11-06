"""Numerical hybrid fusion tests independent of model weights or CUDA."""

from __future__ import annotations

import numpy as np

from lumenx_core.config import load_config
from lumenx_core.diagnose import diagnose
from lumenx_core.fusion import fuse
from lumenx_core.recoverability import estimate_recoverability


def test_weights_normalized_and_confidence_caps_deep_contribution():
    """No smoothing/normalization step may amplify deep weights beyond confidence."""
    cfg = load_config()
    image = np.full((96, 128, 3), 0.03, np.float32)
    image[30:50, 80:100] = 1
    image[50:80, 10:40] = 0
    d = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    r = estimate_recoverability(d, cfg)
    result = fuse(image, np.ones_like(image), np.full_like(image, 0.1), d, r, cfg)
    np.testing.assert_allclose(result.weights.sum(axis=2), 1, atol=1e-5)
    assert np.min(result.weights) >= 0
    assert np.all(result.weights[..., 0] <= cfg["fusion"]["alpha"] * r.soft + 1e-6)
    assert np.all(result.weights[30:50, 80:100, 2] == 1)
    assert np.all(result.weights[50:80, 10:40, 0] == 0)


def test_missing_contributors_go_to_safe_path_and_repair_reduces_deep():
    """Unavailable contributors never receive weights; repair is spatial and conservative."""
    cfg = load_config()
    image = np.full((128, 128, 3), 0.15, np.float32)
    d = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    r = estimate_recoverability(d, cfg)
    deep = np.full_like(image, 0.5)
    baseline = fuse(image, deep, None, d, r, cfg)
    mask = np.zeros(image.shape[:2], np.float32)
    mask[32:64, 32:64] = 1
    repaired = fuse(image, deep, None, d, r, cfg, alpha_fraction=0.5, repair_mask=mask)
    assert np.all(baseline.weights[..., 1] == 0)
    assert np.all(repaired.weights[..., 0] <= baseline.weights[..., 0] + 1e-6)
    assert np.mean(repaired.weights[32:64, 32:64, 2]) > np.mean(repaired.weights[:16, :16, 2])
    safe = fuse(image, None, None, d, r, cfg)
    np.testing.assert_allclose(safe.weights[..., 2], 1)


def test_no_recoverability_disables_deep_confidence_modulation():
    """An ablated control must actually change weights, not merely hide its overlay."""
    cfg = load_config()
    image = np.full((96, 96, 3), 0.005, np.float32)
    d = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    r = estimate_recoverability(d, cfg)
    deep, classical = np.full_like(image, 0.3), np.full_like(image, 0.02)
    enabled = fuse(image, deep, classical, d, r, cfg)
    disabled = fuse(image, deep, classical, d, r, cfg, recovery_enabled=False)
    assert np.max(enabled.weights[..., 0]) == 0
    assert np.min(disabled.weights[..., 0]) > 0
