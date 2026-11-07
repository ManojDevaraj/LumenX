"""Custom metric correctness with reference values and genuinely unavailable cases."""

from __future__ import annotations

import numpy as np
import pytest

from lumenx_core import create_engine
from lumenx_core.config import load_config
from lumenx_core.diagnose import diagnose
from lumenx_core.metrics import chroma_delta_e, request_metrics


@pytest.mark.parametrize("first,second,expected", [
    ([2.6772, -79.7751], [0, -82.7485], 2.0425),
    ([3.1571, -77.2803], [0, -82.7485], 2.8615),
    ([2.8361, -74.0200], [0, -82.7485], 3.4412),
    ([-1.3802, -84.2814], [0, -82.7485], 1.0000),
])
def test_chroma_ciede2000_reference_pairs(first, second, expected):
    """Sharma/Wu/Dalal published pairs with equal L*=50 isolate the chroma terms."""
    assert float(chroma_delta_e(np.array(first), np.array(second))) == pytest.approx(expected, abs=5e-5)
    assert float(chroma_delta_e(np.array(second), np.array(first))) == pytest.approx(expected, abs=5e-5)


def test_chroma_identity_gray_and_hue_wrap():
    """Zero chroma, equal hues and +/-pi wrap remain finite and symmetric."""
    first = np.array([[0, 0], [-0.001, 20], [0.001, -20], [10, 10]], float)
    second = np.array([[0, 0], [0.001, 20], [-0.001, -20], [10, 10]], float)
    a, b = chroma_delta_e(first, second), chroma_delta_e(second, first)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert a[0] == a[3] == 0


def test_real_metric_suite_clipping_and_psnr():
    """Actual statistics have the right direction and nulls always have reasons."""
    suite = create_engine().metric_suite()
    image = np.full((96, 96, 3), 64, np.uint8)
    output = image.copy()
    output[:24] = 255
    values = suite.custom(image, output)
    assert values["clip_pct_in"] == 0
    assert values["clip_pct_out"] == 25
    assert values["edge_retention"] is None
    assert suite.unavailable()["edge_retention"]
    assert suite.paired(image, image)["psnr"] is None
    assert suite.paired(image, output)["psnr"] > 0
    for key, value in suite.unpaired(image).items():
        if value is None:
            assert suite.unavailable()[key]
        else:
            assert np.isfinite(value)


def test_chroma_metric_ignores_gray_brightness_lift():
    """Lightness changes alone do not trigger the chroma guardrail."""
    suite = create_engine().metric_suite()
    first = np.full((96, 96, 3), 64, np.uint8)
    last = np.full_like(first, 128)
    assert suite.custom(first, last)["chroma_delta_e"] == pytest.approx(0, abs=0.1)


def test_request_noise_metrics_remain_at_native_resolution():
    """Request metrics must not report the lower sigma created by a 512px proxy."""
    cfg = load_config()
    rng = np.random.default_rng(61)
    image = np.clip(0.08 + rng.normal(0, 0.015, (1080, 1920, 3)), 0, 1).astype(np.float32)
    diagnosis = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    metrics = request_metrics(np.rint(image * 255).astype(np.uint8), np.rint(image * 255).astype(np.uint8),
                              np.ones(image.shape[:2], bool), cfg)
    assert metrics["noise_sigma"]["before"] == pytest.approx(diagnosis.raw["noise_sigma"], rel=0.12)
