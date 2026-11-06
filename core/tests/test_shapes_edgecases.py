"""Real v0.2 release cases: meaningful input evidence and full public-contract checks."""

from __future__ import annotations

from dataclasses import replace
import json
import threading
import weakref

import numpy as np
import pytest

from lumenx_core import create_engine
from lumenx_core.config import load_config
from lumenx_core.contract_check import validate_run
from lumenx_core.contracts import Cancelled, EngineError, EngineUnavailable, EnhanceRequest, InvalidRequest
from lumenx_core.diagnose import diagnose, luminance
from lumenx_core.recoverability import enforce_gain, estimate_recoverability


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (17, 29), (64, 96)])
@pytest.mark.parametrize("kind", ["black", "flat", "bright", "noisy", "red", "gray"])
def test_real_classical_shapes(shape, kind):
    """Real classical processing returns valid finite arrays for pathological inputs."""
    h, w = shape
    if kind == "noisy":
        image = np.clip(np.random.default_rng(7).normal(0.08, 0.02, (h, w, 3)), 0, 1).astype(np.float32)
    else:
        image = np.full((h, w, 3), {"black": 0, "flat": 0.08, "bright": 0.7, "red": 0, "gray": 0.3}[kind], np.float32)
        if kind == "red":
            image[..., 0] = 0.9
    alpha = np.linspace(0, 1, h * w, dtype=np.float32).reshape(shape)
    req = EnhanceRequest(image, (w, h), alpha=alpha, mode="classical")
    engine = create_engine()
    items = list(engine.run(req))
    result = items[-1]
    assert validate_run(items, req) == []
    assert result.provenance["actual_method"] == "classical"
    assert result.provenance.get("fake", False) is False
    final = result.visuals[0].image
    if kind == "black":
        assert not np.any(final[..., :3])
        assert result.recoverability["unrecoverable_pct"] == 100
    if kind == "bright":
        expected = np.rint(image * 255).astype(np.uint8)
        np.testing.assert_array_equal(final[alpha > 0, :3], expected[alpha > 0])
    assert abs(sum(result.recoverability[k] for k in ("recovered_pct", "estimated_pct", "unrecoverable_pct")) - 100) <= 0.1
    json.dumps(result.metrics, allow_nan=False)
    json.dumps(result.provenance, allow_nan=False)


def test_saturated_colour_not_forced_unrecoverable():
    """Joint darkness is required; two zero channels do not erase a strong red signal."""
    image = np.zeros((192, 192, 3), np.float32)
    image[..., 0] = 0.9
    cfg = load_config()
    diagnosis = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    recovery = estimate_recoverability(diagnosis, cfg)
    assert diagnosis.raw["noise_estimator_confidence"] == 1
    assert recovery.summary["recovered_pct"] > 99


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)])
@pytest.mark.parametrize("color", [[0.08, 0.08, 0.08], [0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]])
def test_opaque_tiny_colours(shape, color):
    """Exercise true tiny colour correction, not just fully transparent tiny fixtures."""
    image = np.tile(np.array(color, np.float32), (*shape, 1))
    req = EnhanceRequest(image, (shape[1], shape[0]), mode="classical")
    assert validate_run(list(create_engine().run(req)), req) == []


def test_neutral_gray_stays_neutral_and_raw_rgb_channels_survive():
    """RGB clamping must apply to every channel, including OpenCV scalar corner cases."""
    engine = create_engine()
    image = np.full((96, 96, 3), 0.08, np.float32)
    req = EnhanceRequest(image, (96, 96), mode="classical")
    final = list(engine.run(req))[-1].visuals[0].image.astype(int)
    assert np.max(np.max(final, axis=2) - np.min(final, axis=2)) <= 2
    assert all(float(final[..., channel].mean()) > 20 for channel in range(3))


def test_gain_cap_applies_to_actual_output():
    """Even a deliberately overbright RGB candidate must obey per-pixel luminance caps."""
    cfg = load_config()
    image = np.full((96, 96, 3), 0.005, np.float32)
    image[32:64, 32:64] = 0
    diagnosis = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    recovery = estimate_recoverability(diagnosis, cfg)
    capped = enforce_gain(np.ones_like(image), diagnosis.luminance, recovery, cfg)
    assert np.all(luminance(capped, cfg) <= diagnosis.luminance * recovery.gain_cap + 1e-7)
    assert np.all(capped[32:64, 32:64] == 0)


def test_native_noise_estimator_detects_noise_removed_by_downsampling():
    """Noise estimate must follow native input patches, not the smoothed 512 proxy."""
    cfg = load_config()
    rng = np.random.default_rng(17)
    image = np.clip(0.1 + rng.normal(0, 0.01, (768, 768, 3)), 0, 1).astype(np.float32)
    d = diagnose(image, np.ones(image.shape[:2], bool), cfg)
    expected = 0.01 * np.linalg.norm(cfg["real"]["luminance_weights"])
    assert d.raw["noise_sigma"] == pytest.approx(expected, rel=0.15)
    assert 0 < d.raw["noise_patch_count"] <= 100
    assert d.raw["noise_estimator_confidence"] > 0.9


def test_transparent_rgb_cannot_drive_diagnosis_and_input_is_unchanged():
    """Hidden arbitrary RGB is excluded from measurements; supplied arrays remain untouched."""
    engine = create_engine()
    image = np.full((64, 96, 3), 0.05, np.float32)
    alpha = np.ones(image.shape[:2], np.float32)
    alpha[:, :48] = 0
    first = image.copy()
    first[:, :48] = [1, 0, 1]
    second = image.copy()
    second[:, :48] = [0, 1, 0]
    original = first.copy()
    a = list(engine.run(EnhanceRequest(first, (96, 64), alpha=alpha, mode="classical")))[-1]
    b = list(engine.run(EnhanceRequest(second, (96, 64), alpha=alpha, mode="classical")))[-1]
    assert a.diagnosis_raw == b.diagnosis_raw
    np.testing.assert_array_equal(a.visuals[0].image, b.visuals[0].image)
    np.testing.assert_array_equal(first, original)


def test_all_transparent_contract_and_no_invented_measurements():
    """Empty visible area has an explicit convention and no claimed clipping/colour data."""
    image = np.ones((7, 13, 3), np.float32)
    req = EnhanceRequest(image, (13, 7), alpha=np.zeros((7, 13), np.float32), mode="classical")
    items = list(create_engine().run(req))
    assert validate_run(items, req) == []
    assert items[-1].diagnosis_raw["empty_visible_area"] is True
    assert items[-1].metrics["clip_pct"]["before"] is None


def test_bright_window_does_not_hide_dark_subject():
    """Brightness identity requires spatial statistics, not a mean-only threshold."""
    image = np.full((96, 96, 3), 0.02, np.float32)
    image[:50] = 1
    req = EnhanceRequest(image, (96, 96), mode="classical")
    result = list(create_engine().run(req))[-1]
    assert float(image.mean()) > 0.5
    assert not result.diagnosis_raw["already_bright"]
    assert len(result.hotspots) > 0


def test_real_determinism_and_config_isolation():
    """Repeated calls give identical arrays without mutable public config state."""
    engine = create_engine()
    image = np.random.default_rng(8).uniform(0.02, 0.2, (95, 127, 3)).astype(np.float32)
    req = EnhanceRequest(image, (127, 95), mode="classical", seed=-5)
    a, b = list(engine.run(req))[-1], list(engine.run(req))[-1]
    np.testing.assert_array_equal(a.visuals[0].image, b.visuals[0].image)
    assert a.provenance == b.provenance
    public = engine.public_config()
    public["real"]["classical"]["gamma"]["severe"] = 1
    assert engine.public_config()["real"]["classical"]["gamma"]["severe"] != 1
    with pytest.raises(TypeError):
        load_config()["real"]["threads"] = 1


@pytest.mark.parametrize("mode", ["hybrid"])
def test_unreleased_modes_fail_before_yield(mode):
    """The unavailable tuner is not silently replaced by an unreported no_tuner flag."""
    image = np.full((8, 8, 3), 0.1, np.float32)
    stream = create_engine().run(EnhanceRequest(image, (8, 8), mode=mode))
    with pytest.raises(EngineUnavailable):
        next(stream)
    assert list(stream) == []


def test_real_validation_cancellation_and_early_close():
    """Real engine obeys the same no-partial-invalid-input and cleanup rules."""
    image = np.full((32, 32, 3), 0.1, np.float32)
    req = EnhanceRequest(image, (32, 32), mode="classical")
    engine = create_engine()
    with pytest.raises(InvalidRequest):
        next(engine.run(replace(req, image=image.astype(np.float64))))
    cancel = threading.Event()
    stream = engine.run(replace(req, cancel=cancel))
    event = next(stream)
    reference = weakref.ref(event.visuals[0].image)
    cancel.set()
    del event
    with pytest.raises(Cancelled):
        next(stream)
    assert reference() is None
    stream = engine.run(req)
    event = next(stream)
    del event
    stream.close()
    assert stream.gi_frame is None
    assert validate_run(list(engine.run(req)), req) == []


def test_warmup_and_real_live_unavailability():
    """Health reports actual available modes, and live never becomes a gamma mock."""
    engine = create_engine()
    assert not engine.health()["warm"]
    engine.warmup()
    engine.warmup()
    assert engine.health()["warm"]
    assert "classical" in engine.health()["modes"]
    with pytest.raises(EngineUnavailable):
        engine.enhance_frame(np.zeros((8, 8, 3), np.uint8))
    assert create_engine(device="cuda:0").health()["device"] == "cpu"


@pytest.mark.parametrize("text", ["[]", "unknown: 1", "real: {threads: 0}", "real: {epsilon: .nan}",
                                 "real: {recoverability: {snr_high: 1}}", "real: {diagnosis: {noise_samples: 101}}",
                                 "real: [", "real: {denoise: {filter_scale: 0.0}}",
                                 "real: {classical: {color_scale: 0.0}}"])
def test_invalid_config_rejected_before_processing(tmp_path, text):
    """Malformed overrides produce an engine error at factory time, not mid-stream."""
    source = tmp_path / "bad.yaml"
    source.write_text(text)
    with pytest.raises(EngineError):
        create_engine(str(source))
