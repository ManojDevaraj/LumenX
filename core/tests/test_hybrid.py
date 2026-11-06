"""Contract and repair integration using explicit unit stubs, not claimed GPU evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lumenx_core import create_engine
from lumenx_core.contract_check import validate_run
from lumenx_core.contracts import EnhanceRequest, GuardrailCheck
from lumenx_core.types import DeepResult


def make_request(mode="hybrid", flags=frozenset({"no_tuner"})):
    """Build a deterministic legal request; real tuning is not present at this milestone."""
    image = np.full((64, 96, 3), 0.05, np.float32)
    return EnhanceRequest(image, (96, 64), mode=mode, ablate=flags if mode == "hybrid" else frozenset())


def stub_model(engine, monkeypatch):
    """Replace inference only; all diagnosis/fusion/denoise/audit code remains real."""
    calls = []

    def infer(image, severity, **kwargs):
        calls.append((severity, kwargs))
        return DeepResult(np.full_like(image, 0.3), "cidnet", "cidnet_lolv2_real",
                          "0123456789ab", "cpu", "fp32", [], [])

    monkeypatch.setattr(engine._registry, "infer", infer)
    return calls


def check_set(failed=False):
    """Inject a controlled clipping failure without relying on accidental image artifacts."""
    return [GuardrailCheck("overexposure", 10.0 if failed else 0.0, 0.5, not failed, 1, None, None),
            GuardrailCheck("color_shift", 0.0, 12.0, True, 1, None, None),
            GuardrailCheck("detail_loss", None, 0.7, None, 1, None, "No supported edges"),
            GuardrailCheck("halo", None, 0.08, None, 1, None, "No hotspot ring")]


@pytest.mark.parametrize("flags", [frozenset({"no_tuner"}), frozenset({"no_tuner", "no_dl"}),
                                  frozenset({"no_tuner", "no_classical"}), frozenset({"no_tuner", "no_fusion"}),
                                  frozenset({"no_tuner", "no_recoverability"}), frozenset({"no_tuner", "no_gain_denoise"}),
                                  frozenset({"no_tuner", "no_guardrails"}), frozenset({"no_tuner", "no_color_correction"})])
def test_no_tuner_hybrid_contract_and_ablation_schedules(flags, monkeypatch):
    """Every component ablation has the frozen schedule and correct model call count."""
    engine = create_engine(device="cpu")
    calls = stub_model(engine, monkeypatch)
    req = make_request(flags=flags)
    items = list(engine.run(req))
    assert validate_run(items, req) == []
    assert len(calls) == (0 if "no_dl" in flags else 1)
    assert items[-1].candidates == [] and items[-1].search_ms is None
    assert items[-1].provenance["ablate"] == sorted(flags)


@pytest.mark.parametrize("mode", ["dl", "zerodce"])
def test_direct_baseline_preserves_model_output(mode, monkeypatch):
    """No post-denoising, fusion or repair is permitted on direct baseline output."""
    engine = create_engine(device="cpu")
    calls = stub_model(engine, monkeypatch)
    req = make_request(mode=mode)
    items = list(engine.run(req))
    assert validate_run(items, req) == []
    assert len(calls) == 1
    np.testing.assert_array_equal(items[-1].visuals[0].image, np.full(req.image.shape, 76, np.uint8))
    assert not items[-1].provenance["repair_attempted"]


def test_repair_reuses_deep_once_and_runs_one_retry(monkeypatch):
    """The second guardrail attempt must not trigger a second model execution."""
    engine = create_engine(device="cpu")
    calls = stub_model(engine, monkeypatch)
    audits = []

    def audit(*args):
        audits.append(1)
        return check_set(failed=len(audits) == 1)

    monkeypatch.setattr("lumenx_core.engine.audit", audit)
    req = make_request()
    items = list(engine.run(req))
    result = items[-1]
    assert validate_run(items, req) == []
    assert len(calls) == 1 and len(audits) == 2
    assert result.provenance["repair_attempted"]
    assert [c.attempt for c in result.guardrails] == [1] * 4 + [2] * 4


def test_unresolved_repair_returns_recorded_conservative_result(monkeypatch):
    """The fallback has independent checks; no third repair or model call happens."""
    engine = create_engine(device="cpu")
    calls = stub_model(engine, monkeypatch)
    monkeypatch.setattr("lumenx_core.engine.audit", lambda *args: check_set(failed=True))
    req = make_request()
    result = list(engine.run(req))[-1]
    assert len(calls) == 1
    assert len(result.guardrails) == 8
    assert result.provenance["conservative_final_checks"]
    assert "overexposure" in result.provenance["guardrails_unresolved"]
    assert any(w.code == "guardrail_unresolved" for w in result.warnings)
    assert float(result.visuals[0].image.mean()) < 76


def test_missing_weights_report_fallback_instead_of_claiming_model_inference():
    """The CPU weight-free suite verifies graceful degradation, not trained quality."""
    engine = create_engine(device="cpu")
    req = make_request(mode="dl")
    items = list(engine.run(req))
    assert validate_run(items, req) == []
    result = items[-1]
    assert result.provenance["actual_method"] == "classical_fallback"
    assert result.provenance["checkpoint"] is None
    assert result.provenance["fallback_chain"][-1] == "classical"


def test_ablation_flags_do_not_change_input_request(monkeypatch):
    """Core must never silently add no_tuner to make a request pass validation."""
    engine = create_engine(device="cpu")
    stub_model(engine, monkeypatch)
    req = make_request()
    list(engine.run(req))
    assert req.ablate == frozenset({"no_tuner"})
    modified = replace(req, seed=7)
    assert modified.ablate == req.ablate


@pytest.mark.parametrize("shape", [(720, 1280), (1080, 1920)])
@pytest.mark.parametrize("kind", ["flat", "noise"])
def test_tuned_hybrid_large_nonblack_cpu_contract(shape, kind, monkeypatch):
    """Proxy tuning retains a two-dimensional valid mask at 720p and 1080p."""
    rng = np.random.default_rng(52)
    image = np.full((*shape, 3), 0.05, np.float32)
    if kind == "noise":
        image += rng.uniform(0, 0.1, image.shape).astype(np.float32)
    engine = create_engine(device="cpu")
    stub_model(engine, monkeypatch)
    req = EnhanceRequest(image, (shape[1], shape[0]), mode="hybrid")
    items = list(engine.run(req))
    assert validate_run(items, req) == []
    assert len(items[-1].candidates) == 12
