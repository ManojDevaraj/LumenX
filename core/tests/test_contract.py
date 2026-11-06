"""Release gate: frozen types, all modes/flags, visuals, and adversarial validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import inspect
import itertools
import json
from pathlib import Path
import re
import threading
import weakref

import numpy as np
import pytest

import lumenx_core
from lumenx_core import create_engine, create_fake_engine
from lumenx_core.contract_check import validate_run
from lumenx_core.contracts import (
    ABLATION_FLAGS,
    CONTRACT_VERSION,
    Cancelled,
    EnhanceRequest,
    EnhanceResult,
    StageEvent,
    Visual,
)


def request(mode: str = "hybrid", ablate: frozenset[str] = frozenset(), **kwargs) -> EnhanceRequest:
    """Build a nontrivial deterministic odd-size fixture without global RNG state."""
    image = np.random.default_rng(42).uniform(0, 0.5, (19, 31, 3)).astype(np.float32)
    return EnhanceRequest(image, (31, 19), mode=mode, ablate=ablate, **kwargs)


def result_of(items: list) -> EnhanceResult:
    """Require and return the single terminal result."""
    assert isinstance(items[-1], EnhanceResult)
    assert sum(isinstance(x, EnhanceResult) for x in items) == 1
    return items[-1]


@pytest.mark.parametrize("mode", ["hybrid", "classical", "dl", "zerodce"])
def test_each_mode_passes_contract(mode):
    """Every publicly supported mode must satisfy the independent checker."""
    req = request(mode)
    engine = create_fake_engine()
    stream = engine.run(req)
    assert inspect.isgenerator(stream)
    items = list(stream)
    assert validate_run(items, req) == []
    result = result_of(items)
    assert result.provenance["core_version"] == lumenx_core.__version__
    assert result.provenance["contract_version"] == CONTRACT_VERSION == "1.0"
    assert len(result.candidates) == (12 if mode == "hybrid" else 0)
    assert len(result.decisions) in range(4, 8)


def legal_flags():
    """Enumerate every legal combination, including interaction cases."""
    flags = sorted(ABLATION_FLAGS)
    for n in range(len(flags) + 1):
        for entries in itertools.combinations(flags, n):
            selected = frozenset(entries)
            if "no_dl" in selected and selected & {"no_fusion", "no_classical"}:
                continue
            yield selected


@pytest.mark.parametrize("flags", list(legal_flags()), ids=lambda flags: ",".join(sorted(flags)) or "none")
def test_every_legal_ablation_combination(flags):
    """Cross-component flag combinations cannot produce a malformed event stream."""
    req = request(ablate=flags)
    assert validate_run(list(create_fake_engine().run(req)), req) == []


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (17, 29)])
@pytest.mark.parametrize("level", [0.0, 0.2, 1.0])
def test_tiny_flat_bright_and_alpha(shape, level):
    """All core visual geometry and alpha values survive pathological small inputs."""
    height, width = shape
    image = np.full((height, width, 3), level, np.float32)
    alpha = np.linspace(0, 1, height * width, dtype=np.float32).reshape(shape)
    req = EnhanceRequest(image, (width, height), alpha=alpha)
    items = list(create_fake_engine().run(req))
    assert validate_run(items, req) == []
    final = result_of(items).visuals[0].image
    assert final.shape == (height, width, 4)
    np.testing.assert_array_equal(final[..., 3], np.rint(alpha * 255).astype(np.uint8))
    np.testing.assert_array_equal(req.image, image)


def test_determinism_no_mutation_and_input_hash():
    """Reuse is deterministic and neither requests nor public config alias engine state."""
    engine = create_fake_engine(seed=12)
    req = request(seed=77)
    original = req.image.copy()
    first = result_of(list(engine.run(req)))
    second = result_of(list(engine.run(req)))
    np.testing.assert_array_equal(first.visuals[0].image, second.visuals[0].image)
    assert first.candidates == second.candidates
    assert first.provenance == second.provenance
    np.testing.assert_array_equal(req.image, original)
    config = engine.public_config()
    config["fake"]["deep_gamma"] = 100
    assert engine.public_config()["fake"]["deep_gamma"] != 100
    different = replace(req, image=np.ascontiguousarray(np.clip(req.image + 0.01, 0, 1)))
    third = result_of(list(engine.run(different)))
    assert first.provenance["input_hash"] != third.provenance["input_hash"]


def test_frozen_contract_source_is_verbatim():
    """Protect exact copied signatures; this check is useful from a source checkout."""
    root = Path(__file__).resolve().parents[2]
    frozen = root / "CONTRACT.md"
    assert frozen.is_file(), "Release-gate tests run from a source checkout with CONTRACT.md."
    expected = re.search(r"```python\n(.*?)```", frozen.read_text(encoding="utf-8"), re.S).group(1)
    source = (root / "core/lumenx_core/contracts.py").read_text(encoding="utf-8")
    assert sha256(source.encode()).digest() == sha256(expected.encode()).digest()


def test_health_config_and_warmup():
    """Mock health must not claim the GPU or real models exist."""
    engine = create_fake_engine()
    health = engine.health()
    assert health["fake"] is True and health["device"] == "cpu"
    assert health["ready"] is True and health["hybrid_ready"] is False
    assert health["warm"] is False
    assert health["models"]["cidnet"]["available"] is False
    assert health["models"]["zerodce"]["available"] is False
    engine.warmup()
    engine.warmup()
    assert engine.health()["warm"] is True
    json.dumps(health, allow_nan=False)
    json.dumps(engine.public_config(), allow_nan=False)
    assert create_engine().health()["fake"] is False


def test_frame_and_metric_suite():
    """Live mock output is deterministic; metrics never invent learned scores."""
    engine = create_fake_engine()
    inp = np.full((480, 640, 3), 20, np.uint8)
    before = inp.copy()
    out = engine.enhance_frame(inp)
    assert out.shape == inp.shape and out.dtype == np.uint8
    assert float(out.mean()) > float(inp.mean())
    np.testing.assert_array_equal(inp, before)
    suite = engine.metric_suite()
    paired = suite.paired(out, inp)
    expected_mse = float(np.mean((out.astype(float) - inp.astype(float)) ** 2))
    assert paired["psnr"] == pytest.approx(10 * np.log10(255**2 / expected_mse))
    assert suite.paired(inp, inp)["psnr"] is None
    assert set(suite.unpaired(out)) == {"niqe", "brisque", "clipiqa", "musiq"}
    assert all(v is None for v in suite.unpaired(out).values())
    assert suite.directions()["niqe"] == "lower"
    assert suite.directions()["psnr"] == "higher"
    assert set(suite.custom(inp, out)) == {
        "clip_pct_in", "clip_pct_out", "chroma_delta_e", "edge_retention",
        "noise_sigma_in", "noise_sigma_out",
    }
    for mapping in (paired, suite.unpaired(out), suite.custom(inp, out),
                    suite.confidence_errors(inp, out, inp)):
        json.dumps(mapping, allow_nan=False)
        for key, value in mapping.items():
            if value is None:
                assert suite.unavailable().get(key)


def test_negative_integer_seeds_and_strided_frames_are_supported():
    """Do not impose restrictions absent from the frozen seed and frame contracts."""
    req = request(seed=-7)
    engine = create_fake_engine(seed=-11)
    items = list(engine.run(req))
    assert validate_run(items, req) == []
    frame = np.full((5, 7, 3), 20, np.uint8)[:, ::-1]
    assert not frame.flags.c_contiguous
    output = engine.enhance_frame(frame)
    assert output.shape == frame.shape and output.dtype == np.uint8


def test_cancel_before_first_yield_and_between_stages():
    """No further processing happens after cancellation becomes visible."""
    cancel = threading.Event()
    req = request(cancel=cancel)
    cancel.set()
    with pytest.raises(Cancelled):
        next(create_fake_engine().run(req))
    cancel.clear()
    stream = create_fake_engine().run(req)
    assert next(stream).name == "diagnose"
    cancel.set()
    with pytest.raises(Cancelled):
        next(stream)


def test_early_close_releases_run_owned_arrays():
    """A closed generator does not leave array references stored on the engine."""
    engine = create_fake_engine()
    stream = engine.run(request())
    event = next(stream)
    reference = weakref.ref(event.visuals[0].image)
    del event
    stream.close()
    assert stream.gi_frame is None
    assert reference() is None
    req = request()
    assert validate_run(list(engine.run(req)), req) == []


def test_stage_delay(monkeypatch):
    """Each stage delay is invoked once; candidates and warnings add no fake delays."""
    calls = []
    monkeypatch.setattr("lumenx_core.fake.sleep", lambda seconds: calls.append(seconds))
    items = list(create_fake_engine(stage_delay_ms=123).run(request()))
    stage_count = sum(isinstance(x, StageEvent) and x.kind == "stage" for x in items)
    assert calls == [0.123] * stage_count


@pytest.mark.parametrize("mutation,fragment", [
    ("missing_final", "exactly one"),
    ("extra_final", "exactly one"),
    ("stage_order", "stage order"),
    ("duplicate_visual", "duplicate visual"),
    ("wrong_dtype", "uint8"),
    ("wrong_metadata", "role/max_side"),
    ("broken_reference", "unresolved visual"),
    ("missing_metric_reason", "unavailable reason"),
    ("nonfinite", "nonfinite"),
    ("numpy_scalar", "not JSON-safe"),
    ("candidate_chosen", "exactly one candidate"),
    ("wrong_final_size", "final shape"),
    ("bad_recovery", "sum to 100"),
    ("reserved_input", "reserved input"),
    ("early_warning", "warning before diagnose"),
])
def test_checker_detects_broken_contracts(mutation, fragment):
    """The checker must reject malformed output, not merely rubber-stamp FakeEngine."""
    req = request()
    items = list(create_fake_engine().run(req))
    result = result_of(items)
    if mutation == "missing_final":
        items.pop()
    elif mutation == "extra_final":
        items.append(result)
    elif mutation == "stage_order":
        items[0], items[2] = items[2], items[0]
    elif mutation == "duplicate_visual":
        result.visuals.append(items[0].visuals[0])
    elif mutation == "wrong_dtype":
        result.visuals[0].image = result.visuals[0].image.astype(np.float32)
    elif mutation == "wrong_metadata":
        items[0].visuals[0].max_side = 512
    elif mutation == "broken_reference":
        result.story[0].visual_key = "does_not_exist"
    elif mutation == "missing_metric_reason":
        result.metrics["unavailable"].pop("edge_retention")
    elif mutation == "nonfinite":
        result.diagnosis_raw["bad"] = float("nan")
    elif mutation == "numpy_scalar":
        items[0].data["bad"] = np.float32(0.5)
    elif mutation == "candidate_chosen":
        for item in items:
            if isinstance(item, StageEvent) and item.kind == "candidate":
                item.data["chosen"] = False
    elif mutation == "wrong_final_size":
        result.visuals[0].image = np.zeros((2, 2, 3), np.uint8)
    elif mutation == "bad_recovery":
        result.recoverability["recovered_pct"] = 0.0
    elif mutation == "reserved_input":
        result.visuals.append(Visual("input", req.image, "stage", 768, True))
    elif mutation == "early_warning":
        items[0], items[1] = items[1], items[0]
    violations = validate_run(items, req)
    assert any(fragment in x for x in violations), violations


def test_result_serialization_and_no_visual_alias_between_runs():
    """The server can serialize metadata and retain outputs across engine calls."""
    req = request()
    engine = create_fake_engine()
    items = list(engine.run(req))
    result = result_of(items)
    metadata = asdict(result)
    metadata.pop("visuals")
    json.dumps(metadata, allow_nan=False)
    original = result.visuals[0].image.copy()
    list(engine.run(req))
    np.testing.assert_array_equal(result.visuals[0].image, original)
    broken = deepcopy(items)
    broken[0].data["cycle"] = broken[0].data
    assert any("cyclic" in x for x in validate_run(broken, req))
