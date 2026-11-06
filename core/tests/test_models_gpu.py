"""Real publisher-weight release gates. Opt in with --run-gpu; missing prerequisites FAIL."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from lumenx_core import create_engine
from lumenx_core.contract_check import validate_run
from lumenx_core.contracts import EnhanceRequest, EnhanceResult

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def gpu_engine():
    """Require the actual GPU and all three verified checkpoints; no random weights."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        pytest.fail("GPU gates require PyTorch CUDA; no GPU gate has passed.")
    assert torch.cuda.is_available(), "Actual CUDA device required for release gates."
    engine = create_engine(device="cuda:0")
    required = {"cidnet_lolv2_real", "cidnet_sid", "zerodce_pp"}
    assert required <= engine._registry.runners.keys(), engine._registry.errors
    assert engine._registry.device.startswith("cuda"), "CPU fallback does not satisfy GPU gates."
    engine.warmup()
    return engine


@pytest.mark.parametrize("mode", ["classical", "dl", "zerodce", "hybrid"])
def test_real_mode_contracts(gpu_engine, mode):
    """Every mode uses actual verified weights where required and passes the contract."""
    image = np.random.default_rng(41).uniform(0.03, 0.15, (96, 128, 3)).astype(np.float32)
    req = EnhanceRequest(image, (128, 96), mode=mode,
                         ablate=frozenset({"no_tuner"}) if mode == "hybrid" else frozenset())
    items = list(gpu_engine.run(req))
    assert validate_run(items, req) == []
    result = items[-1]
    assert not result.provenance["fallback_chain"], result.warnings
    if mode != "classical":
        assert result.provenance["device"].startswith("cuda")
        assert result.provenance["checkpoint"]


@pytest.mark.parametrize("severity,checkpoint", [("moderate", "cidnet_lolv2_real"), ("extreme", "cidnet_sid")])
def test_checkpoint_selection_and_black_precision(gpu_engine, severity, checkpoint):
    """Checkpoint routing and black-input numerical stability are verified on CUDA."""
    image = np.zeros((32, 40, 3), np.float32)
    result = gpu_engine._registry.infer(image, severity)
    assert result.checkpoint == checkpoint and result.image is not None
    assert np.isfinite(result.image).all()
    assert not result.fallback_chain


@pytest.mark.parametrize("shape", [(1080, 1920), (3000, 4000)])
def test_native_hybrid_no_oom(gpu_engine, shape):
    """Run the whole no_tuner hybrid, preserving native 1080p and 12 MP geometry."""
    image = np.full((*shape, 3), 0.08, np.float32)
    req = EnhanceRequest(image, (shape[1], shape[0]), mode="hybrid", ablate=frozenset({"no_tuner"}))
    result = None
    for item in gpu_engine.run(req):
        if isinstance(item, EnhanceResult):
            result = item
    assert result is not None
    assert result.visuals[0].image.shape == (*shape, 3)
    assert result.provenance["checkpoint"] == "cidnet_lolv2_real"
    assert not result.provenance["fallback_chain"], result.warnings
    assert gpu_engine.health()["vram_mb"]["peak"] < gpu_engine.health()["vram_mb"]["total"]


def test_live_sixty_seconds(gpu_engine):
    """Measure actual full frame-call latency for 60 seconds; do not substitute warmup FPS."""
    frame = np.full((480, 640, 3), 16, np.uint8)
    gpu_engine.enhance_frame(frame)
    values = []
    started = perf_counter()
    while perf_counter() - started < 60:
        begin = perf_counter()
        output = gpu_engine.enhance_frame(frame)
        values.append((perf_counter() - begin) * 1000)
        assert output.shape == frame.shape and output.dtype == np.uint8
    report = {"frames": len(values), "seconds": perf_counter() - started,
              "mean_ms": float(np.mean(values)), "p95_ms": float(np.percentile(values, 95)),
              "max_ms": float(np.max(values)), "health": gpu_engine.health()}
    root = Path(__file__).resolve().parents[1] / "bench_results"
    root.mkdir(parents=True, exist_ok=True)
    (root / "gpu_live.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert len(values) / report["seconds"] >= 20, report
    assert report["p95_ms"] <= 40, report
