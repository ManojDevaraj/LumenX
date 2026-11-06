"""CPU unit tests must not accidentally load local models or claim GPU gate coverage."""

from __future__ import annotations

import importlib.util

import pytest


def pytest_addoption(parser):
    """Require explicit opt-in for actual CUDA/weight gates rather than silent skips."""
    parser.addoption("--run-gpu", action="store_true", help="Run required real-CUDA publisher-weight gates too")


def pytest_collection_modifyitems(config, items):
    """Report deselected GPU gates separately from passing CPU unit tests."""
    if config.getoption("--run-gpu"):
        return
    gpu = [item for item in items if item.get_closest_marker("gpu")]
    if gpu:
        config.hook.pytest_deselected(items=gpu)
        items[:] = [item for item in items if not item.get_closest_marker("gpu")]


@pytest.fixture(autouse=True)
def isolated_unit_weights(monkeypatch, tmp_path, request):
    """GPU tests opt in explicitly; all other tests use an empty weight directory."""
    if request.node.get_closest_marker("gpu"):
        return
    monkeypatch.setenv("LUMENX_WEIGHTS_DIR", str(tmp_path / "no_weights"))
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
        except (ImportError, OSError):
            return

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
