"""Release gate: invalid inputs fail before any processing item is yielded."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lumenx_core import create_fake_engine
from lumenx_core.contracts import EnhanceRequest, InvalidRequest


def valid_request() -> EnhanceRequest:
    """Return a minimal legal request for negative-input tests."""
    return EnhanceRequest(np.full((5, 7, 3), 0.1, np.float32), (7, 5))


def bad_cases():
    """Exercise value/type checks, shape limits, and flag interactions independently."""
    req = valid_request()
    yield replace(req, image=req.image.astype(np.float64))
    yield replace(req, image=req.image.astype(np.uint8))
    yield replace(req, image=req.image[..., 0])
    yield replace(req, image=np.zeros((5, 7, 4), np.float32))
    yield replace(req, image=np.zeros((0, 7, 3), np.float32))
    yield replace(req, image=np.zeros((5, 0, 3), np.float32))
    yield replace(req, image=req.image[:, ::-1])
    yield replace(req, image=np.full(req.image.shape, np.nan, np.float32))
    yield replace(req, image=np.full(req.image.shape, np.inf, np.float32))
    yield replace(req, image=np.full(req.image.shape, -0.01, np.float32))
    yield replace(req, image=np.full(req.image.shape, 1.01, np.float32))
    yield replace(req, image="not_an_array")
    yield replace(req, mode="unknown")
    yield replace(req, ablate=frozenset({"typo"}))
    yield replace(req, ablate={"no_tuner"})
    yield replace(req, ablate=frozenset({42}))
    yield replace(req, ablate=frozenset({"no_dl", "no_fusion"}))
    yield replace(req, ablate=frozenset({"no_dl", "no_classical"}))
    for mode in ("classical", "dl", "zerodce"):
        yield replace(req, mode=mode, ablate=frozenset({"no_tuner"}))
    yield replace(req, alpha=np.zeros((5, 7), np.uint8))
    yield replace(req, alpha=np.zeros((5, 7, 1), np.float32))
    yield replace(req, alpha=np.zeros((1, 1), np.float32))
    yield replace(req, alpha=np.full((5, 7), np.nan, np.float32))
    yield replace(req, alpha=np.full((5, 7), 1.1, np.float32))
    yield replace(req, original_size=(0, 5))
    yield replace(req, original_size=[7, 5])
    yield replace(req, original_size=(7, 5.0))
    yield replace(req, seed=True)
    yield replace(req, seed=0.5)
    yield replace(req, cancel=True)


@pytest.mark.parametrize("req", list(bad_cases()))
def test_invalid_request_raises_before_first_yield(req):
    """No partial visual stream may leak out before request validation completes."""
    stream = create_fake_engine().run(req)
    with pytest.raises(InvalidRequest) as caught:
        next(stream)
    assert caught.value.code
    assert list(stream) == []


def test_over_limit_without_materializing_pixels():
    """Reject >12 MP before reading contents of a huge allocation."""
    image = np.empty((3001, 4000, 3), np.float32)
    stream = create_fake_engine().run(EnhanceRequest(image, (4000, 3001)))
    with pytest.raises(InvalidRequest) as caught:
        next(stream)
    assert caught.value.code == "image_size"


@pytest.mark.parametrize("frame", [
    np.zeros((2, 2), np.uint8),
    np.zeros((2, 2, 4), np.uint8),
    np.zeros((2, 2, 3), np.float32),
    np.zeros((0, 2, 3), np.uint8),
    np.zeros((1, 1281, 3), np.uint8),
])
def test_invalid_live_frames(frame):
    """Frame shape and dtype errors use the same public error type."""
    with pytest.raises(InvalidRequest):
        create_fake_engine().enhance_frame(frame)


@pytest.mark.parametrize("kwargs", [
    {"stage_delay_ms": -1}, {"stage_delay_ms": 1.5}, {"stage_delay_ms": True},
    {"seed": 0.5}, {"seed": True},
])
def test_invalid_factory_controls(kwargs):
    """Invalid demo controls are rejected immediately."""
    with pytest.raises(ValueError):
        create_fake_engine(**kwargs)
