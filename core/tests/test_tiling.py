"""Model-independent tiling tests run on CPU without PyTorch or checkpoints."""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from lumenx_core.contracts import Cancelled, EngineError
from lumenx_core.models.cidnet import pad_stride, tile_starts, tiled_inference


@pytest.mark.parametrize("shape", [(1, 1), (1, 35), (35, 1), (33, 49), (511, 513), (1080, 1920)])
def test_identity_tiles_cover_every_pixel(shape):
    """An identity model exposes zero-denominator borders and padding mistakes."""
    image = np.random.default_rng(4).uniform(0, 1, (*shape, 3)).astype(np.float32)
    calls = []

    def infer(patch):
        calls.append(patch.shape)
        assert patch.shape[0] % 8 == patch.shape[1] % 8 == 0
        assert max(patch.shape[:2]) <= 512
        return patch.copy()

    result = tiled_inference(image, infer)
    np.testing.assert_allclose(result, image, atol=2e-7)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    assert len(calls) == len(tile_starts(shape[0], 512, 32)) * len(tile_starts(shape[1], 512, 32))


def test_edges_across_tile_borders_do_not_create_seams():
    """A context-dependent blur of crossing edges must not leave visible tile cuts."""
    image = np.zeros((640, 960, 3), np.float32)
    image[:, 471:481] = 1
    image[471:481] = 1
    result = tiled_inference(image, lambda x: cv2.GaussianBlur(x, (3, 3), 0))
    expected = cv2.GaussianBlur(image, (3, 3), 0)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_cancel_between_tiles_and_invalid_prediction():
    """Cancellation cannot wait for all tiles; malformed output must fail explicitly."""
    cancel = threading.Event()
    calls = []

    def infer(patch):
        calls.append(1)
        cancel.set()
        return patch

    image = np.zeros((600, 600, 3), np.float32)
    with pytest.raises(Cancelled):
        tiled_inference(image, infer, cancel=cancel)
    assert len(calls) == 1
    with pytest.raises(EngineError):
        tiled_inference(image, lambda x: np.full_like(x, np.nan))


def test_stride_padding_preserves_single_pixel():
    """Replication padding works where reflection padding would be undefined."""
    image = np.array([[[0.2, 0.4, 0.6]]], np.float32)
    padded, shape = pad_stride(image, 8)
    assert shape == (1, 1) and padded.shape == (8, 8, 3)
    np.testing.assert_allclose(padded, np.tile(image, (8, 8, 1)))
