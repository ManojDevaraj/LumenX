from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from lumenx_core import EnhanceRequest, create_engine, create_fake_engine  # noqa: E402
from lumenx_core.contract_check import validate_run  # noqa: E402
from lumenx_core.contracts import EnhanceResult  # noqa: E402
from lumenx_core.models.zerodce import ZeroDCEPlusPlus  # noqa: E402


def dark_test_image(height: int = 48, width: int = 64) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    luminance = (0.015 + 0.08 * (x / max(width - 1, 1)) + 0.04 * (y / max(height - 1, 1))).astype(np.float32)
    image = np.stack((luminance, luminance * 0.94, luminance * 0.86), axis=-1)
    image[height // 3 : height // 3 + 2, width // 2 - 1 : width // 2 + 1] = 0.8
    return np.ascontiguousarray(np.clip(image, 0, 1), dtype=np.float32)


def test_fake_engine_satisfies_hybrid_stream_contract() -> None:
    image = dark_test_image()
    request = EnhanceRequest(image=image, original_size=(image.shape[1], image.shape[0]), mode="hybrid")
    items = list(create_fake_engine().run(request))
    errors = validate_run(items, request)
    assert not errors, "; ".join(errors)
    assert isinstance(items[-1], EnhanceResult)
    final = next(visual for visual in items[-1].visuals if visual.key == "final")
    assert final.image.shape == (image.shape[0], image.shape[1], 3)


def test_real_classical_engine_satisfies_stream_contract() -> None:
    image = dark_test_image()
    request = EnhanceRequest(image=image, original_size=(image.shape[1], image.shape[0]), mode="classical")
    engine = create_engine(device="cpu")
    items = list(engine.run(request))
    errors = validate_run(items, request)
    assert not errors, "; ".join(errors)
    assert isinstance(items[-1], EnhanceResult)
    assert items[-1].provenance["actual_method"] == "classical"


def test_zero_dce_checkpoint_loads_and_preserves_image_geometry() -> None:
    checkpoint = CORE / "weights" / "zerodce_pp.pth"
    if not checkpoint.is_file():
        pytest.skip("optional Zero-DCE++ checkpoint has not been downloaded")
    model = ZeroDCEPlusPlus(checkpoint, device="cpu", scale_factor=12)
    output = model.enhance(dark_test_image(48, 60))
    assert output.shape == (48, 60, 3)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0
