from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "core"), str(REPO / "server")]

from lumenx_server.app import create_app  # noqa: E402
from lumenx_server.settings import Settings  # noqa: E402


@pytest.fixture
def workspace_tmp() -> Path:
    target = REPO / "server"
    run_root = target / "runs"
    result_root = target / "results"
    run_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    existing_runs = {child.name for child in run_root.iterdir()}
    try:
        yield target
    finally:
        for child in run_root.iterdir():
            if child.name in existing_runs or child.name == ".gitkeep":
                continue
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)


def settings(tmp_path: Path) -> Settings:
    return Settings(
        root=REPO / "server",
        engine="fake",
        mock_stage_delay_ms=0,
        max_upload_bytes=1024 * 1024,
        max_pixels=1_000_000,
        hard_max_pixels=2_000_000,
        max_pending=2,
        max_live_sessions=0,
        run_retention=4,
        debug_validate=True,
        gallery_dir=REPO / "server" / "gallery",
        runs_dir=tmp_path / "runs",
        results_dir=tmp_path / "results",
    )


def png_bytes() -> bytes:
    image = Image.new("RGB", (64, 48), (12, 10, 8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_gallery_and_enhancement_upload(workspace_tmp: Path) -> None:
    with TestClient(create_app(settings(workspace_tmp))) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ready"
        gallery = client.get("/gallery")
        assert gallery.status_code == 200
        assert len(gallery.json()) >= 1

        response = client.post(
            "/enhance",
            files={"file": ("dark.png", png_bytes(), "image/png")},
            data={"mode": "hybrid"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "complete"
        assert payload["final"]["url"].startswith("/files/runs/")
        assert payload["stages"]
        assert payload["run_id"]


def test_invalid_upload_returns_structured_error(workspace_tmp: Path) -> None:
    with TestClient(create_app(settings(workspace_tmp))) as client:
        response = client.post(
            "/enhance",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
            data={"mode": "hybrid"},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "decode_failed"
