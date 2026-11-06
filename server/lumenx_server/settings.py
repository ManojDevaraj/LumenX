"""Validated, public-safe server configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    root: Path
    engine: str
    mock_stage_delay_ms: int
    max_upload_bytes: int
    max_pixels: int
    hard_max_pixels: int
    max_pending: int
    max_live_sessions: int
    run_retention: int
    debug_validate: bool
    gallery_dir: Path
    runs_dir: Path
    results_dir: Path

    def public(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "mock_stage_delay_ms": self.mock_stage_delay_ms,
            "max_upload_bytes": self.max_upload_bytes,
            "max_pixels": self.max_pixels,
            "hard_max_pixels": self.hard_max_pixels,
            "max_pending": self.max_pending,
            "max_live_sessions": self.max_live_sessions,
            "run_retention": self.run_retention,
            "debug_validate": self.debug_validate,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(os.environ.get("LUMENX_SERVER_CONFIG", root / "server.yaml"))
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    engine = os.environ.get("LUMENX_ENGINE", str(values.get("engine", "fake"))).lower()
    if os.environ.get("LUMENX_MOCK") == "1":
        engine = "fake"
    if engine not in ("fake", "real"):
        raise ValueError("server engine must be 'fake' or 'real'")

    def path_value(key: str) -> Path:
        value = Path(str(values.get(key, key))).expanduser()
        return (root / value).resolve() if not value.is_absolute() else value.resolve()

    settings = Settings(
        root=root,
        engine=engine,
        mock_stage_delay_ms=int(values.get("mock_stage_delay_ms", 180)),
        max_upload_bytes=int(values.get("max_upload_bytes", 32 * 1024 * 1024)),
        max_pixels=int(values.get("max_pixels", 12_000_000)),
        hard_max_pixels=int(values.get("hard_max_pixels", 50_000_000)),
        max_pending=int(values.get("max_pending", 2)),
        max_live_sessions=int(values.get("max_live_sessions", 1)),
        run_retention=int(values.get("run_retention", 30)),
        debug_validate=bool(values.get("debug_validate", False)),
        gallery_dir=path_value("gallery_dir"),
        runs_dir=path_value("runs_dir"),
        results_dir=path_value("results_dir"),
    )
    if settings.mock_stage_delay_ms < 0:
        raise ValueError("mock_stage_delay_ms must not be negative")
    if settings.max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive")
    if not 1 <= settings.max_pixels <= 12_000_000:
        raise ValueError("max_pixels must be between 1 and 12000000")
    if settings.hard_max_pixels < settings.max_pixels:
        raise ValueError("hard_max_pixels must be at least max_pixels")
    if settings.max_pending < 0:
        raise ValueError("max_pending must not be negative")
    if settings.max_live_sessions < 0:
        raise ValueError("max_live_sessions must not be negative")
    if settings.run_retention < 1:
        raise ValueError("run_retention must be positive")
    return settings

