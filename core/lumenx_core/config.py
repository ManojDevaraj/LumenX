"""Frozen configuration loading for the LumenX engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EngineConfig:
    """Parsed configuration and stable digest used for run provenance."""

    values: dict[str, Any]
    digest: str

    def public(self) -> dict[str, Any]:
        """Return a JSON-safe copy that contains no local file paths."""

        return json.loads(json.dumps(self.values, sort_keys=True))


def load_config(config_path: str | None = None) -> EngineConfig:
    """Load YAML configuration from an explicit path or package defaults."""

    if config_path is None:
        path = Path(__file__).parent / "config" / "default.yaml"
    else:
        path = Path(config_path).expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for the real engine; install lumenx-core[vision]."
        ) from exc
    raw = path.read_bytes()
    parsed = yaml.safe_load(raw) or {}
    if not isinstance(parsed, dict):
        raise ValueError("engine configuration must be a YAML mapping")
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return EngineConfig(values=parsed, digest=f"sha256-{digest}")
