"""Read package-owned configuration and expose an isolated JSON-safe copy."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from importlib.resources import files
import json
from typing import Any

import yaml


def load_fake_config() -> tuple[dict[str, Any], str]:
    """Load configuration without consulting the working directory or user paths."""
    source = files("lumenx_core").joinpath("config/default.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(source)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return deepcopy(config), sha256(encoded.encode()).hexdigest()[:12]
