"""Read package-owned configuration and expose an isolated JSON-safe copy."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from importlib.resources import files
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

import yaml


def load_fake_config() -> tuple[dict[str, Any], str]:
    """Load configuration without consulting the working directory or user paths."""
    source = files("lumenx_core").joinpath("config/default.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(source)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return deepcopy(config), sha256(encoded.encode()).hexdigest()[:12]


def _freeze(value: Any) -> Any:
    """Recursively prevent mutation of engine configuration."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return built-in JSON-safe containers without exposing private mappings."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class EngineConfig:
    """Validated immutable values and a reproducible twelve-character hash."""

    values: Mapping[str, Any]
    digest: str

    def __getitem__(self, key: str) -> Any:
        """Access a read-only configuration section."""
        return self.values[key]

    def public(self) -> dict[str, Any]:
        """Produce an isolated public copy with no filesystem paths."""
        return _thaw(self.values)


def load_config(config_path: str | None = None) -> EngineConfig:
    """Load defaults plus known-key overrides; reject malformed configuration early."""
    defaults, _ = load_fake_config()
    if config_path is not None:
        try:
            override = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Malformed YAML configuration: {exc}") from exc
        if not isinstance(override, dict):
            raise ValueError("Configuration override must be a mapping.")

        def merge(target: dict, updates: dict, path: str = "config") -> None:
            """Allow only existing keys and matching value types."""
            for key, value in updates.items():
                if key not in target:
                    raise ValueError(f"Unknown configuration key: {path}.{key}")
                old = target[key]
                if isinstance(old, dict):
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected mapping: {path}.{key}")
                    merge(old, value, f"{path}.{key}")
                else:
                    if type(old) is float and type(value) in (int, float):
                        value = float(value)
                    if type(value) is not type(old):
                        raise ValueError(f"Wrong configuration type: {path}.{key}")
                    target[key] = value

        merge(defaults, override)
    encoded = json.dumps(defaults, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _validate_config(defaults)
    return EngineConfig(_freeze(defaults), sha256(encoded.encode()).hexdigest()[:12])


def _validate_config(cfg: dict[str, Any]) -> None:
    """Enforce numeric domains that otherwise lead to invalid image operations."""
    real = cfg["real"]
    if real["threads"] < 1 or real["epsilon"] <= 0:
        raise ValueError("Threads and numerical epsilon must be positive.")
    for section, keys in {
        "diagnosis": ("proxy_side", "noise_block", "noise_samples", "minimum_patches", "mean_window", "guide_radius"),
        "recoverability": ("guide_radius",),
        "classical": ("clahe_tiles", "minimum_clahe_side"),
        "denoise": ("radius",),
        "metrics": ("proxy_side", "minimum_edges", "halo_radius"),
    }.items():
        for key in keys:
            if type(real[section][key]) is not int or real[section][key] < 1:
                raise ValueError(f"real.{section}.{key} must be a positive integer.")
    if real["diagnosis"]["noise_samples"] > 100:
        raise ValueError("Native noise sampling is limited to 100 accepted patches.")
    if real["recoverability"]["snr_high"] <= real["recoverability"]["snr_low"]:
        raise ValueError("Recovery SNR thresholds must be increasing.")
    if real["recoverability"]["gain_high"] < real["recoverability"]["gain_low"]:
        raise ValueError("Recovery gain limits must be increasing.")
    for value in real["classical"]["gamma"].values():
        if not 0 < value <= 1:
            raise ValueError("Classical gamma must be in (0, 1].")
    for value in real["denoise"]["epsilons"]:
        if type(value) not in (float, int) or value <= 0:
            raise ValueError("Guided-filter epsilons must be positive numbers.")
    if len(real["denoise"]["epsilons"]) != 3:
        raise ValueError("Denoising requires exactly three filter strengths.")
    for section, key in (("diagnosis", "guide_epsilon"), ("diagnosis", "mad_scale"),
                         ("diagnosis", "laplacian_norm"), ("recoverability", "guide_epsilon"),
                         ("classical", "color_scale"), ("denoise", "sigma_reference")):
        if real[section][key] <= 0:
            raise ValueError(f"real.{section}.{key} must be positive.")
    if not 0 < real["denoise"]["filter_scale"] <= 1:
        raise ValueError("Fast guided-filter scale must be in (0, 1].")
    if not 0 < real["classical"]["highlight_start"] < 1:
        raise ValueError("Highlight protection cutoff must be in (0, 1).")
    if not 0 <= real["recoverability"]["class_low"] < real["recoverability"]["class_high"] <= 1:
        raise ValueError("Recovery display thresholds must be increasing within [0, 1].")
    if real["recoverability"]["gain_low"] < 1:
        raise ValueError("Gain caps must allow at least the unchanged input.")
    for section, keys in {"diagnosis": ("patch_valid_fraction",),
                          "recoverability": ("confidence_floor",),
                          "classical": ("clahe_strength", "color_max_strength"),
                          "denoise": ("chroma_strength", "edge_strength_fraction"),
                          "render": ("overlay_alpha",)}.items():
        if any(not 0 <= real[section][key] <= 1 for key in keys):
            raise ValueError(f"Fraction values in real.{section} must be in [0, 1].")
    for key in ("snr_max", "noise_max", "gain_max"):
        if real["render"][key] <= 0:
            raise ValueError(f"Map scale {key} must be positive.")
    weights = real["luminance_weights"]
    if len(weights) != 3 or any(type(x) not in (int, float) or x < 0 for x in weights) or abs(sum(weights) - 1) > 1e-6:
        raise ValueError("Luminance weights must be three nonnegative numbers summing to one.")
    if real["metrics"]["proxy_side"] != 512:
        raise ValueError("The frozen contract requires a 512-pixel request metric proxy.")
