"""Optional model loading, availability reporting, and single-instance warmup."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .zerodce import ZeroDCEPlusPlus


class ModelRegistry:
    """Own one loaded Zero-DCE++ object and report missing-weight degradation."""

    def __init__(self, cfg: dict[str, dict[str, Any]], device: str | None = None):
        self.cfg = cfg
        self.requested_device = device or str(cfg["deep"].get("device", "auto"))
        self.device = "cpu"
        self.model: ZeroDCEPlusPlus | None = None
        self.checkpoint: Path | None = None
        self.checkpoint_hash: str | None = None
        self.error: str | None = None
        self.precision = "fp32"
        self.scale_factor = max(1, int(cfg["deep"].get("scale_factor", 12)))
        self.warm = False
        self._resolve_device()
        override = os.environ.get("LUMENX_ZERODCE_CHECKPOINT")
        if override:
            self.checkpoint = Path(override).expanduser().resolve()
        else:
            weight_root = Path(
                os.environ.get("LUMENX_WEIGHTS_DIR", str(Path(__file__).resolve().parents[2] / "weights"))
            ).expanduser()
            self.checkpoint = (weight_root / str(cfg["deep"]["checkpoint"])).resolve()

    def _resolve_device(self) -> None:
        try:
            import torch
        except ImportError:
            if self.requested_device not in ("auto", "cpu", None):
                self.error = "PyTorch is not installed; using the CPU classical path."
            return
        wants_cuda = str(self.requested_device).startswith("cuda") or self.requested_device == "auto"
        if wants_cuda and torch.cuda.is_available():
            self.device = "cuda:0" if self.requested_device == "auto" else str(self.requested_device)
        elif str(self.requested_device).startswith("cuda"):
            self.error = "CUDA was requested but is unavailable; using CPU FP32."
        else:
            self.device = "cpu"

    @property
    def available(self) -> bool:
        return self.model is not None

    def warmup(self) -> None:
        if self.warm:
            return
        try:
            import torch  # noqa: F401
        except ImportError:
            if self.error is None:
                self.error = "PyTorch is not installed; install lumenx-core[models] for Zero-DCE++."
            self.warm = True
            return
        if self.checkpoint is None or not self.checkpoint.is_file():
            if self.error is None:
                self.error = "Zero-DCE++ weights are missing; run `python -m lumenx_core.download_weights`."
            self.warm = True
            return
        try:
            hasher = hashlib.sha256()
            with self.checkpoint.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            self.model = ZeroDCEPlusPlus(
                self.checkpoint,
                self.device,
                scale_factor=self.scale_factor,
            )
            sample = np.full((48, 48, 3), 0.1, dtype=np.float32)
            self.model.enhance(sample)
            self.checkpoint_hash = f"sha256-{digest[:12]}"
            self.precision = "fp16" if self.device.startswith("cuda") else "fp32"
        except Exception as exc:
            self.model = None
            self.error = f"Zero-DCE++ could not be initialized: {exc}"
        self.warm = True

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if not self.warm:
            self.warmup()
        if self.model is None:
            raise RuntimeError(self.error or "Zero-DCE++ is unavailable.")
        return self.model.enhance(image)

    def health(self) -> dict[str, Any]:
        try:
            import torch

            if torch.cuda.is_available() and self.device.startswith("cuda"):
                props = torch.cuda.get_device_properties(self.device)
                vram = {
                    "total": int(props.total_memory // (1024 * 1024)),
                    "allocated": int(torch.cuda.memory_allocated(self.device) // (1024 * 1024)),
                    "reserved": int(torch.cuda.memory_reserved(self.device) // (1024 * 1024)),
                    "peak": int(torch.cuda.max_memory_allocated(self.device) // (1024 * 1024)),
                }
            else:
                vram = {"total": 0, "allocated": 0, "reserved": 0, "peak": 0}
        except Exception:
            vram = {"total": 0, "allocated": 0, "reserved": 0, "peak": 0}
        degraded = []
        if not self.available:
            degraded.append("zerodce_unavailable")
        if self.error:
            degraded.append("model_load_error" if "could not be initialized" in self.error else "model_runtime_unavailable")
        return {
            "fake": False,
            "ready": self.warm,
            "hybrid_ready": self.available,
            "warm": self.warm,
            "device": self.device,
            "placement_policy": "resident",
            "models": {
                "cidnet": {"available": False, "checkpoints": []},
                "zerodce": {"available": self.available},
                "iqa": {
                    "clipiqa": False,
                    "niqe": False,
                    "brisque": False,
                    "musiq": False,
                    "lpips": False,
                },
            },
            "vram_mb": vram,
            "degraded": degraded,
            "core_version": "0.4.0",
            "contract_version": "1.0",
            "model_message": self.error,
        }

