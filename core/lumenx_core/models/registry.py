"""Load/verify models once, measure safe placement, and report explicit fallback chains."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from importlib.resources import files
import os
from pathlib import Path
import threading
from typing import Any, Iterator

import numpy as np
import yaml

from ..config import EngineConfig
from ..contracts import Cancelled, EngineError, EngineUnavailable, EngineWarning
from ..types import DeepResult
from .cidnet import CIDNetRunner
from .runner import TensorRunner
from .zerodce import ZeroDCERunner


def checkpoint_manifest() -> dict[str, dict[str, str]]:
    """Read the package-owned official-source/hash manifest, never the network."""
    return yaml.safe_load(files("lumenx_core").joinpath("config/checkpoints.yaml").read_text(encoding="utf-8"))


def weights_directory() -> Path:
    """Resolve weight storage relative to the package, not the working directory."""
    configured = os.environ.get("LUMENX_WEIGHTS_DIR")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2] / "weights"


class ModelRegistry:
    """Non-thread-safe single-lane registry; caller supplies all scheduling/locking."""

    def __init__(self, cfg: EngineConfig, device: str | None = None) -> None:
        """Verify checkpoint bytes before deserialization and select measured placement."""
        self.cfg = cfg
        self.device = "cpu"
        self.policy = "resident"
        self.warm = False
        self.runners: dict[str, TensorRunner] = {}
        self.errors: dict[str, str] = {}
        self.hashes: dict[str, str] = {}
        self.warnings: list[EngineWarning] = []
        self._torch: Any = None
        self._active: str | None = None
        if device is not None and device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise EngineUnavailable(f"Unsupported device {device!r}.")
        try:
            if cfg["models"]["deterministic"]:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            import torch
        except (ImportError, OSError):
            self.errors = dict.fromkeys(checkpoint_manifest(), "torch_unavailable")
            self.warnings.append(EngineWarning("torch_unavailable", "PyTorch is unavailable; classical CPU fallback remains active."))
            return
        self._torch = torch
        if cfg["models"]["deterministic"]:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.set_num_threads(max(1, cfg["real"]["threads"]))
        if device != "cpu":
            if torch.cuda.is_available():
                self.device = "cuda:0" if device in (None, "cuda") else device
            else:
                self.warnings.append(EngineWarning("cpu_fallback", "CUDA is unavailable; trained models will use slower CPU inference."))
        for key, entry in checkpoint_manifest().items():
            path = weights_directory() / entry["filename"]
            if not path.is_file():
                self.errors[key] = "missing_weights"
                continue
            try:
                digest = sha256(path.read_bytes()).hexdigest()
                if digest != entry["sha256"]:
                    self.errors[key] = "hash_mismatch"
                    continue
                if key.startswith("cidnet"):
                    from .cidnet_arch.CIDNet import CIDNet

                    model = CIDNet()
                else:
                    from .zerodce_arch.model import enhance_net_nopool

                    model = enhance_net_nopool(cfg["models"]["zerodce_scale"])
                model.float()
                state = torch.load(path, map_location="cpu", weights_only=True)
                for wrapper_key in ("state_dict", "params"):
                    if wrapper_key in state and isinstance(state[wrapper_key], dict):
                        state = state[wrapper_key]
                        break
                if state and all(key.startswith("module.") for key in state):
                    state = {key.removeprefix("module."): value for key, value in state.items()}
                model.load_state_dict(state, strict=True)
                model.eval()
                model.requires_grad_(False)
                self.runners[key] = TensorRunner(model, "cpu", "fp32")
                self.hashes[key] = digest[:12]
            except (ImportError, OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                self.errors[key] = f"load_failed:{type(exc).__name__}"
        self._select_placement()
        for runner in self.runners.values():
            self._remember_warnings(runner.warnings)

    def _remember_warnings(self, warnings: list[EngineWarning]) -> None:
        """Make startup/live precision degradation inspectable through engine health."""
        known = {(warning.code, warning.message) for warning in self.warnings}
        for warning in warnings:
            if (warning.code, warning.message) not in known:
                self.warnings.append(warning)
                known.add((warning.code, warning.message))

    def _select_placement(self) -> None:
        """Test all resident weights plus worst-case activations, then measured swapping."""
        if not self.runners:
            self.device = "cpu"
            return
        if self.device == "cpu":
            return
        torch = self._torch
        reserve = self.cfg["models"]["reserve_mb"] * 1024**2
        try:
            for runner in self.runners.values():
                runner.place(self.device, "fp16")
            for key, runner in self.runners.items():
                self._probe(key, runner)
                free, _ = torch.cuda.mem_get_info(self.device)
                if free < reserve:
                    raise torch.OutOfMemoryError("Configured GPU memory reserve would be violated.")
            self.policy = "resident"
        except (RuntimeError, EngineError):
            for runner in self.runners.values():
                runner.place("cpu", "fp32")
            torch.cuda.empty_cache()
            self.policy = "swap"
            try:
                for key, runner in self.runners.items():
                    runner.place(self.device, "fp16")
                    self._probe(key, runner)
                    free, _ = torch.cuda.mem_get_info(self.device)
                    runner.place("cpu", "fp32")
                    torch.cuda.empty_cache()
                    if free < reserve:
                        raise torch.OutOfMemoryError("A single model cannot preserve the configured reserve.")
            except (RuntimeError, EngineError):
                for runner in self.runners.values():
                    runner.place("cpu", "fp32")
                torch.cuda.empty_cache()
                self.device, self.policy = "cpu", "resident"
                self.warnings.append(EngineWarning("cpu_memory_fallback", "GPU memory preflight failed; models were kept on CPU."))

    def _probe(self, key: str, runner: TensorRunner) -> None:
        """Run one real-size preflight; returned activation arrays are immediately dropped."""
        c = self.cfg["models"]
        height, width = ((c["live_height"], c["live_width"]) if key == "zerodce_pp"
                         else (c["warmup_side"], c["warmup_side"]))
        image = np.full((height, width, 3), 0.1, np.float32)
        if key == "zerodce_pp":
            ZeroDCERunner(runner, c["zerodce_scale"]).infer(image)
        else:
            CIDNetRunner(runner, self.cfg).infer(image)

    @contextmanager
    def activate(self, key: str) -> Iterator[TensorRunner]:
        """Place an existing model and release transient placement on early closure."""
        runner = self.runners[key]
        self._active = key
        runner.warnings.clear()
        if runner.device != self.device:
            runner.place(self.device, "fp16" if self.device.startswith("cuda") else "fp32")
        try:
            yield runner
        finally:
            self._active = None
            if self.policy == "swap" and runner.device != "cpu":
                runner.place("cpu", "fp32")
                self._torch.cuda.empty_cache()

    def infer(self, image: np.ndarray, severity: str, *, zerodce: bool = False,
              cancel: threading.Event | None = None) -> DeepResult:
        """Try the required checkpoint, then Zero-DCE++, then explicit classical fallback."""
        selected = "cidnet_sid" if severity == "extreme" else "cidnet_lolv2_real"
        keys = ["zerodce_pp"] if zerodce else [selected, "zerodce_pp"]
        chain: list[str] = []
        warnings = list(self.warnings)
        for key in keys:
            if cancel is not None and cancel.is_set():
                raise Cancelled("Enhancement cancelled before the next model.")
            if key not in self.runners:
                reason = self.errors.get(key, "unavailable")
                chain.append(f"{key}:{reason}")
                warnings.append(EngineWarning(f"{'cidnet' if key.startswith('cidnet') else 'zerodce'}_unavailable",
                                              f"The selected {'restoration' if key.startswith('cidnet') else 'live'} model is unavailable; trying its fallback."))
                continue
            try:
                with self.activate(key) as runner:
                    adapter = ZeroDCERunner(runner, self.cfg["models"]["zerodce_scale"]) if key == "zerodce_pp" else CIDNetRunner(runner, self.cfg)
                    output = adapter.infer(image, cancel=cancel)
                    precision = runner.precision
                    warnings.extend(runner.warnings)
                if chain:
                    chain.append(key)
                return DeepResult(output, "zerodce" if key == "zerodce_pp" else "cidnet", key,
                                  self.hashes[key], self.device, precision, chain, warnings)
            except (RuntimeError, EngineError) as exc:
                chain.append(f"{key}:{type(exc).__name__}")
                warnings.append(EngineWarning("model_inference_failed", "A trained model failed; the recorded fallback was selected."))
                if self.device.startswith("cuda"):
                    self._torch.cuda.empty_cache()
        chain.append("classical")
        return DeepResult(None, "classical_fallback", None, None, "cpu", "fp32", chain, warnings)

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Live never falls back to classical silently: require a verified Zero-DCE++ model."""
        if "zerodce_pp" not in self.runners:
            raise EngineUnavailable("Verified Zero-DCE++ weights and PyTorch are required for live inference.")
        with self.activate("zerodce_pp") as runner:
            output = ZeroDCERunner(runner, self.cfg["models"]["zerodce_scale"]).infer(frame)
            self._remember_warnings(runner.warnings)
            return output

    def warmup(self) -> None:
        """Run two real-size inferences per loaded model, once per registry lifetime."""
        if not self.warm:
            for key in self.runners:
                with self.activate(key) as runner:
                    self._probe(key, runner)
                    self._probe(key, runner)
                    self._remember_warnings(runner.warnings)
            self.warm = True

    def memory(self) -> dict[str, int]:
        """Report actual GPU allocation counters, or zeros for CPU operation."""
        if self.device.startswith("cuda"):
            torch = self._torch
            return {"total": int(torch.cuda.get_device_properties(self.device).total_memory / 1024**2),
                    "allocated": int(torch.cuda.memory_allocated(self.device) / 1024**2),
                    "reserved": int(torch.cuda.memory_reserved(self.device) / 1024**2),
                    "peak": int(torch.cuda.max_memory_allocated(self.device) / 1024**2)}
        return dict.fromkeys(("total", "allocated", "reserved", "peak"), 0)
