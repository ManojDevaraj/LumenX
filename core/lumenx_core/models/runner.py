"""Shared tensor inference with finite-output checks and a bounded precision retry."""

from __future__ import annotations

import numpy as np

from ..contracts import EngineError, EngineWarning


class TensorRunner:
    """Run an existing eval model; no downloads, locks or model reconstruction."""

    def __init__(self, model: object, device: str, precision: str) -> None:
        """Keep model placement metadata and per-inference warnings."""
        self.model = model
        # Preserve original float32 weights across half/device transitions. This is
        # an in-memory snapshot, never a request-time checkpoint read/rebuild.
        self._cpu_state = {key: value.detach().cpu().clone()
                           for key, value in model.state_dict().items()}
        self.device = device
        self.precision = precision
        self.warnings: list[EngineWarning] = []

    def place(self, device: str, precision: str) -> None:
        """Move existing weights, preserving model identity and explicit precision."""
        import torch

        self.model.to(device=device, dtype=torch.float16 if precision == "fp16" else torch.float32)
        if precision == "fp32":
            self.model.load_state_dict(self._cpu_state, strict=True)
        self.model.eval()
        self.device, self.precision = device, precision

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Return finite float32 CPU RGB and release GPU activation references in finally."""
        import torch

        tensor = prediction = None
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
            tensor = tensor.to(device=self.device, dtype=torch.float16 if self.precision == "fp16" else torch.float32)
            with torch.inference_mode():
                prediction = self.model(tensor)
                if isinstance(prediction, (tuple, list)):
                    prediction = prediction[0]
                if not torch.isfinite(prediction).all():
                    if self.precision != "fp16":
                        raise EngineError("Model produced nonfinite float32 output.")
                    del prediction
                    prediction = None
                    self.warnings.append(EngineWarning("fp16_nonfinite", "The model needed a float32 precision retry."))
                    self.place(self.device, "fp32")
                    tensor = tensor.float()
                    prediction = self.model(tensor)
                    if isinstance(prediction, (tuple, list)):
                        prediction = prediction[0]
                    if not torch.isfinite(prediction).all():
                        raise EngineError("Model produced nonfinite output after its float32 retry.")
                if prediction.shape != tensor.shape:
                    raise EngineError("Model output does not match its padded RGB input.")
                result = prediction.clamp(0, 1).float().squeeze(0).permute(1, 2, 0).cpu().numpy().copy()
            return np.ascontiguousarray(result)
        finally:
            del prediction, tensor
