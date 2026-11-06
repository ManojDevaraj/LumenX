"""Optional Zero-DCE++ inference adapter.

The network structure follows the authors' Zero-DCE++ implementation. Its
upstream repository is licensed CC BY-NC 4.0; see LICENSE_NOTICES.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _torch_components() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Zero-DCE++; install lumenx-core[models].") from exc
    return torch, nn, functional


def _network_class() -> Any:
    torch, nn, functional = _torch_components()

    class DepthwisePointwise(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.depth_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels
            )
            self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        def forward(self, value: Any) -> Any:
            return self.point_conv(self.depth_conv(value))

    class ZeroDCEPlusPlusNet(nn.Module):
        """Compact depthwise curve estimator with shared eight-step curves."""

        def __init__(self, scale_factor: int = 12):
            super().__init__()
            self.scale_factor = scale_factor
            self.relu = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            width = 32
            self.e_conv1 = DepthwisePointwise(3, width)
            self.e_conv2 = DepthwisePointwise(width, width)
            self.e_conv3 = DepthwisePointwise(width, width)
            self.e_conv4 = DepthwisePointwise(width, width)
            self.e_conv5 = DepthwisePointwise(width * 2, width)
            self.e_conv6 = DepthwisePointwise(width * 2, width)
            self.e_conv7 = DepthwisePointwise(width * 2, 3)

        @staticmethod
        def _apply_curves(image: Any, curves: Any) -> Any:
            for _ in range(8):
                image = image + curves * (image.square() - image)
            return image

        def forward(self, image: Any) -> tuple[Any, Any]:
            low = functional.interpolate(
                image, scale_factor=1.0 / self.scale_factor, mode="bilinear", align_corners=False
            )
            first = self.relu(self.e_conv1(low))
            second = self.relu(self.e_conv2(first))
            third = self.relu(self.e_conv3(second))
            fourth = self.relu(self.e_conv4(third))
            fifth = self.relu(self.e_conv5(torch.cat((third, fourth), dim=1)))
            sixth = self.relu(self.e_conv6(torch.cat((second, fifth), dim=1)))
            curves = torch.tanh(self.e_conv7(torch.cat((first, sixth), dim=1)))
            curves = self.upsample(curves)
            return self._apply_curves(image, curves), curves

    return ZeroDCEPlusPlusNet


class ZeroDCEPlusPlus:
    """Load a local checkpoint once and provide deterministic RGB inference."""

    def __init__(self, checkpoint: Path, device: str = "cpu", scale_factor: int = 12):
        torch, _, _ = _torch_components()
        network_type = _network_class()
        if scale_factor < 1:
            raise ValueError("Zero-DCE++ scale_factor must be positive")
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.scale_factor = int(scale_factor)
        self.model = network_type(scale_factor=self.scale_factor)
        try:
            try:
                state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(self.checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError("checkpoint does not contain a state dictionary")
            if state and all(str(key).startswith("module.") for key in state):
                state = {str(key)[7:]: value for key, value in state.items()}
            self.model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load Zero-DCE++ checkpoint {self.checkpoint.name}: {exc}") from exc
        self.model.to(self.device).eval()
        self.torch = torch

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance normalized RGB float32 input while preserving exact geometry."""

        if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Zero-DCE++ expects a float32 HxWx3 RGB image.")
        h, w = image.shape[:2]
        pad_h = (-h) % self.scale_factor
        pad_w = (-w) % self.scale_factor
        pad_mode = "reflect" if min(h, w) > 1 else "edge"
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
        tensor = self.torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.torch.float32)
        with self.torch.inference_mode():
            if self.device.type == "cuda":
                with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                    output, _ = self.model(tensor)
                if not self.torch.isfinite(output).all():
                    with self.torch.autocast(device_type="cuda", enabled=False):
                        output, _ = self.model(tensor)
            else:
                output, _ = self.model(tensor)
        result = output[0, :, :h, :w].permute(1, 2, 0).float().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Zero-DCE++ returned non-finite pixels.")
        return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

