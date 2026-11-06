"""Atomic, size-aware publication of core visuals and JSON results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _fit(image: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _atomic_image(path: Path, image: np.ndarray, *, lossless: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if lossless else ".jpg"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        arr = np.asarray(image)
        if arr.ndim == 2:
            pil = Image.fromarray(arr.astype(np.uint8), mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pil = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        else:
            raise ValueError(f"unsupported visual shape: {arr.shape}")
        if not lossless and pil.mode == "RGBA":
            pil = pil.convert("RGB")
        if lossless:
            pil.save(temporary, format="PNG", compress_level=1)
        else:
            pil.save(temporary, format="JPEG", quality=90, optimize=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_visual(run_dir: Path, visual: Any) -> Path:
    extension = ".png" if visual.lossless else ".jpg"
    path = run_dir / f"{visual.key}{extension}"
    _atomic_image(path, _fit(visual.image, visual.max_side), lossless=visual.lossless)
    return path


def publish_input(run_dir: Path, image: np.ndarray, alpha: np.ndarray | None) -> Path:
    rgb = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    data = (
        np.dstack((rgb, np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)))
        if alpha is not None
        else rgb
    )
    path = run_dir / "input.png"
    _atomic_image(path, data, lossless=True)
    return path


def publish_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

