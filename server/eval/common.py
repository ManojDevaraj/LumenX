"""Shared dataset discovery and image conversion for evaluation commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lumenx_server.decode import DecodedImage, decode_upload


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    input_path: Path
    reference_path: Path | None


def _images(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).casefold(),
    )


def _discover_input_dir(root: Path) -> Path:
    preferred = (
        "test/low",
        "eval/low",
        "test/low light",
        "real_captured/test/low",
        "synthetic/test/low",
        "low",
    )
    by_lower = {str(path.relative_to(root)).replace("\\", "/").lower(): path for path in root.rglob("*") if path.is_dir()}
    for suffix in preferred:
        for relative, path in by_lower.items():
            if relative.endswith(suffix):
                return path
    low_dirs = [path for path in by_lower.values() if path.name.lower() in {"low", "lowlight", "low_light"}]
    if len(low_dirs) == 1:
        return low_dirs[0]
    return root


def _discover_reference_dir(input_dir: Path) -> Path | None:
    parent = input_dir.parent
    names = {path.name.lower(): path for path in parent.iterdir() if path.is_dir()}
    for name in ("high", "normal", "gt", "ground_truth", "reference", "target"):
        if name in names:
            return names[name]
    return None


def discover_records(
    root: Path,
    *,
    input_dir: Path | None = None,
    reference_dir: Path | None = None,
    paired: bool = False,
) -> list[ImageRecord]:
    """Find low-light images and optionally match references by relative path/stem."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    low = (input_dir or _discover_input_dir(root)).expanduser().resolve()
    if not low.is_dir():
        raise FileNotFoundError(f"Input image directory does not exist: {low}")
    high = reference_dir.expanduser().resolve() if reference_dir else _discover_reference_dir(low)
    if paired and (high is None or not high.is_dir()):
        raise FileNotFoundError(
            "Paired evaluation needs a reference directory. Pass --reference-dir or use sibling Low/High or Low/Normal folders."
        )
    reference_by_stem: dict[str, Path] = {}
    if high is not None and high.is_dir():
        for path in _images(high):
            reference_by_stem.setdefault(path.stem.casefold(), path)
    records: list[ImageRecord] = []
    unmatched = 0
    for path in _images(low):
        relative = str(path.relative_to(low)).replace("\\", "/")
        ref = None
        if high is not None:
            candidate = high / Path(relative)
            if candidate.is_file():
                ref = candidate
            else:
                ref = reference_by_stem.get(path.stem.casefold())
        if paired and ref is None:
            unmatched += 1
            continue
        records.append(ImageRecord(relative, path, ref))
    if paired and unmatched:
        raise FileNotFoundError(f"{unmatched} input image(s) under {low} have no matching reference under {high}")
    if not records:
        raise FileNotFoundError(f"No supported images found under {low}")
    if paired and not any(record.reference_path is not None for record in records):
        raise FileNotFoundError(f"No paired references matched images under {low}")
    return records


def decode_path(path: Path, *, max_pixels: int = 12_000_000) -> DecodedImage:
    return decode_upload(
        path.read_bytes(),
        max_upload_bytes=max(path.stat().st_size, 1),
        max_pixels=max_pixels,
        hard_max_pixels=max(max_pixels, 50_000_000),
    )


def as_rgb_u8(decoded: DecodedImage) -> np.ndarray:
    return np.rint(np.clip(decoded.image, 0.0, 1.0) * 255.0).astype(np.uint8)


def align_reference(reference: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if reference.shape[:2] != (height, width):
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(reference.astype(np.uint8, copy=False))
