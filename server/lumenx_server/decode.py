"""Safe upload decoding, orientation normalization, alpha split, and 12 MP downscale."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class UploadDecodeError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class DecodedImage:
    image: np.ndarray
    alpha: np.ndarray | None
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    downscaled: bool
    format: str


def decode_upload(
    data: bytes,
    *,
    max_upload_bytes: int,
    max_pixels: int,
    hard_max_pixels: int,
) -> DecodedImage:
    """Decode supported raster formats into contiguous normalized RGB arrays."""

    if not data:
        raise UploadDecodeError("empty_upload", "The uploaded file is empty.")
    if len(data) > max_upload_bytes:
        raise UploadDecodeError("upload_too_large", "Compressed upload exceeds the configured limit.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(BytesIO(data))
            image_format = str(source.format or "").upper()
            if image_format not in {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}:
                raise UploadDecodeError("unsupported_media", "Use a PNG, JPEG, WebP, TIFF, or BMP image.", 415)
            width, height = source.size
            if width <= 0 or height <= 0:
                raise UploadDecodeError("invalid_dimensions", "Image dimensions must be positive.")
            pixels = width * height
            if pixels > hard_max_pixels:
                raise UploadDecodeError("decoded_image_too_large", "Decoded image exceeds the hard pixel limit.", 413)
            oriented = ImageOps.exif_transpose(source)
            oriented.load()
    except UploadDecodeError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise UploadDecodeError("decode_failed", "The upload is not a valid supported image.") from exc

    original_size = (int(oriented.width), int(oriented.height))
    alpha: np.ndarray | None = None
    if oriented.mode in ("I;16", "I;16B", "I;16L", "I"):
        raw = np.asarray(oriented)
        gray = np.clip(raw.astype(np.float32), 0.0, 65535.0) / 65535.0
        rgb = np.repeat(gray[..., None], 3, axis=2)
    else:
        if oriented.mode == "P" and "transparency" not in oriented.info:
            oriented = oriented.convert("RGB")
        elif "A" in oriented.getbands() or "transparency" in oriented.info:
            rgba = np.asarray(oriented.convert("RGBA"), dtype=np.uint8)
            rgb = rgba[..., :3].astype(np.float32) / 255.0
            alpha = rgba[..., 3].astype(np.float32) / 255.0
        else:
            rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0
    if rgb.ndim == 2:
        rgb = np.repeat((rgb / 255.0)[..., None], 3, axis=2)
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    rgb = np.ascontiguousarray(np.clip(rgb, 0.0, 1.0))
    if alpha is not None:
        alpha = np.ascontiguousarray(alpha.astype(np.float32))

    h, w = rgb.shape[:2]
    downscaled = h * w > max_pixels
    if downscaled:
        scale = float(np.sqrt(max_pixels / (h * w)))
        out_width = max(1, int(round(w * scale)))
        out_height = max(1, int(round(h * scale)))
        while out_width * out_height > max_pixels:
            if out_width >= out_height:
                out_width -= 1
            else:
                out_height -= 1
        out_size = (out_width, out_height)
        rgb = cv2.resize(rgb, out_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        if alpha is not None:
            alpha = cv2.resize(alpha, out_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        rgb = np.ascontiguousarray(np.clip(rgb, 0.0, 1.0))
        if alpha is not None:
            alpha = np.ascontiguousarray(np.clip(alpha, 0.0, 1.0))
    processed_size = (int(rgb.shape[1]), int(rgb.shape[0]))
    return DecodedImage(rgb, alpha, original_size, processed_size, downscaled, image_format)

