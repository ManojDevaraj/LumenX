"""Developer-only file reader/writer; production image I/O belongs to server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import create_engine
from .contracts import EnhanceRequest, EnhanceResult, LumenXError


def main() -> int:
    """Write natural-size debugging visuals with plain OpenCV image I/O."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    enhance = sub.add_parser("enhance")
    enhance.add_argument("image", type=Path)
    enhance.add_argument("outdir", type=Path)
    enhance.add_argument("--mode", choices=("classical", "hybrid", "dl", "zerodce"), default="classical")
    enhance.add_argument("--config")
    args = parser.parse_args()
    source = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if source is None or source.dtype != np.uint8:
        parser.error("Debug CLI requires a decodable uint8 image; production decoding belongs to server.")
    alpha = None
    if source.ndim == 2:
        rgb = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    elif source.shape[2] == 4:
        rgb = cv2.cvtColor(source[..., :3], cv2.COLOR_BGR2RGB)
        alpha = source[..., 3].astype(np.float32) / 255
    else:
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(rgb.astype(np.float32) / 255)
    request = EnhanceRequest(image, (image.shape[1], image.shape[0]), alpha=alpha, mode=args.mode)
    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        engine = create_engine(args.config)
        engine.warmup()
        stream = engine.run(request)
        try:
            for item in stream:
                for visual in item.visuals:
                    array = visual.image
                    if array.ndim == 3:
                        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA if array.shape[2] == 4 else cv2.COLOR_RGB2BGR)
                    path = args.outdir / f"{visual.key}.{'png' if visual.lossless else 'jpg'}"
                    if not cv2.imwrite(str(path), array):
                        raise OSError(f"Could not write {path}")
                if isinstance(item, EnhanceResult):
                    print(json.dumps({"headline": item.headline, "timings_ms": item.timings_ms,
                                      "warnings": [warning.message for warning in item.warnings]}, indent=2))
        finally:
            stream.close()
    except (LumenXError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
