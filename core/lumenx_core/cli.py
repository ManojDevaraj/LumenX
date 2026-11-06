"""Development-only image runner for inspecting engine visuals locally."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import create_engine, create_fake_engine
from .contracts import EnhanceRequest, EnhanceResult, StageEvent, Visual
from .contract_check import validate_run


def _read_image(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(f"Could not read image: {path}")
    if not np.issubdtype(decoded.dtype, np.integer):
        raise ValueError("Input image must use an integer pixel format (8-bit or 16-bit).")
    scale = float(np.iinfo(decoded.dtype).max)
    alpha: np.ndarray | None = None
    if decoded.ndim == 2:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    elif decoded.shape[2] == 4:
        bgra = decoded
        alpha = bgra[..., 3].astype(np.float32) / scale
        rgb = cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2RGB)
    elif decoded.shape[2] == 3:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Input must decode as grayscale, RGB, or RGBA.")
    return np.ascontiguousarray(rgb.astype(np.float32) / scale), alpha


def _write_visual(visual: Visual, output: Path, *, final_path: Path | None = None) -> Path:
    if final_path is not None:
        path = final_path
    else:
        suffix = ".png" if visual.lossless else ".jpg"
        path = output.with_name(f"{output.stem}.{visual.key}{suffix}")
    image = visual.image
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if path.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Could not write visual: {path}")
    return path


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def _write_result(result: EnhanceResult, paths: dict[str, Path], output: Path) -> Path:
    payload: dict[str, Any] = {}
    for field in fields(result):
        value = getattr(result, field.name)
        if field.name == "visuals":
            payload[field.name] = [
                {
                    "key": visual.key,
                    "role": visual.role,
                    "path": str(paths[visual.key]),
                    "max_side": visual.max_side,
                    "lossless": visual.lossless,
                }
                for visual in result.visuals
            ]
        else:
            payload[field.name] = _to_jsonable(value)
    path = output.with_name(f"{output.stem}.result.json")
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _enhance(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image, alpha = _read_image(input_path)
    if args.fake:
        engine = create_fake_engine(seed=args.seed)
    else:
        engine = create_engine(config_path=args.config, device=args.device)
    engine.warmup()
    request = EnhanceRequest(
        image=image,
        original_size=(image.shape[1], image.shape[0]),
        alpha=alpha,
        mode=args.mode,
        seed=args.seed,
    )
    items = list(engine.run(request))
    violations = validate_run(items, request)
    if violations:
        raise RuntimeError("Stream contract violations:\n- " + "\n- ".join(violations))

    paths: dict[str, Path] = {}
    for item in items:
        visuals = item.visuals if isinstance(item, StageEvent) else item.visuals
        for visual in visuals:
            final_path = output_path if visual.key == "final" else None
            paths[visual.key] = _write_visual(visual, output_path, final_path=final_path)
    result = items[-1]
    if not isinstance(result, EnhanceResult):
        raise RuntimeError("Engine stream ended without an EnhanceResult.")
    result_path = _write_result(result, paths, output_path)
    print(result.headline)
    print(f"Final image: {paths['final']}")
    print(f"Run report: {result_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LumenX on one image for local inspection.")
    commands = parser.add_subparsers(dest="command", required=True)
    enhance = commands.add_parser("enhance", help="enhance one image")
    enhance.add_argument("input", help="input image file")
    enhance.add_argument("-o", "--output", required=True, help="final output image path")
    enhance.add_argument(
        "--mode",
        choices=("hybrid", "classical", "dl", "zerodce"),
        default="classical",
    )
    enhance.add_argument("--device", default="cpu")
    enhance.add_argument("--config", help="optional YAML configuration path")
    enhance.add_argument("--seed", type=int, default=0)
    enhance.add_argument("--fake", action="store_true", help="use FakeEngine for a deterministic preview")
    enhance.set_defaults(handler=_enhance)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except Exception as exc:  # CLI boundary: present the engine error without a traceback.
        parser.exit(2, f"lumenx-core: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
