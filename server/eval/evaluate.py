"""Run resumable paired/unpaired and ablation evaluation on a local dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "core"))
sys.path.insert(0, str(REPO))

from lumenx_core import create_engine  # noqa: E402
from lumenx_core.contracts import EnhanceRequest, EnhanceResult, StageEvent  # noqa: E402
from lumenx_core.contract_check import validate_run  # noqa: E402
from common import align_reference, as_rgb_u8, decode_path, discover_records  # noqa: E402


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("key"):
                rows[str(row["key"])] = row
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _variants(modes: list[str], ablations: list[str]) -> list[tuple[str, frozenset[str]]]:
    variants = [(mode, frozenset()) for mode in modes]
    variants.extend(("hybrid", frozenset({flag})) for flag in ablations)
    return variants


def _run_one(
    engine: Any,
    record: Any,
    mode: str,
    ablate: frozenset[str],
    preview_path: Path | None = None,
) -> dict[str, Any]:
    decoded = decode_path(record.input_path)
    image = decoded.image
    reference = None
    if record.reference_path:
        reference = as_rgb_u8(decode_path(record.reference_path))
        reference = align_reference(reference, image.shape[:2])
    request = EnhanceRequest(
        image=image,
        original_size=decoded.original_size,
        alpha=decoded.alpha,
        mode=mode,
        ablate=ablate,
        seed=0,
    )
    started = time.perf_counter()
    items = list(engine.run(request))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    violations = validate_run(items, request)
    if violations:
        raise RuntimeError("Core stream contract violations: " + "; ".join(violations))
    result = items[-1]
    if not isinstance(result, EnhanceResult):
        raise RuntimeError("Engine stream ended without an EnhanceResult")
    final_visual = next((visual for visual in result.visuals if visual.key == "final"), None)
    if final_visual is None:
        raise RuntimeError("Engine did not return a final image")
    output = final_visual.image[..., :3]
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(preview_path, format="PNG", optimize=False)
    suite = engine.metric_suite()
    input_u8 = as_rgb_u8(decoded)
    custom = suite.custom(input_u8, output)
    unpaired = suite.unpaired(output)
    paired = suite.paired(output, reference) if reference is not None else {}
    confidence = suite.confidence_errors(input_u8, output, reference) if reference is not None else {}
    return {
        "status": "complete",
        "relative_path": record.relative_path,
        "mode": mode,
        "ablate": sorted(ablate),
        "actual_method": result.provenance.get("actual_method"),
        "fallback_chain": result.provenance.get("fallback_chain", []),
        "paired": reference is not None,
        "metrics": {**paired, **unpaired, **custom, **confidence},
        "stage_timings_ms": result.timings_ms,
        "latency_ms": elapsed_ms,
        "guardrail_failures": [check.check for check in result.guardrails if check.passed is False],
        "warnings": [warning.code for warning in result.warnings],
        "metric_unavailable": suite.unavailable(),
        "provenance": result.provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset label used in output rows")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root")
    parser.add_argument("--input-dir", type=Path, help="Optional low-light image directory")
    parser.add_argument("--reference-dir", type=Path, help="Optional ground-truth directory")
    parser.add_argument("--paired", action="store_true", help="Require matched reference images")
    parser.add_argument("--modes", default="classical,dl,hybrid", help="Comma-separated processing modes")
    parser.add_argument("--ablations", default="", help="Comma-separated hybrid ablation flags")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "server" / "results")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of source images")
    parser.add_argument("--save-images", action="store_true", help="Save final images under results/previews for manual failure review")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes or any(mode not in {"classical", "dl", "hybrid", "zerodce"} for mode in modes):
        parser.error("--modes must contain classical, dl, hybrid and/or zerodce")
    allowed_flags = {"no_tuner", "no_fusion", "no_classical", "no_dl", "no_gain_denoise", "no_recoverability", "no_guardrails", "no_color_correction"}
    ablations = [flag.strip() for flag in args.ablations.split(",") if flag.strip()]
    invalid = set(ablations) - allowed_flags
    if invalid:
        parser.error("unknown ablation flag(s): " + ", ".join(sorted(invalid)))

    records = discover_records(
        args.root,
        input_dir=args.input_dir,
        reference_dir=args.reference_dir,
        paired=args.paired,
    )
    if args.limit > 0:
        records = records[: args.limit]
    variants = _variants(modes, ablations)
    args.output.mkdir(parents=True, exist_ok=True)
    safe_dataset = re.sub(r"[^A-Za-z0-9_-]+", "_", args.dataset).strip("_") or "dataset"
    output_file = args.output / f"{safe_dataset}_evaluation.jsonl"
    progress_file = args.output / "progress.json"
    existing = _read_rows(output_file)
    total = len(records) * len(variants)
    engine = create_engine(
        config_path=str(args.config) if args.config else None,
        device=args.device,
    )
    engine.warmup()
    completed = 0
    errors = sum(row.get("status") == "error" for row in existing.values())
    with output_file.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            for mode, ablate in variants:
                key = "|".join((args.dataset, record.relative_path, mode, ",".join(sorted(ablate))))
                completed += 1
                if key in existing:
                    continue
                started = time.perf_counter()
                row: dict[str, Any] = {
                    "key": key,
                    "dataset": args.dataset,
                    "relative_path": record.relative_path,
                    "mode": mode,
                    "ablate": sorted(ablate),
                    "created_at": time.time(),
                }
                try:
                    preview_path = None
                    if args.save_images:
                        ablation_label = ",".join(sorted(ablate)) or "default"
                        preview_path = args.output / "previews" / safe_dataset / mode / ablation_label / Path(record.relative_path).with_suffix(".png")
                    row.update(_run_one(engine, record, mode, ablate, preview_path))
                except Exception as exc:
                    errors += 1
                    row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                row["preview"] = (
                    str(preview_path.relative_to(args.output))
                    if preview_path is not None and preview_path.is_file()
                    else None
                )
                row["wall_ms"] = (time.perf_counter() - started) * 1000.0
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                _atomic_json(
                    progress_file,
                    {
                        "dataset": args.dataset,
                        "completed": completed,
                        "total": total,
                        "errors": errors,
                        "output": output_file.name,
                    },
                )
                print(f"[{completed}/{total}] {record.relative_path} · {mode} · {row['status']}")
    _atomic_json(
        progress_file,
        {
            "dataset": args.dataset,
            "completed": total,
            "total": total,
            "errors": errors,
            "output": output_file.name,
        },
    )
    print(f"Evaluation rows: {output_file}")
    print("Run `python server/eval/aggregate.py` to refresh tables and charts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
