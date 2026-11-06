"""Aggregate JSONL benchmark runs into API-ready tables and an SVG chart."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    rows = []
    for source in sorted(path.glob("*_evaluation.jsonl")):
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _svg_chart(rows: list[dict[str, Any]]) -> str:
    metric_candidates = ("psnr", "ssim", "niqe", "clipiqa", "latency_ms")
    summaries: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "complete":
            continue
        metrics = row.get("metrics", {})
        ablation = ",".join(row.get("ablate", [])) or "default"
        for metric in metric_candidates:
            value = row.get("latency_ms") if metric == "latency_ms" else metrics.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                summaries[(str(row.get("mode")), ablation, metric)].append(float(value))
    available = [(key, _mean(values)) for key, values in summaries.items()]
    available = [(key, value) for key, value in available if value is not None]
    if not available:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="180"><rect width="100%" height="100%" fill="#101820"/><text x="24" y="90" fill="#e8eef5" font-family="sans-serif" font-size="18">No completed benchmark rows yet</text></svg>'
    metric_order = [metric for metric in metric_candidates if any(key[2] == metric for key, _ in available)]
    width = 900
    panel_height = 150
    height = max(200, 52 + len(metric_order) * panel_height)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#101820"/>']
    colors = {"classical": "#54d6bd", "dl": "#b699ff", "hybrid": "#ffb454"}
    for panel, metric in enumerate(metric_order):
        vals = [(key, value) for key, value in available if key[2] == metric]
        maximum = max(float(value) for _, value in vals) or 1.0
        y0 = 30 + panel * panel_height
        parts.append(f'<text x="24" y="{y0}" fill="#edf4fa" font-family="sans-serif" font-size="16">{metric}</text>')
        bar_y = y0 + 15
        bar_h = min(22, 70 // max(1, len(vals)))
        for index, ((mode, ablation, _), value) in enumerate(vals):
            y = bar_y + index * (bar_h + 5)
            bar_w = max(2.0, 620.0 * float(value) / maximum)
            label = f"{mode} · {ablation} · {float(value):.3f}"
            color = colors.get(mode, "#7aa7ff")
            parts.append(f'<rect x="24" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="5" fill="{color}"/>')
            parts.append(f'<text x="660" y="{y + bar_h - 4}" fill="#cad5df" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    raw_rows = _read(args.results)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("status") == "complete":
            groups[(str(row.get("dataset", "unknown")), str(row.get("mode", "unknown")), ",".join(row.get("ablate", [])) or "default")].append(row)
    metrics = ("psnr", "ssim", "lpips", "niqe", "brisque", "clipiqa", "musiq", "clip_pct_out", "chroma_delta_e", "edge_retention", "noise_sigma_out", "err_recovered", "err_estimated", "err_unrecoverable", "latency_ms")
    summary_rows: list[dict[str, Any]] = []
    for (dataset, mode, ablation), group in sorted(groups.items()):
        output: dict[str, Any] = {"dataset": dataset, "mode": mode, "ablation": ablation, "images": len(group), "paired_images": sum(bool(row.get("paired")) for row in group), "failed_guardrail_runs": sum(bool(row.get("guardrail_failures")) for row in group)}
        for metric in metrics:
            values = [row.get("latency_ms") if metric == "latency_ms" else row.get("metrics", {}).get(metric) for row in group]
            clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
            output[metric] = _mean(clean)
        summary_rows.append(output)
    with (args.results / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = list(summary_rows[0]) if summary_rows else ["dataset", "mode", "ablation", "images", "paired_images", "failed_guardrail_runs", *metrics]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    chart_path = args.results / "benchmark_summary.svg"
    chart_path.write_text(_svg_chart(raw_rows), encoding="utf-8")
    summary = {
        "status": "ready" if summary_rows else "not_ready",
        "rows": len(raw_rows),
        "tables": {"Dataset · method · ablation": summary_rows},
        "charts": [{"url": "/files/results/benchmark_summary.svg", "title": "Benchmark metric averages"}] if summary_rows else [],
        "notice": "Scores average completed images only. Null metrics indicate unavailable measurements.",
    }
    _atomic_json(args.results / "benchmark_summary.json", summary)
    print(f"Wrote {args.results / 'benchmark_summary.json'} and benchmark_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
