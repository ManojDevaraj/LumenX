"""Rank failed and weak-signal evaluation rows for manual inspection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.results.glob("*_evaluation.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics", {})
            confidence_error = max(
                (float(metrics.get(name) or 0.0) for name in ("err_recovered", "err_estimated", "err_unrecoverable")),
                default=0.0,
            )
            failures = list(row.get("guardrail_failures", []))
            if row.get("status") == "error" or failures or confidence_error > 0:
                rows.append({
                    "dataset": row.get("dataset"),
                    "relative_path": row.get("relative_path"),
                    "mode": row.get("mode"),
                    "ablation": ",".join(row.get("ablate", [])) or "default",
                    "status": row.get("status"),
                    "guardrail_failures": ",".join(failures),
                    "max_confidence_error": confidence_error,
                    "latency_ms": row.get("latency_ms"),
                    "preview": row.get("preview", ""),
                    "error": row.get("error", ""),
                })
    rows.sort(key=lambda row: (row["status"] != "error", not bool(row["guardrail_failures"]), -row["max_confidence_error"]))
    output = args.results / "failure_review.csv"
    fields = ["dataset", "relative_path", "mode", "ablation", "status", "guardrail_failures", "max_confidence_error", "latency_ms", "preview", "error"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[: max(1, args.limit)])
    print(f"Wrote {min(len(rows), max(1, args.limit))} ranked rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
