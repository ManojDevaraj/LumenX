"""Measure end-to-end request latency against a running local LumenX API."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import IMAGE_SUFFIXES


def _multipart(path: Path, mode: str) -> tuple[bytes, str]:
    boundary = "lumenx-" + uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    file_header = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    mode_field = f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{mode}".encode()
    ablate_field = f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"ablate\"\r\n\r\n[]".encode()
    body = file_header + path.read_bytes() + mode_field + ablate_field + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("classical", "dl", "hybrid"), default="hybrid")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "server_latency.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="Number of initial requests excluded from timing")
    args = parser.parse_args()
    files = sorted(path for path in args.images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        parser.error(f"no supported images found under {args.images}")
    warmup_count = max(0, min(args.warmup, len(files) - 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(files, start=1):
        data, content_type = _multipart(path, args.mode)
        request = Request(args.url.rstrip("/") + "/enhance", data=data, headers={"Content-Type": content_type}, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
                status = response.status
            row = {"image": str(path), "status": status, "latency_ms": (time.perf_counter() - started) * 1000.0, "server_total_ms": payload.get("timings_ms", {}).get("total"), "actual_method": payload.get("provenance", {}).get("actual_method"), "error": ""}
            if index > warmup_count:
                rows.append(row)
                print(f"[{index}/{len(files)}] {path.name} · {row['latency_ms']:.1f} ms")
            else:
                print(f"[{index}/{len(files)}] {path.name} · warm-up")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            row = {"image": str(path), "status": "error", "latency_ms": (time.perf_counter() - started) * 1000.0, "server_total_ms": None, "actual_method": None, "error": str(exc)}
            if index > warmup_count:
                rows.append(row)
            print(f"[{index}/{len(files)}] {path.name} · ERROR: {exc}")
    fields = ["image", "status", "latency_ms", "server_total_ms", "actual_method", "error"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Latency rows written to {args.output}; this captures the active server and machine configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
