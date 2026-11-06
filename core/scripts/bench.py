"""CPU classical milestone benchmark; synthetic inputs are not a quality dataset."""

from __future__ import annotations

import argparse
import cProfile
import ctypes
import json
from pathlib import Path
import platform
import pstats
from time import perf_counter

import numpy as np

from lumenx_core import create_engine
from lumenx_core.contracts import EnhanceRequest, EnhanceResult


def peak_ram_mb() -> float | None:
    """Read actual process peak RSS on Windows without adding a runtime dependency."""
    if platform.system() != "Windows":
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    function = kernel.K32GetProcessMemoryInfo
    function.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    function.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if function(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return counters.peak_rss / 1024**2
    return None


def synthetic_image(index: int) -> np.ndarray:
    """Generate full-resolution flat/noisy/gradient/hotspot cases with fixed seeds."""
    rng = np.random.default_rng(index)
    image = np.full((1080, 1920, 3), 0.04 + 0.005 * index, np.float32)
    if index % 4 == 1:
        image += rng.normal(0, 0.015, image.shape).astype(np.float32)
    elif index % 4 == 2:
        image += np.linspace(0, 0.15, image.shape[1], dtype=np.float32)[None, :, None]
    elif index % 4 == 3:
        image[120:240, 1400:1550] = 0.98
    return np.clip(image, 0, 1)


def profile_hotspots(engine: object) -> str:
    """Profile one separate noisy 1080p pass; do not contaminate benchmark timings."""
    image = synthetic_image(1)
    request = EnhanceRequest(image, (1920, 1080), mode="classical")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in engine.run(request):
        pass
    profiler.disable()
    stats = pstats.Stats(profiler)
    ordered = sorted(stats.stats.items(), key=lambda entry: entry[1][2], reverse=True)[:5]
    lines = ["## Development CPU function profile", "",
             "Separate warm synthetic noisy 1080p classical pass, ranked by exclusive self time.",
             "This is a development hotspot profile, not a hybrid/GPU benchmark or a timing-gate pass.",
             "It does not replace the required full-hybrid v0.4 profile or alter the v0.2 exception.", "",
             "| Function | Calls | Self ms | Cumulative ms |", "|---|---:|---:|---:|"]
    for (filename, line, name), (primitive, calls, own, total, callers) in ordered:
        label = f"{name} ({Path(filename).name}:{line})".replace("|", "/")
        lines.append(f"| {label} | {calls} | {own * 1000:.2f} | {total * 1000:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Report true timings and fail the gate explicitly when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "bench_results.md")
    parser.add_argument("--check-gates", action="store_true")
    parser.add_argument("--profile-only", action="store_true", help="Append a separate top-five function profile without rewriting benchmark data")
    args = parser.parse_args()
    if args.images < 1 or args.passes < 1:
        parser.error("Image and pass counts must be positive.")
    start = perf_counter()
    engine = create_engine(device="cpu")
    engine.warmup()
    cold_ms = (perf_counter() - start) * 1000
    if args.profile_only:
        profile = profile_hotspots(engine)
        previous = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        previous = previous.split("## Development CPU function profile")[0].rstrip()
        args.output.write_text(previous + "\n\n" + profile, encoding="utf-8")
        print(profile, flush=True)
        return 0
    rows: list[dict[str, float]] = []
    for index in range(args.images):
        image = synthetic_image(index)
        request = EnhanceRequest(image, (1920, 1080), mode="classical")
        for _ in range(args.passes):
            for item in engine.run(request):
                if isinstance(item, EnhanceResult):
                    row = dict(item.timings_ms)
                    row["numerical"] = sum(row[key] for key in ("diagnose", "recoverability", "classical", "denoise"))
                    rows.append(row)
            del item
        print(f"Completed {index + 1}/{args.images} synthetic 1080p images", flush=True)
    statistics = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        statistics[key] = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}
    passed = statistics["numerical"]["p95"] < 100 and statistics["diagnose"]["p95"] < 30
    lines = ["# v0.2.0 CPU classical benchmark", "", f"Python {platform.python_version()}; {platform.platform()}",
             f"Processor: {platform.processor()}", f"Synthetic cases: {args.images}; warm passes: {args.passes}; samples: {len(rows)}",
             "All inputs are 1920x1080. These are deterministic performance cases, not real photos or quality benchmarks.",
             f"Cold factory + warmup: {cold_ms:.2f} ms", f"Peak process RAM: {peak_ram_mb()} MiB",
             "GPU VRAM / hybrid inference: not applicable, absent in this milestone.",
             "Policy: CPU resident; no perf levers applied; full input resolution preserved.", "",
             "| Stage | Mean ms | p95 ms | Maximum ms |", "|---|---:|---:|---:|"]
    lines.extend(f"| {key} | {value['mean']:.2f} | {value['p95']:.2f} | {value['max']:.2f} |" for key, value in statistics.items())
    lines.extend(["", "Numerical = diagnosis + recoverability + classical + denoise, excluding render, metrics and report.",
                  f"Release performance gate: {'PASS' if passed else 'FAIL'} (numerical p95 <100 ms and diagnosis p95 <30 ms).",
                  "core_total additionally includes required full-resolution visual array generation and remaining core work.",
                  "Not an RTX 2050 GPU measurement; no end-to-end server latency claim.", "", "## Observed health", "", "```json",
                  json.dumps(engine.health(), indent=2), "```", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}; gate {'PASS' if passed else 'FAIL'}", flush=True)
    return 1 if args.check_gates and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
