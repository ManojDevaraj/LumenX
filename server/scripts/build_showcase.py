"""Record one real LumenX run as static files for the landing page.

The landing scene must open instantly (and work on a static host), so it plays
back a recorded run instead of restoring on page load. Everything written here
is genuine engine output from a running server; the manifest records the
method, timings and provenance so the page can label it truthfully.

Usage (server running):  python server/scripts/build_showcase.py --server http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

FRONTEND = Path(__file__).resolve().parents[1] / "lumenx_server" / "frontend"
OUTPUT = FRONTEND / "showcase"
# Kept under one megapixel so the engine processes it in a single pass; larger inputs are tiled.
PHOTO = {
    "url": "https://images.unsplash.com/photo-1455593984172-9f753a2e1ebd?w=1200&q=92&fm=jpg",
    "title": "Salon at dusk",
    "attribution": "Photo by Abigail Keenan on Unsplash (Unsplash License).",
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LumenX showcase builder"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def enhance(server: str, image: bytes, mode: str) -> dict:
    boundary = uuid.uuid4().hex
    fields = [("mode", mode), ("ablate", "[]")]
    body = b"".join(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode() for name, value in fields
    )
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
    body += image + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(f"{server}/enhance", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def save_jpeg(payload: bytes, target: Path, quality: int = 90) -> None:
    Image.open(io.BytesIO(payload)).convert("RGB").save(target, quality=quality, optimize=True, progressive=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument("--mode", default="hybrid", choices=("hybrid", "classical", "dl"))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    original = fetch(PHOTO["url"])
    save_jpeg(original, OUTPUT / "original.jpg", quality=92)
    result = enhance(args.server, original, args.mode)
    if result.get("status") not in (None, "ok", "done", "complete") and "final" not in result:
        raise SystemExit(f"Engine run failed: {result}")

    save_jpeg(fetch(args.server + result["final"]["url"]), OUTPUT / "restored.jpg", quality=92)
    maps = result.get("maps") or {}
    height_url = maps.get("heightmap") or maps.get("illumination_gray")
    Image.open(io.BytesIO(fetch(args.server + height_url))).convert("L").save(OUTPUT / "heightmap.png", optimize=True)

    stages = []
    for index, stage in enumerate(item for item in result["stages"] if item.get("image_url")):
        filename = f"stage-{index + 1:02d}.jpg"
        save_jpeg(fetch(args.server + stage["image_url"]), OUTPUT / filename, quality=86)
        stages.append({"name": stage["name"], "title": stage.get("title") or stage["name"], "caption": stage.get("caption") or "", "image": filename})

    provenance = result.get("provenance", {})
    manifest = {
        "title": PHOTO["title"],
        "attribution": PHOTO["attribution"],
        "mode": result.get("mode", args.mode),
        "method": provenance.get("actual_method"),
        "device": provenance.get("device"),
        "fake": bool(provenance.get("fake")),
        "core_version": provenance.get("core_version"),
        "total_ms": round(float(result.get("timings_ms", {}).get("total", 0))),
        "input": {"width": result["input"]["width"], "height": result["input"]["height"]},
        "original": "original.jpg",
        "restored": "restored.jpg",
        "heightmap": "heightmap.png",
        "stages": stages,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("method", "total_ms", "input", "core_version")}), f"{len(stages)} stages")


if __name__ == "__main__":
    main()
