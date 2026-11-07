"""Fetch real low-light photos for the local demo gallery.

Two sources, both kept in the git-ignored gallery folder and never committed:
- High-resolution night photos under the Unsplash License (free to use, credited).
- LIME benchmark images as redistributed in the upstream Zero-DCE repository.
The first entry is the landing-page hero, so it must be a high-resolution photo.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
UNSPLASH = "https://images.unsplash.com/photo-{}?w=2400&q=90&fm=jpg"
BENCHMARK = "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/data/test_data/{}"
BENCHMARK_SOURCE = "LIME low-light test set, via github.com/Li-Chongyi/Zero-DCE (research benchmark image)."

# (id, url, title, why it is hard, attribution)
PHOTOS = [
    ("tree-path", UNSPLASH.format("1734635701461-0879f73e1c9a"), "Tree-lined path at night",
     "Lamps, foliage and brick facades buried at about 9% brightness.", "Photo by Fons Heijnsbroek on Unsplash (Unsplash License)."),
    ("empty-alley", UNSPLASH.format("1740928980430-93310de220dd"), "Empty alley",
     "A long wall and wiring that vanish into shadow.", "Photo by Julien (domsson) on Unsplash (Unsplash License)."),
    ("rider-alley", UNSPLASH.format("1759248762794-71a284210a55"), "Rider under a wall lamp",
     "About 4% brightness; one lamp against near-black streets.", "Photo by Julien (domsson) on Unsplash (Unsplash License)."),
    ("curtain-window", UNSPLASH.format("1632968491095-8f4d9f57e67a"), "Window with yellow curtain",
     "A glowing window surrounded by dark balconies and plants.", "Photo by Julio Lopez on Unsplash (Unsplash License)."),
    ("venice-canal", BENCHMARK.format("LIME/1.bmp"), "Venice canal at night",
     "Bright window lamps beside water and facades lost in shadow.", BENCHMARK_SOURCE),
    ("courtyard-dusk", BENCHMARK.format("LIME/10.bmp"), "Courtyard at dusk",
     "Wide scene with a street lamp, deep shade and fine paving detail.", BENCHMARK_SOURCE),
]


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    metadata_path = GALLERY / "metadata.json"
    existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    synthetic = [item for item in existing if str(item.get("id", "")).startswith("synthetic")]
    entries = []
    for photo_id, url, title, why_hard, attribution in PHOTOS:
        # Unsplash JPEGs are kept as delivered; benchmark BMPs are stored losslessly as PNG.
        filename = f"{photo_id}.jpg" if "unsplash" in url else f"{photo_id}.png"
        target = GALLERY / filename
        if not target.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "LumenX demo gallery"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if filename.endswith(".jpg"):
                target.write_bytes(payload)
            else:
                Image.open(io.BytesIO(payload)).convert("RGB").save(target, optimize=True)
        entries.append({"id": photo_id, "title": title, "why_hard": why_hard, "attribution": attribution, "image": filename})
        print(f"ready {filename}")
    # Real photos lead; the synthetic scenes remain as a fallback when these are not fetched.
    metadata_path.write_text(json.dumps(entries + synthetic, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
