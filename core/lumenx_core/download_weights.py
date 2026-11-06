"""Fetch the authors' optional Zero-DCE++ Epoch99 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen


CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/"
    "Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth"
)


def download(output: Path, expected_sha256: str | None = None) -> str:
    """Download atomically and return the checkpoint SHA-256 digest."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    request = Request(CHECKPOINT_URL, headers={"User-Agent": "LumenX-weight-fetcher/0.4"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise RuntimeError("the checkpoint URL returned an HTML page, not model weights")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                digest.update(block)
                size += len(block)
        if size < 1024:
            raise RuntimeError("downloaded checkpoint is unexpectedly small")
        checksum = digest.hexdigest()
        if expected_sha256 and checksum.lower() != expected_sha256.lower():
            raise RuntimeError(f"SHA-256 mismatch: expected {expected_sha256}, received {checksum}")
        os.replace(temporary, output)
        return checksum
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="checkpoint destination")
    parser.add_argument("--sha256", help="optional expected SHA-256 for verification")
    args = parser.parse_args()
    default_dir = Path(__file__).resolve().parents[1] / "weights"
    output = (args.output or default_dir / "zerodce_pp.pth").expanduser().resolve()
    try:
        checksum = download(output, args.sha256)
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"lumenx-weights: {exc}\n")
    print(f"Saved: {output}")
    print(f"SHA-256: {checksum}")
    print("License: academic, non-commercial use only unless separately licensed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

