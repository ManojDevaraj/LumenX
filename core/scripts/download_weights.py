"""Explicit developer acquisition only: download official files and verify SHA256."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import yaml


def main() -> int:
    """Fetch only manifest entries, never invoked by create_engine or a request."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=root / "weights")
    parser.add_argument("--only", nargs="+", choices=("cidnet_lolv2_real", "cidnet_sid", "zerodce_pp"))
    args = parser.parse_args()
    manifest = yaml.safe_load((root / "lumenx_core/config/checkpoints.yaml").read_text(encoding="utf-8"))
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for key, item in manifest.items():
        if args.only and key not in args.only:
            continue
        path = args.weights_dir / item["filename"]
        if path.exists() and sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            print(f"{key}: already verified", flush=True)
            continue
        partial = path.with_suffix(".download")
        try:
            digest = sha256()
            print(f"{key}: downloading publisher checkpoint", flush=True)
            request = Request(item["url"], headers={"User-Agent": "LumenX weight installer/0.3"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("SHA256 mismatch; refusing to load or publish this checkpoint")
            partial.replace(path)
            print(f"{key}: verified SHA256 {item['sha256']}", flush=True)
        except (OSError, ValueError) as exc:
            failed = True
            if partial.exists():
                partial.unlink()
            print(f"{key}: {exc}\nManual source: {item['url']}\nExpected SHA256: {item['sha256']}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
