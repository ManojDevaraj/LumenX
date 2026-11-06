"""Explicit startup diagnostic: warm loaded models and print actual engine health."""

from __future__ import annotations

import argparse
import json

from lumenx_core import create_engine


def main() -> int:
    """Require verified publisher weights only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()
    engine = create_engine(device=args.device)
    engine.warmup()
    health = engine.health()
    print(json.dumps(health, indent=2))
    if args.require_models:
        loaded = set(health["models"]["cidnet"]["checkpoints"])
        return int(not ({"cidnet_lolv2_real", "cidnet_sid"} <= loaded and health["models"]["zerodce"]["available"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
