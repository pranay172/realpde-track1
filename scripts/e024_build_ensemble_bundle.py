#!/usr/bin/env python3
"""Build or verify the E024 shared-backbone ensemble bundle from member checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "realpde_t1_starting_kit_v9"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(KIT))

from realpde_t1.ensemble import (  # noqa: E402
    assert_bundle_matches_sources,
    build_ensemble_bundle,
    load_member_checkpoints,
)

DEFAULT_SOURCES = [
    REPO / "artifacts" / "e019_residual_adapter_e005" / "final.pth",
    REPO / "artifacts" / "e022_heteroskedastic_residual" / "final.pth",
    REPO / "artifacts" / "e023_adapter3" / "final.pth",
]
DEFAULT_OUTPUT = REPO / "artifacts" / "e023_ensemble" / "ensemble_bundle.pth"
EXPECTED_SHA256 = "dfbd15d93d8b6324e06b6d5e4f66ad004e5441b825b1708465719f71dc0ff10b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing bundle against sources; do not write",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write a new bundle (refuses to replace the frozen E024 artifact)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.check == args.write:
        raise ValueError("specify exactly one of --check or --write")
    missing = [str(path) for path in args.sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing ensemble member checkpoints:\n" + "\n".join(missing))
    checkpoints, source_hashes = load_member_checkpoints(args.sources)
    bundle = build_ensemble_bundle(checkpoints, source_sha256=source_hashes)
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(f"missing ensemble bundle: {args.output}")
        existing = torch.load(args.output, map_location="cpu", weights_only=False)
        assert_bundle_matches_sources(existing, checkpoints)
        report = {
            "status": "passed",
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
            "source_checkpoint_sha256": source_hashes,
            "n_adapters": len(existing["adapters"]),
            "matches_frozen_e024_hash": sha256_file(args.output) == EXPECTED_SHA256,
        }
        print(json.dumps(report, indent=2))
        return
    if args.output.resolve() == DEFAULT_OUTPUT.resolve() and args.output.is_file():
        raise FileExistsError(
            f"refusing to replace the frozen E024 ensemble bundle: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "source_checkpoint_sha256": source_hashes,
                "n_adapters": len(bundle["adapters"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
