#!/usr/bin/env python3
"""Stream an existing H40M arena into a planned physical expert ordering."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

CHUNK = 16 * 1024 * 1024
ZERO_CHUNK = b"\0" * (1024 * 1024)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_slice(handle: BinaryIO, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    handle.seek(offset)
    remaining = length
    while remaining:
        block = handle.read(min(CHUNK, remaining))
        if not block:
            raise EOFError(f"short read at offset {offset}")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def repack(args: argparse.Namespace) -> dict[str, Any]:
    source_layout = load_json(args.source_layout)
    target_layout = load_json(args.target_layout)
    source_records = {
        (int(record["layer"]), int(record["expert_id"])): record
        for record in source_layout["records"]
    }
    target_records = sorted(target_layout["records"], key=lambda record: int(record["offset"]))
    if args.source.stat().st_size != int(source_layout["arena"]["size"]):
        raise ValueError("source arena size does not match source layout")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arena_digest = hashlib.sha256()
    copied = 0
    with args.source.open("rb") as src, args.output.open("wb") as dst:
        cursor = 0
        for target in target_records:
            key = (int(target["layer"]), int(target["expert_id"]))
            source = source_records[key]
            target_offset = int(target["offset"])
            if target_offset < cursor:
                raise ValueError(f"overlapping target record {key}")
            while cursor < target_offset:
                block = ZERO_CHUNK[: min(len(ZERO_CHUNK), target_offset - cursor)]
                dst.write(block)
                arena_digest.update(block)
                cursor += len(block)
            remaining = int(source["length"])
            src.seek(int(source["offset"]))
            record_digest = hashlib.sha256()
            while remaining:
                block = src.read(min(CHUNK, remaining))
                if not block:
                    raise EOFError(f"short source read for {key}")
                dst.write(block)
                arena_digest.update(block)
                record_digest.update(block)
                remaining -= len(block)
                cursor += len(block)
            actual = record_digest.hexdigest()
            if actual != source["sha256"] or actual != target["sha256"]:
                raise ValueError(f"source checksum mismatch for {key}: {actual}")
            copied += 1
        final_size = int(target_layout["arena"]["size"])
        while cursor < final_size:
            block = ZERO_CHUNK[: min(len(ZERO_CHUNK), final_size - cursor)]
            dst.write(block)
            arena_digest.update(block)
            cursor += len(block)

    mismatches: list[dict[str, Any]] = []
    with args.output.open("rb") as handle:
        for target in target_records:
            actual = hash_slice(handle, int(target["offset"]), int(target["length"]))
            if actual != target["sha256"]:
                mismatches.append(
                    {
                        "layer": target["layer"],
                        "expert_id": target["expert_id"],
                        "expected": target["sha256"],
                        "actual": actual,
                    }
                )

    target_layout["arena"]["path"] = args.output.as_posix()
    target_layout["arena"]["sha256"] = arena_digest.hexdigest()
    target_layout["byte_correctness"] = {
        "status": "pass" if not mismatches else "fail",
        "records_checked": len(target_records),
        "records_copied": copied,
        "mismatches": mismatches,
    }
    args.target_layout.write_text(
        json.dumps(target_layout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-layout", type=Path, required=True)
    parser.add_argument("--target-layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = repack(args)
    print(
        json.dumps(
            {
                "status": result["byte_correctness"]["status"],
                "records_checked": result["byte_correctness"]["records_checked"],
                "arena_sha256": result["arena"]["sha256"],
                "arena_size": result["arena"]["size"],
            },
            sort_keys=True,
        )
    )
    if result["byte_correctness"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
