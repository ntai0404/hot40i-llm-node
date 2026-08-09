#!/usr/bin/env python3
"""Repack gpt-oss expert tensors into deterministic per-expert arena records."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[2]
PART_ORDER = (
    "down_proj_bias",
    "down_proj_blocks",
    "down_proj_scales",
    "gate_up_proj_bias",
    "gate_up_proj_blocks",
    "gate_up_proj_scales",
)
COPY_CHUNK = 16 * 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_index(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    data_base = 8 + header_len
    result: dict[str, dict[str, Any]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        result[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "absolute_start": data_base + int(start),
            "absolute_end": data_base + int(end),
            "length": int(end) - int(start),
        }
    return result


def part_name(tensor_name: str) -> str:
    for part in PART_ORDER:
        if tensor_name.endswith(part):
            return part
    raise ValueError(f"unknown expert tensor part: {tensor_name}")


def copy_slice(src: BinaryIO, dst: BinaryIO, offset: int, length: int, digest: hashlib._Hash) -> None:
    src.seek(offset)
    remaining = length
    while remaining:
        chunk = src.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise EOFError(f"unexpected EOF at source offset {offset}")
        dst.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def read_slice(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        raise EOFError(f"short read from {path} at {offset}")
    return data


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir
    inventory = load_json(args.inventory)
    storage = load_json(args.storage)

    shard_paths = {shard["file"]: source_dir / shard["file"] for shard in inventory["shards"]}
    for shard in inventory["shards"]:
        path = shard_paths[shard["file"]]
        if not path.exists():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(shard["size"]):
            raise ValueError(f"{path} size {size} != inventory {shard['size']}")

    shard_sha256 = {name: sha256_file(path) for name, path in sorted(shard_paths.items())}
    for shard in inventory["shards"]:
        expected = shard.get("x_linked_etag")
        actual = shard_sha256[shard["file"]]
        if expected and actual != expected:
            raise ValueError(f"{shard['file']} sha256 {actual} != {expected}")

    tensor_indexes = {name: safetensors_index(path) for name, path in shard_paths.items()}
    source_handles = {name: path.open("rb") for name, path in shard_paths.items()}

    expert_tensors = [t for t in inventory["tensors"] if t["role"] == "expert"]
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for tensor in expert_tensors:
        shape = tensor["shape"]
        if not shape or int(shape[0]) != inventory["config"]["num_local_experts"]:
            raise ValueError(f"expert tensor lacks first expert axis: {tensor['name']} {shape}")
        shard = tensor["source_shard"]
        indexed = tensor_indexes[shard][tensor["name"]]
        if indexed["length"] != int(tensor["byte_size"]):
            raise ValueError(f"indexed length mismatch for {tensor['name']}")
        per_expert = int(tensor["byte_size"]) // int(shape[0])
        part = part_name(tensor["name"])
        for expert_id in range(int(shape[0])):
            grouped.setdefault((int(tensor["layer"]), expert_id), {})[part] = {
                "tensor": tensor["name"],
                "source_shard": shard,
                "source_offset": indexed["absolute_start"] + expert_id * per_expert,
                "length": per_expert,
                "dtype": tensor["dtype"],
                "shape": shape[1:],
            }

    expected_records = inventory["config"]["num_hidden_layers"] * inventory["config"]["num_local_experts"]
    if len(grouped) != expected_records:
        raise ValueError(f"expected {expected_records} expert records, got {len(grouped)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    arena_digest = hashlib.sha256()
    zero_chunk = b"\0" * (1024 * 1024)

    with args.out.open("wb") as dst:
        for layer, expert_id in sorted(grouped):
            offset = align(dst.tell(), args.alignment)
            while dst.tell() < offset:
                dst.write(zero_chunk[: min(len(zero_chunk), offset - dst.tell())])
            record_digest = hashlib.sha256()
            parts = []
            for part in PART_ORDER:
                source = grouped[(layer, expert_id)].get(part)
                if source is None:
                    raise ValueError(f"missing {part} for layer {layer} expert {expert_id}")
                part_offset = dst.tell() - offset
                src_handle = source_handles[source["source_shard"]]
                copy_slice(src_handle, dst, source["source_offset"], source["length"], record_digest)
                parts.append({**source, "arena_relative_offset": part_offset})
            length = dst.tell() - offset
            records.append(
                {
                    "layer": layer,
                    "expert_id": expert_id,
                    "file": args.out.name,
                    "offset": offset,
                    "length": length,
                    "sha256": record_digest.hexdigest(),
                    "parts": parts,
                }
            )
        final_size = dst.tell()

    for handle in source_handles.values():
        handle.close()
    arena_sha256 = sha256_file(args.out)

    sample_records = [0, len(records) // 2, len(records) - 1]
    rng = random.Random(4002)
    sample_records.extend(rng.sample(range(len(records)), 13))
    sample_checks = []
    for index in sorted(set(sample_records)):
        record = records[index]
        for part in record["parts"]:
            source_bytes = read_slice(
                shard_paths[part["source_shard"]],
                part["source_offset"],
                part["length"],
            )
            arena_bytes = read_slice(
                args.out,
                record["offset"] + part["arena_relative_offset"],
                part["length"],
            )
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            arena_sha = hashlib.sha256(arena_bytes).hexdigest()
            sample_checks.append(
                {
                    "layer": record["layer"],
                    "expert_id": record["expert_id"],
                    "part": part["tensor"],
                    "length": part["length"],
                    "source_sha256": source_sha,
                    "arena_sha256": arena_sha,
                    "matched": source_sha == arena_sha,
                }
            )

    source_bytes_total = sum(int(t["byte_size"]) for t in expert_tensors)
    padding_bytes = final_size - source_bytes_total
    random_16m = storage["metrics"]["summary_by_pattern_block"]["random_16777216"]
    random_8m = storage["metrics"]["summary_by_pattern_block"]["random_8388608"]
    layout = {
        "schema_version": 1,
        "format": "H40M_EXPERT_ARENA/1",
        "source_model": inventory["model"],
        "source_ref": inventory["repo_commit"],
        "source_shards": [
            {
                "file": shard["file"],
                "size": shard["size"],
                "sha256": shard_sha256[shard["file"]],
                "expected_sha256": shard.get("x_linked_etag"),
            }
            for shard in inventory["shards"]
        ],
        "arena": {
            "path": args.out.as_posix(),
            "alignment": args.alignment,
            "size": final_size,
            "sha256": arena_sha256,
            "source_expert_bytes": source_bytes_total,
            "padding_bytes": padding_bytes,
            "record_count": len(records),
            "record_payload_bytes": records[0]["length"],
        },
        "records": records,
        "sample_checks": sample_checks,
    }
    args.layout.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    benchmark = {
        "schema_version": 1,
        "status": "pass" if all(item["matched"] for item in sample_checks) else "fail",
        "arena_path": args.out.as_posix(),
        "alignment_bytes": args.alignment,
        "record_count": len(records),
        "record_payload_bytes": records[0]["length"],
        "arena_size_bytes": final_size,
        "source_expert_bytes": source_bytes_total,
        "padding_bytes": padding_bytes,
        "sample_check_count": len(sample_checks),
        "sample_checks_passed": sum(1 for item in sample_checks if item["matched"]),
        "d04_basis": {
            "random_8m_mib_per_second_median": random_8m["mib_per_second_median"],
            "random_16m_mib_per_second_median": random_16m["mib_per_second_median"],
            "alignment_decision": "1MiB expert-record alignment; D04 shows expert-sized 8/16MiB random reads exceed 1.1GiB/s while 1MiB avoids multi-GB padding.",
        },
    }
    args.benchmark.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=ROOT / "artifacts/model/gpt_oss_20b_inventory.json")
    parser.add_argument("--storage", type=Path, default=ROOT / "benchmarks/stock/storage.json")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/model/source")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/model/h40m/expert_arena.bin")
    parser.add_argument("--layout", type=Path, default=ROOT / "artifacts/model/h40m/expert_arena.layout.json")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmarks/model/h40m_layout.json")
    parser.add_argument("--alignment", type=int, default=1024 * 1024)
    args = parser.parse_args()
    benchmark = build(args)
    print(json.dumps({
        "status": benchmark["status"],
        "arena_size_bytes": benchmark["arena_size_bytes"],
        "record_count": benchmark["record_count"],
        "sample_checks_passed": benchmark["sample_checks_passed"],
        "sample_check_count": benchmark["sample_check_count"],
    }, sort_keys=True))
    if benchmark["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
