#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def load_header(path: Path) -> dict:
    data = path.read_bytes()
    header_len = struct.unpack("<Q", data[:8])[0]
    return json.loads(data[8 : 8 + header_len].decode("utf-8")), header_len


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--header-dir", required=True, type=Path)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", nargs="+", type=int, default=[0, 2, 201087])
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    embedding = next(t for t in manifest["tensors"] if t["name"] == "model.embed_tokens.weight")
    rows, dim = embedding["shape"]
    row_bytes = embedding["length"] // rows
    source_shard = embedding["source"]["source_shard"]
    header, header_len = load_header(args.header_dir / f"{source_shard.removesuffix('.safetensors')}.header.bin")
    source_start = header["model.embed_tokens.weight"]["data_offsets"][0]
    source_data_base = 8 + header_len

    samples = []
    for token in args.tokens:
        if token < 0 or token >= rows:
            raise ValueError(f"token {token} outside embedding rows")
        virtual_offset = embedding["offset"] + token * row_bytes
        source_offset = source_data_base + source_start + token * row_bytes
        sample_file = args.sample_dir / f"embed_token_{token}.bin"
        sample = {
            "token_id": token,
            "row_bytes": row_bytes,
            "dim": dim,
            "h40m_file_id": embedding["file_id"],
            "h40m_offset": virtual_offset,
            "source_shard": source_shard,
            "source_offset": source_offset,
            "source_length": row_bytes,
            "sample_file": str(sample_file.as_posix()),
            "sample_sha256": sha256(sample_file) if sample_file.exists() else None,
            "sample_present": sample_file.exists(),
        }
        samples.append(sample)

    report = {
        "schema_version": 1,
        "status": "pass" if all(item["sample_present"] for item in samples) else "needs_samples",
        "embedding": {
            "name": embedding["name"],
            "shape": embedding["shape"],
            "dtype": embedding["dtype"],
            "placement": embedding["placement"],
            "row_bytes": row_bytes,
            "full_matrix_bytes": embedding["length"],
        },
        "cache_policy": {
            "type": "bounded_lru_rows",
            "default_capacity_bytes": row_bytes * 8,
            "default_capacity_rows": 8,
        },
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
