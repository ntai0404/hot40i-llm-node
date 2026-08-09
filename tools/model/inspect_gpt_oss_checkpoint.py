#!/usr/bin/env python3
"""Inventory the official gpt-oss-20b checkpoint from safetensors headers."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any


SHARDS = [
    "model-00000-of-00002.safetensors",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safetensors_header(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path} is too short for a safetensors header")
    header_len = struct.unpack("<Q", data[:8])[0]
    end = 8 + header_len
    if len(data) < end:
        raise ValueError(f"{path} contains {len(data)} bytes but header needs {end}")
    return json.loads(data[8:end].decode("utf-8"))


def parse_headers(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.lower()] = value.strip().strip('"')
    return values


def classify(name: str) -> tuple[int | None, str, int | None]:
    layer_match = re.search(r"model\.layers\.(\d+)\.", name)
    layer = int(layer_match.group(1)) if layer_match else None
    expert_match = re.search(r"\.experts\.(\d+)\.", name)
    expert = int(expert_match.group(1)) if expert_match else None

    if name.startswith("model.embed_tokens"):
        role = "embedding"
    elif name.startswith("lm_head"):
        role = "lm_head"
    elif "self_attn" in name:
        role = "attention"
    elif "input_layernorm" in name or "post_attention_layernorm" in name or name == "model.norm.weight":
        role = "normalization"
    elif ".mlp.router" in name:
        role = "router"
    elif ".mlp.experts." in name:
        role = "expert"
    else:
        role = "UNKNOWN"
    return layer, role, expert


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    shard_headers = {
        shard: load_safetensors_header(Path(args.header_dir) / f"{shard.removesuffix('.safetensors')}.header.bin")
        for shard in SHARDS
    }
    shard_http = {
        shard: parse_headers(Path(args.header_dir) / f"{shard.removesuffix('.safetensors')}.headers.txt")
        for shard in SHARDS
    }

    tensors = []
    totals_by_role: dict[str, int] = {}
    totals_by_dtype: dict[str, int] = {}
    unknown_count = 0
    seen_names: set[str] = set()

    for shard in SHARDS:
        header = shard_headers[shard]
        for name in sorted(key for key in header if key != "__metadata__"):
            info = header[name]
            expected_shard = weight_map.get(name)
            if expected_shard != shard:
                raise ValueError(f"{name} header shard {shard} disagrees with index {expected_shard}")
            start, end = info["data_offsets"]
            byte_size = int(end) - int(start)
            layer, role, expert = classify(name)
            if role == "UNKNOWN":
                unknown_count += 1
            totals_by_role[role] = totals_by_role.get(role, 0) + byte_size
            totals_by_dtype[info["dtype"]] = totals_by_dtype.get(info["dtype"], 0) + byte_size
            seen_names.add(name)
            tensors.append(
                {
                    "name": name,
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                    "byte_size": byte_size,
                    "layer": layer,
                    "role": role,
                    "expert_id": expert,
                    "source_shard": shard,
                    "source_shard_checksum": shard_http[shard].get("x-linked-etag")
                    or shard_http[shard].get("etag")
                    or shard_http[shard].get("x-xet-hash"),
                    "source_shard_size": int(shard_http[shard].get("x-linked-size") or shard_http[shard].get("content-length") or 0),
                }
            )

    missing = sorted(set(weight_map) - seen_names)
    extra = sorted(seen_names - set(weight_map))
    if missing or extra:
        raise ValueError(f"index/header mismatch missing={len(missing)} extra={len(extra)}")

    return {
        "schema_version": 1,
        "model": "openai/gpt-oss-20b",
        "repo_commit": args.repo_commit,
        "checkpoint_format": "safetensors",
        "inventory_method": "safetensors_headers_via_http_range",
        "config": {
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_local_experts": config.get("num_local_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "quantization_config": config.get("quantization_config"),
        },
        "tensor_count": len(tensors),
        "unknown_tensor_count": unknown_count,
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "tensor_bytes_total": sum(item["byte_size"] for item in tensors),
        "totals_by_role": dict(sorted(totals_by_role.items())),
        "totals_by_dtype": dict(sorted(totals_by_dtype.items())),
        "shards": [
            {
                "file": shard,
                "size": int(shard_http[shard].get("x-linked-size") or shard_http[shard].get("content-length") or 0),
                "x_linked_etag": shard_http[shard].get("x-linked-etag"),
                "x_xet_hash": shard_http[shard].get("x-xet-hash") or shard_http[shard].get("etag"),
                "header_tensor_count": len([key for key in shard_headers[shard] if key != "__metadata__"]),
            }
            for shard in SHARDS
        ],
        "tensors": tensors,
    }


def write_checksums(args: argparse.Namespace, inventory: dict[str, Any]) -> None:
    lines = [
        f"model openai/gpt-oss-20b",
        f"repo_commit {args.repo_commit}",
        f"config.json sha256 {sha256_file(Path(args.config))}",
        f"model.safetensors.index.json sha256 {sha256_file(Path(args.index))}",
    ]
    for shard in inventory["shards"]:
        lines.append(
            f"{shard['file']} size {shard['size']} x_linked_etag {shard.get('x_linked_etag')} x_xet_hash {shard.get('x_xet_hash')}"
        )
    Path(args.checksums).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checksums", required=True)
    args = parser.parse_args()

    inventory = build_inventory(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = Path(args.checksums)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    write_checksums(args, inventory)


if __name__ == "__main__":
    main()
