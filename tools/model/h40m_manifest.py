#!/usr/bin/env python3
"""Create and validate project-local H40M/1 manifests.

The full converter/repacker is implemented by roadmap task M01. This utility is
kept schema-driven so the scaffold cannot drift from the storage contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas/h40m_manifest.schema.json"


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(document: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError('jsonschema is required: pip install -e ".[dev]"') from exc
    jsonschema.Draft202012Validator.check_schema(_schema())
    jsonschema.validate(document, _schema())


def init(path: Path, model: str, checkpoint_sha256: str) -> None:
    document = {
        "format": "H40M/1",
        "source": {
            "model": model,
            "checkpoint_sha256": checkpoint_sha256,
            "upstream_ref": None,
            "converter_commit": None,
        },
        "files": [],
        "tensors": [],
        "extensions": {},
    }
    validate(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _synthetic_sha256(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _placement(role: str) -> str:
    if role == "embedding":
        return "token_lookup"
    if role == "expert":
        return "cache"
    if role == "lm_head":
        return "stream"
    return "resident"


def _quant_type(tensor: dict[str, Any]) -> str | None:
    if tensor["role"] == "expert" and tensor["dtype"] == "U8":
        if tensor["name"].endswith("_blocks"):
            return "MXFP4_E2M1_PACKED"
        if tensor["name"].endswith("_scales"):
            return "MXFP4_E8M0_SCALE"
    return None


def convert(inventory_path: Path, output_path: Path, alignment: int = 4096) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")

    files = [
        {
            "id": 0,
            "path": "h40m/model.h40m",
            "size": 0,
            "sha256": "0" * 64,
            "checksum_kind": "layout_manifest_sha256",
        }
    ]
    tensors = []
    offset = 0
    for source in sorted(inventory["tensors"], key=lambda item: item["name"]):
        offset = _align(offset, alignment)
        length = int(source["byte_size"])
        tensors.append(
            {
                "name": source["name"],
                "role": source["role"],
                "layer": source["layer"],
                "expert_id": source["expert_id"],
                "shape": source["shape"],
                "dtype": source["dtype"],
                "quant_type": _quant_type(source),
                "layout": "h40m-contiguous-v1",
                "alignment": alignment,
                "file_id": 0,
                "offset": offset,
                "length": length,
                "placement": _placement(source["role"]),
                "sha256": _synthetic_sha256(
                    source["name"],
                    source["source_shard"],
                    source["source_shard_checksum"],
                    source["byte_size"],
                    offset,
                ),
                "checksum_kind": "source_range_id_sha256",
                "source_tensor": source["name"],
                "source": {
                    "model": inventory["model"],
                    "repo_commit": inventory["repo_commit"],
                    "source_shard": source["source_shard"],
                    "source_shard_checksum": source["source_shard_checksum"],
                    "source_shard_size": source["source_shard_size"],
                },
            }
        )
        offset += length

    files[0]["size"] = _align(offset, alignment)
    manifest = {
        "format": "H40M/1",
        "source": {
            "model": inventory["model"],
            "checkpoint_sha256": _synthetic_sha256(inventory["model"], inventory["repo_commit"], inventory["tensor_bytes_total"]),
            "upstream_ref": inventory["repo_commit"],
            "converter_commit": None,
            "inventory_method": inventory.get("inventory_method"),
            "checkpoint_sha256_kind": "inventory_identity_sha256",
        },
        "files": files,
        "tensors": tensors,
        "extensions": {
            "alignment": alignment,
            "tensor_count": len(tensors),
            "source_inventory": str(inventory_path.as_posix()),
            "limitations": [
                "Tensor sha256 fields are deterministic source-range identifiers until M02 materializes payload bytes."
            ],
        },
    }
    files[0]["sha256"] = _synthetic_sha256(
        manifest["source"]["checkpoint_sha256"],
        files[0]["size"],
        len(tensors),
        manifest["extensions"]["alignment"],
    )
    validate(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="H40M/1 manifest helper")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    command = subparsers.add_parser("init")
    command.add_argument("path", type=Path)
    command.add_argument("--model", default="openai/gpt-oss-20b")
    command.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="64-hex checksum of the pinned source checkpoint/artifact",
    )

    command = subparsers.add_parser("validate")
    command.add_argument("path", type=Path)

    command = subparsers.add_parser("convert")
    command.add_argument("--inventory", required=True, type=Path)
    command.add_argument("--out", required=True, type=Path)
    command.add_argument("--alignment", default=4096, type=int)

    args = parser.parse_args()
    if args.cmd == "init":
        init(args.path, args.model, args.checkpoint_sha256)
    elif args.cmd == "validate":
        document = json.loads(args.path.read_text(encoding="utf-8"))
        validate(document)
        print("OK")
    else:
        convert(args.inventory, args.out, args.alignment)


if __name__ == "__main__":
    main()
