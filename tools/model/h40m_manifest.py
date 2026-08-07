#!/usr/bin/env python3
"""Create and validate project-local H40M/1 manifests.

The full converter/repacker is implemented by roadmap task M01. This utility is
kept schema-driven so the scaffold cannot drift from the storage contract.
"""
from __future__ import annotations

import argparse
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

    args = parser.parse_args()
    if args.cmd == "init":
        init(args.path, args.model, args.checkpoint_sha256)
    else:
        document = json.loads(args.path.read_text(encoding="utf-8"))
        validate(document)
        print("OK")


if __name__ == "__main__":
    main()
