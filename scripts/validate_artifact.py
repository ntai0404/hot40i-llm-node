#!/usr/bin/env python3
"""Validate a JSON artifact against one of the repository JSON Schemas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - environment setup failure
    raise SystemExit('jsonschema is required: pip install -e ".[dev]"') from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("schema", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    document = json.loads(args.artifact.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(document, schema)
    print("VALID")


if __name__ == "__main__":
    main()
