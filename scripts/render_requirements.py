#!/usr/bin/env python3
"""Render the human requirements traceability table from canonical YAML."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "roadmap/requirements.yaml"
OUTPUT = ROOT / "docs/18_REQUIREMENTS_TRACEABILITY.md"


def render() -> str:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    lines = [
        "# Requirements traceability",
        "",
        "> Generated from `roadmap/requirements.yaml` by `scripts/render_requirements.py`. Edit the YAML, not this table.",
        "",
        "| ID | Type | Requirement | Verified by | Acceptance gate |",
        "|---|---|---|---|---|",
    ]
    for item in data["requirements"]:
        statement = str(item["statement"]).replace("|", "\\|")
        tasks = ", ".join(f"`{task}`" for task in item["verified_by_tasks"])
        gate = f"`{item['acceptance_gate']}`" if item.get("acceptance_gate") else "—"
        lines.append(
            f"| `{item['id']}` | {item['type']} | {statement} | {tasks} | {gate} |"
        )
    lines.extend(
        [
            "",
            "The task DAG is canonical for execution. This table exists so a human reviewer can trace every system requirement to concrete implementation/verification work.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            raise SystemExit("requirements traceability document is stale; run scripts/render_requirements.py")
        print("REQUIREMENTS_TRACEABILITY_OK")
        return
    OUTPUT.write_text(expected, encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
