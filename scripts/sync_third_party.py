#!/usr/bin/env python3
"""Fetch selected upstreams and, when locked, checkout the immutable commit."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "third_party/manifest.yaml"
LOCK = ROOT / "third_party/LOCK.yaml"


def run(argv: list[str], cwd: Path | None = None) -> None:
    subprocess.run(argv, cwd=cwd or ROOT, check=True)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", help="sync only named upstream; repeatable")
    parser.add_argument("--dest", type=Path, default=ROOT / "third_party/src")
    parser.add_argument(
        "--require-locked",
        action="store_true",
        help="fail if a selected upstream has no resolved immutable commit",
    )
    args = parser.parse_args()

    manifest = load_yaml(MANIFEST)
    lock = load_yaml(LOCK)
    lock_by_name = {item["name"]: item for item in lock["upstreams"]}
    selected = set(args.name or [])
    args.dest.mkdir(parents=True, exist_ok=True)

    for item in manifest["upstreams"]:
        name = item["name"]
        if selected and name not in selected:
            continue
        locked = lock_by_name[name]
        commit = locked.get("resolved_commit")
        if args.require_locked and not commit:
            raise SystemExit(f"{name} is not locked; run scripts/resolve_upstreams.py first")

        target = args.dest / name
        if not target.exists():
            run(["git", "clone", "--filter=blob:none", item["url"], str(target)])
        else:
            run(["git", "fetch", "--all", "--tags", "--prune"], cwd=target)

        if commit:
            run(["git", "checkout", "--detach", commit], cwd=target)
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=target, text=True
            ).strip()
            if head != commit:
                raise SystemExit(f"{name} checkout mismatch: {head} != {commit}")
            print(f"{name}: {target} @ {commit}")
        else:
            print(f"{name}: {target} fetched but UNLOCKED; do not benchmark against it")


if __name__ == "__main__":
    main()
