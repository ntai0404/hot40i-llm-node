#!/usr/bin/env python3
"""Resolve third-party Git refs to immutable full commit SHAs for R00."""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "third_party/LOCK.yaml"
FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def _load() -> dict[str, Any]:
    return yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))


def _save(document: dict[str, Any]) -> None:
    LOCK_PATH.write_text(
        yaml.safe_dump(document, sort_keys=False, width=110), encoding="utf-8"
    )


def _ls_remote(url: str, *patterns: str) -> list[tuple[str, str]]:
    process = subprocess.run(
        ["git", "ls-remote", url, *patterns],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or f"git ls-remote failed for {url}")
    rows: list[tuple[str, str]] = []
    for line in process.stdout.splitlines():
        if not line.strip():
            continue
        sha, ref = line.split(None, 1)
        rows.append((sha, ref))
    return rows


def resolve(url: str, requested_ref: str) -> tuple[str, str]:
    if FULL_SHA.fullmatch(requested_ref):
        rows = _ls_remote(url)
        if any(sha == requested_ref for sha, _ in rows):
            return requested_ref, "direct-sha"
        raise RuntimeError(f"requested SHA not advertised by {url}: {requested_ref}")

    patterns = [
        f"refs/tags/{requested_ref}^{{}}",
        f"refs/tags/{requested_ref}",
        f"refs/heads/{requested_ref}",
        requested_ref,
    ]
    rows = _ls_remote(url, *patterns)
    if not rows:
        raise RuntimeError(f"no Git ref matched {requested_ref!r} at {url}")

    # Prefer an annotated-tag peeled commit, then a branch, then the tag object.
    preference = [
        f"refs/tags/{requested_ref}^{{}}",
        f"refs/heads/{requested_ref}",
        f"refs/tags/{requested_ref}",
    ]
    by_ref = {ref: sha for sha, ref in rows}
    for ref in preference:
        if ref in by_ref:
            return by_ref[ref], ref
    sha, ref = rows[0]
    return sha, ref


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", help="resolve only named upstream; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    document = _load()
    selected = set(args.name or [])
    failures: list[str] = []

    for item in document["upstreams"]:
        if selected and item["name"] not in selected:
            continue
        try:
            sha, matched_ref = resolve(item["url"], item["requested_ref"])
        except Exception as exc:
            item["verification_status"] = "RESOLUTION_FAILED"
            item["resolution_error"] = str(exc)
            failures.append(f"{item['name']}: {exc}")
            print(f"FAIL {item['name']}: {exc}")
            continue
        item["resolved_commit"] = sha
        item["resolution_required"] = False
        item["verification_status"] = "RESOLVED"
        item["matched_ref"] = matched_ref
        item["resolved_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        item.pop("resolution_error", None)
        print(f"{item['name']}: {item['requested_ref']} -> {sha} ({matched_ref})")

    if not args.dry_run:
        _save(document)
    if failures:
        raise SystemExit("one or more upstreams could not be resolved")


if __name__ == "__main__":
    main()
