#!/usr/bin/env python3
"""Estimate the upper bound of expert-output window reuse from route traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(trace: Path, windows: list[int], expert_bytes: int) -> dict[str, object]:
    last_seen = {window: {} for window in windows}
    hits = {window: 0 for window in windows}
    requests = 0
    tokens: set[int] = set()
    with trace.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "route":
                continue
            token = int(row["token"])
            key = (int(row["layer"]), int(row["expert"]))
            requests += 1
            tokens.add(token)
            for window in windows:
                previous = last_seen[window].get(key)
                if previous is not None and 0 < token - previous <= window:
                    hits[window] += 1
                last_seen[window][key] = token
    return {
        "schema_version": 1,
        "trace": trace.as_posix(),
        "requests": requests,
        "tokens_seen": len(tokens),
        "windows": [
            {
                "window_tokens": window,
                "potential_reuses": hits[window],
                "potential_reuse_rate": hits[window] / requests if requests else 0.0,
                "upper_bound_flash_bytes_saved": hits[window] * expert_bytes,
            }
            for window in windows
        ],
        "quality_caveat": (
            "A route recurrence proves only that the same expert was selected; exact output reuse "
            "also requires an identical normalized hidden input. Savings are therefore an upper bound."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--windows", default="1,2,4")
    parser.add_argument("--expert-bytes", type=int, default=13_236_480)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    windows = [int(value) for value in args.windows.split(",")]
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("windows must be positive")
    document = analyze(args.trace, windows, args.expert_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
