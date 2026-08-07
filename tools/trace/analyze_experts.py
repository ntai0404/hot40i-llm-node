#!/usr/bin/env python3
"""Summarize canonical runtime `route` events from a JSONL trace."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JSONL expert-routing traces")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    counts: collections.Counter[tuple[int, int]] = collections.Counter()
    transitions: collections.Counter[
        tuple[tuple[int, int], tuple[int, int]]
    ] = collections.Counter()
    previous: tuple[int, int] | None = None
    tokens = 0
    route_events = 0

    for line_number, line in enumerate(
        args.trace.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "route":
            continue
        if "layer" not in row or "expert" not in row:
            raise ValueError(f"route event missing layer/expert at line {line_number}")
        key = (int(row["layer"]), int(row["expert"]))
        counts[key] += 1
        route_events += 1
        if previous is not None:
            transitions[(previous, key)] += 1
        previous = key
        tokens = max(tokens, int(row.get("token", 0)) + 1)

    print(
        json.dumps(
            {
                "tokens_seen": tokens,
                "route_events": route_events,
                "top_experts": [
                    {"layer": key[0], "expert": key[1], "count": count}
                    for key, count in counts.most_common(32)
                ],
                "top_transitions": [
                    {"from": list(source), "to": list(target), "count": count}
                    for (source, target), count in transitions.most_common(32)
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
