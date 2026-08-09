#!/usr/bin/env python3
"""Summarize canonical runtime JSONL traces."""
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
    cache_hits = 0
    cache_lookups = 0
    flash_bytes_by_token: collections.Counter[int] = collections.Counter()
    events = 0
    last_ts = -1

    for line_number, line in enumerate(
        args.trace.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        events += 1
        ts_ns = int(row["ts_ns"])
        if ts_ns < last_ts:
            raise ValueError(f"trace timestamp moved backwards at line {line_number}")
        last_ts = ts_ns
        event = row.get("event")
        if "token" in row:
            tokens = max(tokens, int(row["token"]) + 1)
        if event in {"cache_hit", "cache_miss"}:
            cache_lookups += 1
            if event == "cache_hit" or row.get("cache_hit") is True:
                cache_hits += 1
        if event == "read_end":
            flash_bytes_by_token[int(row.get("token", 0))] += int(row.get("bytes", 0))
        if event != "route":
            continue
        if "layer" not in row or "expert" not in row:
            raise ValueError(f"route event missing layer/expert at line {line_number}")
        key = (int(row["layer"]), int(row["expert"]))
        counts[key] += 1
        route_events += 1
        if previous is not None:
            transitions[(previous, key)] += 1
        previous = key

    print(
        json.dumps(
            {
                "events": events,
                "tokens_seen": tokens,
                "route_events": route_events,
                "cache_lookups": cache_lookups,
                "cache_hits": cache_hits,
                "cache_misses": cache_lookups - cache_hits,
                "cache_hit_rate": (cache_hits / cache_lookups) if cache_lookups else None,
                "flash_bytes_total": sum(flash_bytes_by_token.values()),
                "flash_bytes_per_token": {
                    str(token): flash_bytes_by_token[token]
                    for token in sorted(flash_bytes_by_token)
                },
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
