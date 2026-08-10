"""Summarize bounded decoder JSONL traces without discarding raw events."""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path: Path) -> dict:
    events = 0
    counts = Counter()
    bytes_by_event = Counter()
    duration_by_event = defaultdict(list)
    cache_by_token = defaultdict(Counter)
    layers = set()
    tokens = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            events += 1
            event = row.get("event", "unknown")
            counts[event] += 1
            if "bytes" in row:
                bytes_by_event[event] += int(row["bytes"])
            if "duration_ns" in row:
                duration_by_event[event].append(int(row["duration_ns"]))
            if "layer" in row:
                layers.add(int(row["layer"]))
            if "token" in row:
                tokens.add(int(row["token"]))
            if event in {"cache_hit", "cache_miss"}:
                cache_by_token[int(row.get("token", -1))][event] += 1
    duration_summary = {
        event: {
            "count": len(values),
            "mean_ns": statistics.mean(values),
            "p95_ns": sorted(values)[max(0, int(len(values) * 0.95) - 1)],
        }
        for event, values in duration_by_event.items()
    }
    return {
        "schema_version": 1,
        "trace": str(path),
        "events": events,
        "tokens_seen": len(tokens),
        "layers_seen": len(layers),
        "event_counts": dict(sorted(counts.items())),
        "bytes_by_event": dict(sorted(bytes_by_event.items())),
        "duration_ns": duration_summary,
        "cache_by_token": {str(k): dict(v) for k, v in sorted(cache_by_token.items())},
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: analyze_long_decode.py TRACE.jsonl OUTPUT.json")
    output = analyze(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("events", "tokens_seen", "layers_seen")}))
