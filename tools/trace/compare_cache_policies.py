#!/usr/bin/env python3
"""Replay expert routes through bounded cache-policy simulations."""
from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ExpertKey = tuple[int, int]


@dataclass
class PolicyResult:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class PolicySimulation:
    def __init__(self, name: str, slots: int, decay_interval: int = 4096) -> None:
        if slots <= 0:
            raise ValueError("slots must be positive")
        self.name = name
        self.slots = slots
        self.decay_interval = decay_interval
        self.resident: set[ExpertKey] = set()
        self.last_access: dict[ExpertKey, int] = {}
        self.frequency: collections.Counter[ExpertKey] = collections.Counter()
        self.layer_frequency: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
        self.layer_requests: collections.Counter[int] = collections.Counter()
        self.result = PolicyResult()

    def _decay(self) -> None:
        for key in list(self.frequency):
            self.frequency[key] = max(1, self.frequency[key] // 2)
        for counts in self.layer_frequency.values():
            for expert in list(counts):
                counts[expert] = max(1, counts[expert] // 2)
        for layer in list(self.layer_requests):
            self.layer_requests[layer] = max(1, self.layer_requests[layer] // 2)

    def _victim(self) -> ExpertKey:
        if self.name == "lru":
            return min(self.resident, key=lambda key: self.last_access[key])
        if self.name == "lfu_decay":
            return min(
                self.resident,
                key=lambda key: (self.frequency[key], self.last_access[key]),
            )
        if self.name == "per_layer_hotset":
            return min(
                self.resident,
                key=lambda key: (
                    self.layer_frequency[key[0]][key[1]] / self.layer_requests[key[0]],
                    self.last_access[key],
                ),
            )
        raise ValueError(f"unknown policy: {self.name}")

    def access(self, key: ExpertKey) -> None:
        self.result.requests += 1
        now = self.result.requests
        self.frequency[key] += 1
        self.layer_frequency[key[0]][key[1]] += 1
        self.layer_requests[key[0]] += 1
        if self.name != "lru" and now % self.decay_interval == 0:
            self._decay()
        if key in self.resident:
            self.result.hits += 1
            self.last_access[key] = now
            return
        self.result.misses += 1
        if len(self.resident) == self.slots:
            victim = self._victim()
            self.resident.remove(victim)
            self.result.evictions += 1
        self.resident.add(key)
        self.last_access[key] = now


def route_keys(path: Path) -> Iterable[ExpertKey]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "route":
                continue
            try:
                yield int(row["layer"]), int(row["expert"])
            except KeyError as exc:
                raise ValueError(f"route missing {exc.args[0]} at line {line_number}") from exc


def compare(trace: Path, slots: int, expert_bytes: int, decay_interval: int) -> dict[str, object]:
    policies = [
        PolicySimulation("lru", slots, decay_interval),
        PolicySimulation("lfu_decay", slots, decay_interval),
        PolicySimulation("per_layer_hotset", slots, decay_interval),
    ]
    for key in route_keys(trace):
        for policy in policies:
            policy.access(key)

    rows: list[dict[str, object]] = []
    for policy in policies:
        result = policy.result
        rows.append(
            {
                "policy": policy.name,
                "requests": result.requests,
                "hits": result.hits,
                "misses": result.misses,
                "evictions": result.evictions,
                "hit_rate": result.hits / result.requests if result.requests else 0.0,
                "flash_bytes": result.misses * expert_bytes,
                "flash_bytes_per_request": result.misses * expert_bytes / result.requests
                if result.requests
                else 0.0,
                "peak_slots": len(policy.resident),
            }
        )
    baseline = rows[0]
    for row in rows:
        row["miss_reduction_vs_lru"] = int(baseline["misses"]) - int(row["misses"])
        row["flash_reduction_vs_lru"] = int(baseline["flash_bytes"]) - int(row["flash_bytes"])
    return {
        "schema_version": 1,
        "trace": trace.as_posix(),
        "slots": slots,
        "budget_bytes": slots * expert_bytes,
        "expert_bytes": expert_bytes,
        "decay_interval": decay_interval,
        "policies": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--expert-bytes", type=int, default=13_236_480)
    parser.add_argument("--decay-interval", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = compare(args.trace, args.slots, args.expert_bytes, args.decay_interval)
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
