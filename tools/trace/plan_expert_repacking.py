#!/usr/bin/env python3
"""Plan a deterministic expert-arena ordering from canonical route traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def route_sequences(path: Path) -> tuple[dict[int, list[int]], dict[tuple[int, int], list[int]], int]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    by_token_layer: dict[tuple[int, int], list[int]] = defaultdict(list)
    events = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "route":
                continue
            try:
                layer = int(row["layer"])
                token = int(row["token"])
                expert = int(row["expert"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid route event at line {line_number}") from exc
            by_layer[layer].append(expert)
            by_token_layer[(token, layer)].append(expert)
            events += 1
    if not events:
        raise ValueError(f"no route events in {path}")
    return dict(by_layer), dict(by_token_layer), events


def weight_matrix(
    sequence: list[int], groups: list[list[int]], expert_count: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    transitions = [[0] * expert_count for _ in range(expert_count)]
    coaccess = [[0] * expert_count for _ in range(expert_count)]
    for left, right in zip(sequence, sequence[1:]):
        if left != right:
            transitions[left][right] += 1
    for group in groups:
        unique = list(dict.fromkeys(group))
        for index, left in enumerate(unique):
            for right in unique[index + 1 :]:
                coaccess[left][right] += 1
                coaccess[right][left] += 1
    combined = [[0] * expert_count for _ in range(expert_count)]
    for left in range(expert_count):
        for right in range(expert_count):
            combined[left][right] = (
                transitions[left][right]
                + transitions[right][left]
                + coaccess[left][right]
            )
    return transitions, coaccess, combined


def objective(order: list[int], weights: list[list[int]]) -> int:
    position = {expert: index for index, expert in enumerate(order)}
    total = 0
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            total += weights[left][right] * abs(position[left] - position[right])
    return total


def improve_order(expert_count: int, weights: list[list[int]]) -> tuple[list[int], int, int, int]:
    order = list(range(expert_count))
    before = objective(order, weights)
    current = before
    swaps = 0
    while True:
        best_score = current
        best_pair: tuple[int, int] | None = None
        for left in range(expert_count - 1):
            for right in range(left + 1, expert_count):
                candidate = order.copy()
                candidate[left], candidate[right] = candidate[right], candidate[left]
                score = objective(candidate, weights)
                if score < best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        order[left], order[right] = order[right], order[left]
        current = best_score
        swaps += 1
    return order, before, current, swaps


def seek_distance(sequence: list[int], positions: dict[int, int]) -> int:
    return sum(abs(positions[left] - positions[right]) for left, right in zip(sequence, sequence[1:]))


def matrix_summary(matrix: list[list[int]]) -> dict[str, Any]:
    edges = [
        (count, source, target)
        for source, row in enumerate(matrix)
        for target, count in enumerate(row)
        if count
    ]
    payload = json.dumps(matrix, separators=(",", ":")).encode("ascii")
    return {
        "rows": len(matrix),
        "columns": len(matrix[0]),
        "nonzero_entries": len(edges),
        "total_weight": sum(count for count, _, _ in edges),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "top_edges": [
            {"source": source, "target": target, "count": count}
            for count, source, target in sorted(edges, reverse=True)[:32]
        ],
    }


def plan(args: argparse.Namespace) -> dict[str, Any]:
    source = load_json(args.layout)
    previous = load_json(args.output) if args.output.exists() else None
    sequences, grouped, route_events = route_sequences(args.trace)
    source_records = {
        (int(record["layer"]), int(record["expert_id"])): record
        for record in source["records"]
    }
    layer_count = max(layer for layer, _ in source_records) + 1
    expert_count = max(expert for _, expert in source_records) + 1
    expected = layer_count * expert_count
    if len(source_records) != expected:
        raise ValueError(f"layout has {len(source_records)} records, expected {expected}")
    if set(sequences) != set(range(layer_count)):
        raise ValueError("trace does not cover every model layer")

    source_slots = {
        layer: sorted(source_records[(layer, expert)]["offset"] for expert in range(expert_count))
        for layer in range(layer_count)
    }
    layer_plans: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0
    seek_before = 0
    seek_after = 0

    for layer in range(layer_count):
        groups = [experts for (token, item_layer), experts in grouped.items() if item_layer == layer]
        transitions, coaccess, weights = weight_matrix(sequences[layer], groups, expert_count)
        order, before, after, swaps = improve_order(expert_count, weights)
        old_positions = {expert: expert for expert in range(expert_count)}
        new_positions = {expert: position for position, expert in enumerate(order)}
        layer_seek_before = seek_distance(sequences[layer], old_positions)
        layer_seek_after = seek_distance(sequences[layer], new_positions)
        total_before += before
        total_after += after
        seek_before += layer_seek_before
        seek_after += layer_seek_after
        layer_plans.append(
            {
                "layer": layer,
                "physical_expert_order": order,
                "pair_distance_objective_before": before,
                "pair_distance_objective_after": after,
                "sequential_seek_records_before": layer_seek_before,
                "sequential_seek_records_after": layer_seek_after,
                "accepted_swaps": swaps,
                "transition_matrix": matrix_summary(transitions),
                "coaccess_matrix": matrix_summary(coaccess),
            }
        )
        for physical_position, expert in enumerate(order):
            source_record = source_records[(layer, expert)]
            record = {
                "layer": layer,
                "expert_id": expert,
                "file": args.arena_name,
                "offset": source_slots[layer][physical_position],
                "length": source_record["length"],
                "sha256": source_record["sha256"],
                "source_offset": source_record["offset"],
                "physical_ordinal": layer * expert_count + physical_position,
            }
            target_records.append(record)

    target_records.sort(key=lambda record: int(record["offset"]))
    result = {
        "schema_version": 2,
        "format": "H40M_EXPERT_ARENA/2",
        "source_model": source["source_model"],
        "source_ref": source["source_ref"],
        "source_shards": source["source_shards"],
        "source_arena": {
            "path": source["arena"]["path"],
            "size": source["arena"]["size"],
            "sha256": source["arena"]["sha256"],
            "layout": args.layout.as_posix(),
        },
        "arena": {
            **source["arena"],
            "path": args.arena_name,
            "sha256": None,
        },
        "trace_provenance": {
            "path": args.trace.as_posix(),
            "route_events": route_events,
            "tokens": len({token for token, _ in grouped}),
            "layer_count": layer_count,
            "expert_count": expert_count,
        },
        "optimization": {
            "algorithm": "deterministic_best_pair_swap_on_transition_plus_coaccess_distance",
            "pair_distance_objective_before": total_before,
            "pair_distance_objective_after": total_after,
            "pair_distance_reduction": 1.0 - (total_after / total_before),
            "sequential_seek_records_before": seek_before,
            "sequential_seek_records_after": seek_after,
            "sequential_seek_reduction": 1.0 - (seek_after / seek_before),
            "layers": layer_plans,
        },
        "records": target_records,
        "byte_correctness": {"status": "pending_repack"},
    }
    if previous is not None:
        previous_mapping = [
            (record["layer"], record["expert_id"], record["offset"], record["sha256"])
            for record in previous.get("records", [])
        ]
        current_mapping = [
            (record["layer"], record["expert_id"], record["offset"], record["sha256"])
            for record in result["records"]
        ]
        if previous_mapping == current_mapping and previous.get("byte_correctness", {}).get("status") == "pass":
            result["arena"]["path"] = previous["arena"]["path"]
            result["arena"]["sha256"] = previous["arena"]["sha256"]
            result["byte_correctness"] = previous["byte_correctness"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena-name", default="expert_arena_v2.bin")
    args = parser.parse_args()
    result = plan(args)
    optimization = result["optimization"]
    print(
        json.dumps(
            {
                "status": "pass",
                "route_events": result["trace_provenance"]["route_events"],
                "pair_distance_reduction": optimization["pair_distance_reduction"],
                "sequential_seek_reduction": optimization["sequential_seek_reduction"],
                "layers_planned": len(optimization["layers"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
