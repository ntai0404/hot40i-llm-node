#!/usr/bin/env python3
"""Evaluate bounded expert-prefetch predictors on canonical route traces."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


RouteGroups = dict[tuple[int, int], tuple[int, ...]]


def load_routes(trace: Path) -> tuple[RouteGroups, int, int]:
    groups: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    max_token = -1
    max_layer = -1
    with trace.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "route":
                continue
            token = int(row["token"])
            layer = int(row["layer"])
            groups[(token, layer)].append(int(row["expert"]))
            max_token = max(max_token, token)
            max_layer = max(max_layer, layer)
    normalized = {key: tuple(dict.fromkeys(experts)) for key, experts in groups.items()}
    return normalized, max_token + 1, max_layer + 1


def top_experts(counts: collections.Counter[int], width: int) -> tuple[int, ...]:
    return tuple(expert for expert, unused in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:width])


def metric_row(
    name: str,
    intersections: int,
    predictions: int,
    actuals: int,
    first_choice_hits: int,
    opportunities: int,
    expert_bytes: int,
) -> dict[str, object]:
    useful = intersections * expert_bytes
    wasted = (predictions - intersections) * expert_bytes
    return {
        "predictor": name,
        "precision": intersections / predictions if predictions else 0.0,
        "recall": intersections / actuals if actuals else 0.0,
        "first_choice_accuracy": first_choice_hits / opportunities if opportunities else 0.0,
        "first_choice_hits": first_choice_hits,
        "opportunities": opportunities,
        "useful_predictions": intersections,
        "wasted_predictions": predictions - intersections,
        "useful_prefetch_bytes": useful,
        "wasted_prefetch_bytes": wasted,
        "useful_to_wasted_ratio": useful / wasted if wasted else None,
    }


def evaluate(trace: Path, train_tokens: int, width: int, expert_bytes: int) -> dict[str, object]:
    routes, token_count, layer_count = load_routes(trace)
    if train_tokens <= 0 or train_tokens >= token_count:
        raise ValueError("train_tokens must leave a non-empty evaluation split")

    layer_counts = [collections.Counter() for _ in range(layer_count)]
    transitions = [collections.defaultdict(collections.Counter) for _ in range(layer_count - 1)]
    for token in range(train_tokens):
        for layer in range(layer_count):
            layer_counts[layer].update(routes[(token, layer)])
        for layer in range(layer_count - 1):
            target = routes[(token, layer + 1)]
            for expert in routes[(token, layer)]:
                transitions[layer][expert].update(target)

    totals = {
        "previous_token": [0, 0, 0, 0, 0],
        "layer_frequency": [0, 0, 0, 0, 0],
        "cross_layer_transition": [0, 0, 0, 0, 0],
    }
    for token in range(train_tokens, token_count):
        for layer in range(layer_count - 1):
            actual_ordered = routes[(token, layer + 1)]
            actual = set(actual_ordered)
            previous = routes[(token - 1, layer + 1)][:width]
            frequent = top_experts(layer_counts[layer + 1], width)
            scores: collections.Counter[int] = collections.Counter()
            for expert in routes[(token, layer)]:
                scores.update(transitions[layer][expert])
            transition = top_experts(scores, width)
            if len(transition) < width:
                transition = tuple(dict.fromkeys(transition + frequent))[:width]
            for name, predicted in (
                ("previous_token", previous),
                ("layer_frequency", frequent),
                ("cross_layer_transition", transition),
            ):
                totals[name][0] += len(actual.intersection(predicted))
                totals[name][1] += len(predicted)
                totals[name][2] += len(actual)
                totals[name][3] += int(bool(predicted) and predicted[0] == actual_ordered[0])
                totals[name][4] += 1

    rows = [
        metric_row(name, values[0], values[1], values[2], values[3], values[4], expert_bytes)
        for name, values in totals.items()
    ]
    best = max(rows, key=lambda row: (row["precision"], row["recall"]))
    return {
        "schema_version": 1,
        "trace": trace.as_posix(),
        "train_tokens": train_tokens,
        "evaluation_tokens": token_count - train_tokens,
        "layers": layer_count,
        "prediction_width": width,
        "transition_counter_entries_max": (layer_count - 1) * 32 * 32,
        "transition_counter_bytes_max": (layer_count - 1) * 32 * 32 * 4,
        "predictors": rows,
        "best_predictor": best["predictor"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--train-tokens", type=int, default=512)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--expert-bytes", type=int, default=13_236_480)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = evaluate(args.trace, args.train_tokens, args.width, args.expert_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
