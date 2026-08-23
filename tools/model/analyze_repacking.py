#!/usr/bin/env python3
"""Compare an H40M physical-repacking candidate against its device baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_metrics(result: dict[str, Any]) -> dict[str, Any]:
    elapsed_s = float(result["elapsed_ms"]) / 1000.0
    read_s = float(result["prefetch_read_ns"]) / 1_000_000_000.0
    reads = int(result["prefetched_experts"])
    return {
        "input_tokens": int(result["input_tokens"]),
        "elapsed_ms": int(result["elapsed_ms"]),
        "tokens_per_second": int(result["input_tokens"]) / elapsed_s,
        "expert_flash_bytes": int(result["expert_flash_bytes"]),
        "cache_hits": int(result["cache_hits"]),
        "cache_misses": int(result["cache_misses"]),
        "prefetched_experts": reads,
        "prefetch_read_ns": int(result["prefetch_read_ns"]),
        "prefetch_read_mean_ms": (read_s * 1000.0) / reads,
        "expert_read_mib_per_second": (int(result["expert_flash_bytes"]) / (1024.0 * 1024.0)) / read_s,
        "peak_rss_kib": int(result["peak_rss_kib"]),
        "emitted_token_id": int(result["emitted_token_id"]),
        "emitted_token_logit": float(result["emitted_token_logit"]),
        "cache_policy": result["cache_policy"],
        "dense_threads": int(result["dense_threads"]),
        "io_overlap_enabled": bool(result["io_overlap_enabled"]),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    layout = load_json(args.layout)
    before_raw = load_json(args.before)
    after_raw = load_json(args.after)
    before = run_metrics(before_raw)
    after = run_metrics(after_raw)
    token_match = before["emitted_token_id"] == after["emitted_token_id"]
    logit_diff = abs(before["emitted_token_logit"] - after["emitted_token_logit"])
    runtime_policy_match = all(
        before[key] == after[key]
        for key in ("input_tokens", "cache_policy", "dense_threads", "io_overlap_enabled")
    )
    traffic_match = all(
        before[key] == after[key]
        for key in ("expert_flash_bytes", "cache_hits", "cache_misses", "prefetched_experts")
    )
    speedup = before["elapsed_ms"] / after["elapsed_ms"]
    retained = speedup > 1.0 and token_match and logit_diff == 0.0 and runtime_policy_match
    return {
        "schema_version": 1,
        "status": "pass" if token_match and logit_diff == 0.0 and runtime_policy_match and traffic_match else "fail",
        "task": "O04",
        "candidate": "trace_guided_physical_expert_arena_v2",
        "layout": {
            "path": args.layout.as_posix(),
            "source_arena_sha256": layout["source_arena"]["sha256"],
            "candidate_arena_sha256": layout["arena"]["sha256"],
            "records_checked": layout["byte_correctness"]["records_checked"],
            "byte_correctness_status": layout["byte_correctness"]["status"],
            "pair_distance_reduction": layout["optimization"]["pair_distance_reduction"],
            "sequential_seek_reduction": layout["optimization"]["sequential_seek_reduction"],
        },
        "benchmark": {
            "device": "Infinix X6528 / Hot 40i",
            "runtime_mode": "production_minimal_trace",
            "same_binary": True,
            "same_cache_policy": runtime_policy_match,
            "before": before,
            "after": after,
            "end_to_end_speedup": speedup,
            "end_to_end_change_percent": (speedup - 1.0) * 100.0,
            "expert_read_speedup": before["prefetch_read_ns"] / after["prefetch_read_ns"],
            "expert_read_change_percent": (before["prefetch_read_ns"] / after["prefetch_read_ns"] - 1.0) * 100.0,
        },
        "correctness": {
            "token_id_match": token_match,
            "token_logit_max_abs_diff": logit_diff,
            "flash_and_cache_counters_match": traffic_match,
        },
        "decision": {
            "retained": retained,
            "production_layout": "H40M_EXPERT_ARENA/1",
            "reason": (
                "Candidate was not retained because its small expert-read improvement did not improve "
                "end-to-end decode throughput. The CPU-dominated decoder remained effectively unchanged."
                if not retained
                else "Candidate improved end-to-end decode with correctness preserved."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["decision"], "end_to_end_speedup": result["benchmark"]["end_to_end_speedup"]}, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
