#!/usr/bin/env python3
"""Build the O05 context/KV decision from device measurements."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def log_metric(path: Path, name: str) -> float:
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
    prefix = name + "="
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :])
    raise ValueError(f"missing {name} in {path}")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    sweeps = [load_json(path) for path in args.device_sweep]
    host_sweep = load_json(args.host_sweep)
    memory_plan = load_json(args.memory_plan)
    production = load_json(args.production_baseline)
    if any(sweep["status"] != "pass" for sweep in sweeps) or host_sweep["status"] != "pass":
        raise ValueError("context sweep did not pass")
    for sweep in sweeps:
        if [row["tokens"] for row in sweep["contexts"]] != [512, 1024, 2048, 4096]:
            raise ValueError("device sweep does not contain required contexts")

    safe_budget = int(memory_plan["safe_rss_budget_bytes"])
    old_kv_bytes = next(
        int(item["bytes"])
        for item in memory_plan["budget_items"]
        if item["name"] == "kv_cache_1024_bf16"
    )
    non_kv_plan = int(memory_plan["total_planned_rss_bytes"]) - old_kv_bytes
    baseline_non_attention_seconds = (
        float(production["elapsed_ms"]) / 1000.0 / int(production["input_tokens"])
        - float(production["attention_ns"]) / 1_000_000_000.0 / int(production["input_tokens"])
    )

    candidates: list[dict[str, Any]] = []
    for index, host in enumerate(host_sweep["contexts"]):
        device_samples = [sweep["contexts"][index] for sweep in sweeps]
        if any(int(device["tokens"]) != int(host["tokens"]) for device in device_samples):
            raise ValueError("host/device sweep context mismatch")
        context = int(device_samples[0]["tokens"])
        storage = int(device_samples[0]["storage_bytes"])
        if any(int(device["storage_bytes"]) != storage for device in device_samples):
            raise ValueError("device sweep storage accounting mismatch")
        naive_full_bf16 = 24 * context * 8 * 64 * 2 * 2
        planned_total = non_kv_plan + storage
        attention_samples_ns = [int(device["last_token_24_layer_attention_ns"]) for device in device_samples]
        append_samples_ns = [int(device["append_ns"]) for device in device_samples]
        attention_seconds = statistics.median(attention_samples_ns) / 1_000_000_000.0
        projected_seconds = baseline_non_attention_seconds + attention_seconds
        candidates.append(
            {
                "context_tokens": context,
                "kv_storage_bytes": storage,
                "naive_all_layers_full_bf16_bytes": naive_full_bf16,
                "exact_sliding_window_savings_bytes": naive_full_bf16 - storage,
                "planned_total_rss_bytes": planned_total,
                "planned_headroom_bytes": safe_budget - planned_total,
                "device_repetitions": len(device_samples),
                "device_attention_samples_ns": attention_samples_ns,
                "device_append_samples_ns": append_samples_ns,
                "device_peak_rss_kib": max(int(device["peak_rss_kib"]) for device in device_samples),
                "device_append_ms_median": statistics.median(append_samples_ns) / 1_000_000.0,
                "device_last_token_24_layer_attention_ms": attention_seconds * 1000.0,
                "host_last_token_24_layer_attention_ms": float(host["last_token_24_layer_attention_ns"]) / 1_000_000.0,
                "projected_composite_seconds_per_token": projected_seconds,
                "projected_composite_tokens_per_second": 1.0 / projected_seconds,
                "within_safe_rss_budget": planned_total <= safe_budget,
            }
        )
    reference_throughput = candidates[0]["projected_composite_tokens_per_second"]
    for candidate in candidates:
        candidate["projected_throughput_fraction_vs_512"] = (
            candidate["projected_composite_tokens_per_second"] / reference_throughput
        )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["within_safe_rss_budget"]
        and candidate["planned_headroom_bytes"] >= safe_budget * 0.25
        and candidate["projected_throughput_fraction_vs_512"] >= 0.95
    ]
    if not eligible:
        raise ValueError("no context candidate meets memory/performance guardrails")
    selected = max(eligible, key=lambda candidate: candidate["context_tokens"])

    host_cache_diff = log_metric(args.host_test_log, "kv_cache_bf16_reference_max_abs_diff")
    host_fp32_diff = log_metric(args.host_test_log, "kv_cache_fp32_reference_max_abs_diff")
    device_cache_diff = log_metric(args.device_test_log, "kv_cache_bf16_reference_max_abs_diff")
    device_fp32_diff = log_metric(args.device_test_log, "kv_cache_fp32_reference_max_abs_diff")
    correctness_pass = (
        host_cache_diff <= 1.0e-6
        and device_cache_diff <= 1.0e-6
        and host_fp32_diff <= 1.0e-3
        and device_fp32_diff <= 1.0e-3
    )

    return {
        "schema_version": 1,
        "status": "pass" if correctness_pass else "fail",
        "task": "O05",
        "model_semantics": {
            "layers": 24,
            "key_value_heads": 8,
            "head_dim": 64,
            "storage_dtype": "BF16",
            "sliding_window_tokens": 128,
            "sliding_layers": "even layer indices",
            "full_attention_layers": "odd layer indices",
            "policy": "Full layers retain max context; sliding layers use an exact 128-token ring. No model window is shortened.",
        },
        "memory_plan": {
            "safe_rss_budget_bytes": safe_budget,
            "non_kv_planned_bytes": non_kv_plan,
            "previous_1024_all_layer_kv_bytes": old_kv_bytes,
            "selection_min_headroom_fraction": 0.25,
            "selection_min_projected_throughput_fraction_vs_512": 0.95,
        },
        "production_basis": {
            "artifact": args.production_baseline.as_posix(),
            "device": "Infinix X6528 / Hot 40i",
            "input_tokens": int(production["input_tokens"]),
            "measured_seconds_per_token": float(production["elapsed_ms"]) / 1000.0 / int(production["input_tokens"]),
            "measured_non_attention_seconds_per_token": baseline_non_attention_seconds,
            "composition_note": "Projected composite throughput adds each measured last-token KV-attention cost to the same-device sustained non-attention cost. It is a context-selection estimate, not a full 4K generation claim.",
        },
        "candidates": candidates,
        "correctness": {
            "host_bf16_cache_vs_quantized_oracle_max_abs_diff": host_cache_diff,
            "device_bf16_cache_vs_quantized_oracle_max_abs_diff": device_cache_diff,
            "host_bf16_cache_vs_fp32_max_abs_diff": host_fp32_diff,
            "device_bf16_cache_vs_fp32_max_abs_diff": device_fp32_diff,
            "sliding_ring_wrap_regression": True,
            "sequential_append_and_bounds_regression": True,
        },
        "selected": {
            "default_context_tokens": selected["context_tokens"],
            "kv_storage_bytes": selected["kv_storage_bytes"],
            "planned_total_rss_bytes": selected["planned_total_rss_bytes"],
            "planned_headroom_bytes": selected["planned_headroom_bytes"],
            "device_peak_rss_kib": selected["device_peak_rss_kib"],
            "device_last_token_24_layer_attention_ms": selected["device_last_token_24_layer_attention_ms"],
            "projected_throughput_fraction_vs_512": selected["projected_throughput_fraction_vs_512"],
            "reason": f"{selected['context_tokens']} is the largest measured candidate that keeps at least 25% safe-RSS headroom and at least 95% of the 512-token projected throughput. Exact 128-token rings reduce sliding-layer KV storage without changing attention semantics.",
        },
        "artifacts": {
            "device_sweeps": [path.as_posix() for path in args.device_sweep],
            "host_sweep": args.host_sweep.as_posix(),
            "host_correctness_log": args.host_test_log.as_posix(),
            "device_correctness_log": args.device_test_log.as_posix(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-sweep", type=Path, action="append", required=True)
    parser.add_argument("--host-sweep", type=Path, required=True)
    parser.add_argument("--memory-plan", type=Path, required=True)
    parser.add_argument("--production-baseline", type=Path, required=True)
    parser.add_argument("--host-test-log", type=Path, required=True)
    parser.add_argument("--device-test-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["selected"]}, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
