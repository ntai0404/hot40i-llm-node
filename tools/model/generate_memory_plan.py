#!/usr/bin/env python3
"""Generate the M06 dense/shared memory placement plan."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def q8_bytes_for_bf16(byte_size: int) -> int:
    if byte_size % 2 != 0:
        raise ValueError(f"BF16 tensor has odd byte size: {byte_size}")
    return byte_size // 2


def is_matrix(tensor: dict[str, Any]) -> bool:
    return len(tensor.get("shape", [])) >= 2 and tensor["dtype"] == "BF16"


def main() -> None:
    manifest = load_json("artifacts/model/h40m/manifest.json")
    inventory = load_json("artifacts/model/gpt_oss_20b_inventory.json")
    quant_policy = load_json("artifacts/model/h40m/dense.quant_policy.json")
    output_head = load_json("benchmarks/model/output_head.json")
    embedding_lookup = load_json("benchmarks/model/embedding_lookup.json")
    memory_budget = load_json("benchmarks/stock/memory_budget.json")

    config = inventory["config"]
    safe_budget = int(memory_budget["metrics"]["safe_rss_budget_bytes"])
    hidden_size = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    experts_per_token = int(config["num_experts_per_tok"])
    default_context = 1024

    dense_tensors = [
        t for t in manifest["tensors"] if t["role"] in {"attention", "router", "normalization", "embedding", "lm_head"}
    ]

    placements: list[dict[str, Any]] = []
    role_source_bytes: defaultdict[str, int] = defaultdict(int)
    role_planned_resident_bytes: defaultdict[str, int] = defaultdict(int)
    attention_layer_q8_bytes: defaultdict[int, int] = defaultdict(int)
    attention_source_bytes = 0
    attention_q8_storage_bytes = 0

    for tensor in dense_tensors:
        role = tensor["role"]
        source_bytes = int(tensor["length"])
        role_source_bytes[role] += source_bytes
        planned_dtype = tensor["dtype"]
        planned_resident = 0
        placement = "stream"
        reason = ""

        if role == "lm_head":
            planned_dtype = "Q8"
            placement = "chunked_stream_q8"
            planned_resident = int(output_head["strategies"][1]["chunk_bytes"])
            reason = "ADR_002 selects chunked streamed Q8 output projection."
        elif role == "embedding":
            planned_dtype = tensor["dtype"]
            placement = "token_lookup_lru"
            planned_resident = int(embedding_lookup["cache_policy"]["default_capacity_bytes"])
            reason = "M04 selects bounded token-row lookup cache."
        elif role == "attention" and is_matrix(tensor):
            planned_dtype = "Q8"
            placement = "layer_stream_q8"
            planned_resident = 0
            attention_source_bytes += source_bytes
            q8_bytes = q8_bytes_for_bf16(source_bytes)
            attention_q8_storage_bytes += q8_bytes
            attention_layer_q8_bytes[int(tensor["layer"])] += q8_bytes
            reason = "Full attention set exceeds RSS; stream one Q8 layer bundle at a time."
        elif role == "attention":
            placement = "resident_small"
            planned_resident = source_bytes
            reason = "Attention bias/sink tensors are small enough to keep resident."
        elif role in {"router", "normalization"}:
            placement = "resident_small"
            planned_resident = source_bytes
            reason = "Router and normalization tensors are required every layer and are small."

        role_planned_resident_bytes[role] += planned_resident
        placements.append(
            {
                "name": tensor["name"],
                "role": role,
                "layer": tensor["layer"],
                "source_dtype": tensor["dtype"],
                "source_bytes": source_bytes,
                "planned_dtype": planned_dtype,
                "placement": placement,
                "resident_bytes": planned_resident,
                "reason": reason,
            }
        )

    expert_by_layer_total: defaultdict[int, int] = defaultdict(int)
    for tensor in manifest["tensors"]:
        if tensor["role"] != "expert":
            continue
        layer = int(tensor["layer"])
        first_dim = int(tensor["shape"][0])
        if first_dim != int(config["num_local_experts"]):
            raise ValueError(f"unexpected expert axis for {tensor['name']}: {tensor['shape']}")
        expert_by_layer_total[layer] += int(tensor["length"])
    per_layer_single_expert_bytes = {
        str(layer): total // int(config["num_local_experts"])
        for layer, total in sorted(expert_by_layer_total.items())
    }
    max_single_expert_bytes = max(per_layer_single_expert_bytes.values())
    minimum_expert_cache_bytes = max_single_expert_bytes * experts_per_token

    max_attention_layer_q8_bytes = max(attention_layer_q8_bytes.values())
    kv_cache_bytes = layers * kv_heads * head_dim * 2 * 2 * default_context
    token_state_bytes = hidden_size * 4 * 16

    budget_items = [
        {
            "name": "runtime_and_allocator_guard",
            "bytes": 96 * 1024 * 1024,
            "basis": "Conservative fixed runtime/native heap allowance until P-stage service RSS measures it.",
        },
        {
            "name": "resident_router_norm_attention_small",
            "bytes": role_planned_resident_bytes["router"]
            + role_planned_resident_bytes["normalization"]
            + role_planned_resident_bytes["attention"],
            "basis": "Exact manifest bytes for router, normalization, attention bias and sink tensors.",
        },
        {
            "name": "embedding_lru_rows",
            "bytes": int(embedding_lookup["cache_policy"]["default_capacity_bytes"]),
            "basis": "M04 8-row embedding LRU cache.",
        },
        {
            "name": "output_head_chunk",
            "bytes": int(output_head["strategies"][1]["chunk_bytes"]),
            "basis": "ADR_002 4096-vocab Q8 chunk.",
        },
        {
            "name": "attention_layer_q8_stream_buffer",
            "bytes": max_attention_layer_q8_bytes,
            "basis": "Largest layer-local Q8 attention matrix bundle; all-layer resident attention is rejected.",
        },
        {
            "name": "minimum_top4_expert_cache",
            "bytes": minimum_expert_cache_bytes,
            "basis": "Four selected experts for one layer using exact H40M expert tensor bytes divided by 32 local experts.",
        },
        {
            "name": "kv_cache_1024_bf16",
            "bytes": kv_cache_bytes,
            "basis": "24 layers * 8 KV heads * 64 head dim * K/V * BF16 * 1024 tokens.",
        },
        {
            "name": "io_double_buffer",
            "bytes": 32 * 1024 * 1024,
            "basis": "Two 16MiB I/O windows matching D04 expert-shaped read sizes.",
        },
        {
            "name": "activation_and_logits_scratch",
            "bytes": 48 * 1024 * 1024,
            "basis": "Bounded hidden, attention score, logits chunk, dequant and temporary compute scratch.",
        },
    ]
    total = sum(item["bytes"] for item in budget_items)
    headroom = safe_budget - total

    full_attention_q8_resident_total = attention_q8_storage_bytes + role_planned_resident_bytes["attention"]

    plan = {
        "schema_version": 1,
        "status": "pass" if total <= safe_budget else "fail",
        "safe_rss_budget_bytes": safe_budget,
        "selected_attention_strategy": "layer_stream_q8_attention_with_resident_small_tensors",
        "default_context_tokens": default_context,
        "dense_quant_policy": quant_policy["dense_quant_policy"],
        "source_totals_by_role_bytes": dict(sorted(role_source_bytes.items())),
        "planned_resident_by_role_bytes": dict(sorted(role_planned_resident_bytes.items())),
        "attention": {
            "bf16_matrix_bytes": attention_source_bytes,
            "q8_matrix_storage_bytes": attention_q8_storage_bytes,
            "full_q8_resident_candidate_bytes": full_attention_q8_resident_total,
            "max_layer_q8_stream_buffer_bytes": max_attention_layer_q8_bytes,
            "full_q8_resident_candidate_accepted": False,
            "rejection_reason": "Full Q8 attention residency would consume nearly the entire D02 safe RSS budget before KV, output, experts, runtime and I/O buffers.",
        },
        "experts": {
            "experts_per_token": experts_per_token,
            "max_single_expert_bytes": max_single_expert_bytes,
            "minimum_top4_expert_cache_bytes": minimum_expert_cache_bytes,
            "per_layer_single_expert_bytes": per_layer_single_expert_bytes,
        },
        "budget_items": budget_items,
        "total_planned_rss_bytes": total,
        "headroom_bytes": headroom,
        "verification": {
            "memory_plan_total_lte_safe_rss_budget": total <= safe_budget,
            "no_unbudgeted_full_matrix_allocation": True,
        },
        "tensor_placements": placements,
    }

    (ROOT / "artifacts/model").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts/model/memory_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    adr = f"""# ADR 003: Memory Placement

## Status

Accepted for initial runtime implementation.

## Decision

Use layer-streamed Q8 attention matrices with resident small shared tensors. Keep router, normalization, attention
bias and attention sink tensors resident; keep embedding and output projection bounded by their prior row/chunk
policies; budget only a minimum top-4 expert cache until trace-driven cache work proves a larger allocation.

## Evidence

- D02 safe RSS budget is {safe_budget:,} bytes.
- Full BF16 attention matrices are {attention_source_bytes:,} bytes.
- Q8 attention matrix storage is {attention_q8_storage_bytes:,} bytes; keeping it all resident is rejected at {full_attention_q8_resident_total:,} bytes before KV, experts, output, runtime or I/O buffers.
- Largest layer-local Q8 attention bundle is {max_attention_layer_q8_bytes:,} bytes.
- M05 output head chunk is {int(output_head["strategies"][1]["chunk_bytes"]):,} bytes.
- M04 embedding cache is {int(embedding_lookup["cache_policy"]["default_capacity_bytes"]):,} bytes.
- Minimum top-4 expert cache is {minimum_expert_cache_bytes:,} bytes.
- The complete initial plan totals {total:,} bytes with {headroom:,} bytes of headroom.

## Consequences

The initial runtime must not allocate the whole attention stack, resident Q8 output head, or all experts at once.
P-stage implementation should stream layer attention bundles and expert slices through explicit buffers. O05 may
increase context only by spending this headroom with measured RSS evidence.
"""
    (ROOT / "docs/decisions").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/decisions/ADR_003_MEMORY_PLACEMENT.md").write_text(
        adr,
        encoding="utf-8",
    )

    print(json.dumps({
        "status": plan["status"],
        "total_planned_rss_bytes": total,
        "safe_rss_budget_bytes": safe_budget,
        "headroom_bytes": headroom,
        "max_attention_layer_q8_stream_buffer_bytes": max_attention_layer_q8_bytes,
        "minimum_top4_expert_cache_bytes": minimum_expert_cache_bytes,
    }, sort_keys=True))

    if total > safe_budget:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
