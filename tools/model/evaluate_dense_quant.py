#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(x, w):
    return [dot(x, col) for col in zip(*w)]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def rms_norm(x, weight, eps):
    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [(v / rms) * w for v, w in zip(x, weight)]


def silu(x):
    return x / (1.0 + math.exp(-x))


def clamp(x, limit):
    return max(-limit, min(limit, x))


def reshape_heads(x, n_heads, head_dim):
    return [x[i * head_dim : (i + 1) * head_dim] for i in range(n_heads)]


def causal_attention(x_norm, wq, wk, wv, wo, n_heads, head_dim):
    q = [reshape_heads(matvec(row, wq), n_heads, head_dim) for row in x_norm]
    k = [reshape_heads(matvec(row, wk), n_heads, head_dim) for row in x_norm]
    v = [reshape_heads(matvec(row, wv), n_heads, head_dim) for row in x_norm]
    scale = 1.0 / math.sqrt(head_dim)
    rows = []
    for pos in range(len(x_norm)):
        merged = []
        for head in range(n_heads):
            scores = [dot(q[pos][head], k[src][head]) * scale for src in range(pos + 1)]
            probs = softmax(scores)
            for dim in range(head_dim):
                merged.append(sum(probs[src] * v[src][head][dim] for src in range(pos + 1)))
        rows.append(matvec(merged, wo))
    return rows


def top_k(values, k):
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def quant_dequant(values: Any, bits: int):
    flat = []

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        else:
            flat.append(float(value))

    collect(values)
    max_abs = max((abs(v) for v in flat), default=0.0)
    max_q = (1 << (bits - 1)) - 1
    scale = max_abs / max_q if max_abs else 1.0

    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        q = round(float(value) / scale)
        q = max(-max_q, min(max_q, q))
        return q * scale

    return rewrite(values), {"scale": scale, "values": len(flat), "storage_bytes": math.ceil(len(flat) * bits / 8) + 4}


def run_network(fixture, weights):
    cfg = fixture["config"]
    tokens = fixture["tokens"]
    x = [weights["embedding"][token] for token in tokens]
    x_attn_norm = [rms_norm(row, weights["attn_norm_weight"], cfg["rms_eps"]) for row in x]
    attn = causal_attention(x_attn_norm, weights["wq"], weights["wk"], weights["wv"], weights["wo"], cfg["n_heads"], cfg["head_dim"])
    post_attention = [add(row, out) for row, out in zip(x, attn)]
    x_ffn_norm = [rms_norm(row, weights["ffn_norm_weight"], cfg["rms_eps"]) for row in post_attention]
    router_logits = [matvec(row, weights["router_w"]) for row in x_ffn_norm]
    router_ids = [top_k(row, cfg["top_k"]) for row in router_logits]
    router_scores = [softmax([router_logits[pos][i] for i in ids]) for pos, ids in enumerate(router_ids)]

    moe_output = []
    for pos, row in enumerate(x_ffn_norm):
        per_expert = []
        for expert in weights["experts"]:
            gate = [clamp(v, cfg["swiglu_limit"]) for v in matvec(row, expert["w_gate"])]
            up = [clamp(v, cfg["swiglu_limit"]) for v in matvec(row, expert["w_up"])]
            hidden = [silu(g) * u for g, u in zip(gate, up)]
            per_expert.append(matvec(hidden, expert["w_down"]))
        weighted = [0.0] * cfg["d_model"]
        for score, expert_id in zip(router_scores[pos], router_ids[pos]):
            weighted = [acc + score * value for acc, value in zip(weighted, per_expert[expert_id])]
        moe_output.append(weighted)

    layer_output = [add(row, moe) for row, moe in zip(post_attention, moe_output)]
    final_norm = [rms_norm(row, weights["final_norm_weight"], cfg["rms_eps"]) for row in layer_output]
    return [matvec(row, weights["lm_head"]) for row in final_norm], router_ids


def max_abs(a, b):
    return max(abs(x - y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--accept-logit-diff", type=float, default=0.01)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    dense_roles = ["attention", "embedding", "lm_head", "router"]
    dense_bytes = sum(inventory["totals_by_role"][role] for role in dense_roles)
    expert_bytes = inventory["totals_by_role"]["expert"]

    candidates = ["embedding", "wq", "wk", "wv", "wo", "router_w", "lm_head"]
    golden_logits = fixture["golden"]["logits"]
    golden_router_ids = fixture["golden"]["router_ids"]
    results = []
    for bits in [8, 6, 5, 4]:
        weights = json.loads(json.dumps(fixture["weights"]))
        storage_bytes = 0
        source_values = 0
        for name in candidates:
            weights[name], stats = quant_dequant(weights[name], bits)
            storage_bytes += stats["storage_bytes"]
            source_values += stats["values"]
        logits, router_ids = run_network(fixture, weights)
        logit_diff = max_abs(logits, golden_logits)
        routing_match = router_ids == golden_router_ids
        estimated_bytes = math.ceil(dense_bytes * bits / 16)
        accepted = routing_match and logit_diff <= args.accept_logit_diff
        results.append(
            {
                "bits": bits,
                "accepted": accepted,
                "max_abs_logit_diff": logit_diff,
                "routing_match": routing_match,
                "tiny_quantized_values": source_values,
                "tiny_quantized_storage_bytes": storage_bytes,
                "estimated_dense_bytes": estimated_bytes,
                "estimated_reduction_bytes": dense_bytes - estimated_bytes,
            }
        )

    accepted = [item for item in results if item["accepted"]]
    selected = min(accepted, key=lambda item: item["estimated_dense_bytes"]) if accepted else None
    report = {
        "schema_version": 1,
        "fixture": fixture["name"],
        "dense_roles": dense_roles,
        "baseline_dense_bytes": dense_bytes,
        "expert_bytes_out_of_scope": expert_bytes,
        "accept_logit_diff": args.accept_logit_diff,
        "results": results,
        "selected": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(json.dumps({"schema_version": 1, "dense_quant_policy": selected}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selected is None:
        raise SystemExit("no dense quantization candidate met acceptance threshold")


if __name__ == "__main__":
    main()
