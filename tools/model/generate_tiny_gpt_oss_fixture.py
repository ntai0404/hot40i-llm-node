#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


def rounded(value: float) -> float:
    return round(float(value), 10)


def round_nested(value):
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, float):
        return rounded(value)
    return value


def matrix(rows: int, cols: int, rng: random.Random, scale: float) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def vector(size: int, rng: random.Random, scale: float) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(size)]


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def matvec(x: list[float], w: list[list[float]]) -> list[float]:
    return [dot(x, col) for col in zip(*w)]


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def clamp(x: float, limit: float) -> float:
    return max(-limit, min(limit, x))


def rms_norm(x: list[float], weight: list[float], eps: float) -> list[float]:
    rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [(v / rms) * w for v, w in zip(x, weight)]


def reshape_heads(x: list[float], n_heads: int, head_dim: int) -> list[list[float]]:
    return [x[i * head_dim : (i + 1) * head_dim] for i in range(n_heads)]


def causal_attention(
    x_norm: list[list[float]],
    wq: list[list[float]],
    wk: list[list[float]],
    wv: list[list[float]],
    wo: list[list[float]],
    n_heads: int,
    head_dim: int,
) -> dict:
    q = [reshape_heads(matvec(row, wq), n_heads, head_dim) for row in x_norm]
    k = [reshape_heads(matvec(row, wk), n_heads, head_dim) for row in x_norm]
    v = [reshape_heads(matvec(row, wv), n_heads, head_dim) for row in x_norm]
    scale = 1.0 / math.sqrt(head_dim)
    attention_scores: list[list[list[float | None]]] = []
    attention_probs: list[list[list[float]]] = []
    context_rows: list[list[float]] = []
    for pos in range(len(x_norm)):
        merged: list[float] = []
        per_head_scores: list[list[float | None]] = []
        per_head_probs: list[list[float]] = []
        for head in range(n_heads):
            scores = [dot(q[pos][head], k[src][head]) * scale for src in range(pos + 1)]
            probs = softmax(scores)
            masked_scores: list[float | None] = scores + [None] * (len(x_norm) - pos - 1)
            padded_probs = probs + [0.0] * (len(x_norm) - pos - 1)
            per_head_scores.append(masked_scores)
            per_head_probs.append(padded_probs)
            for dim in range(head_dim):
                merged.append(sum(probs[src] * v[src][head][dim] for src in range(pos + 1)))
        attention_scores.append(per_head_scores)
        attention_probs.append(per_head_probs)
        context_rows.append(matvec(merged, wo))
    return {
        "q": q,
        "k": k,
        "v": v,
        "attention_scores": attention_scores,
        "attention_probs": attention_probs,
        "attention_output": context_rows,
    }


def top_k(values: list[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def build_fixture() -> dict:
    rng = random.Random(20260809)
    cfg = {
        "d_model": 8,
        "seq_len": 3,
        "n_heads": 2,
        "head_dim": 4,
        "n_experts": 4,
        "top_k": 2,
        "expert_hidden": 12,
        "vocab_size": 10,
        "rms_eps": 1e-5,
        "swiglu_limit": 7.0,
        "seed": 20260809,
    }
    tokens = [2, 5, 7]
    embedding = matrix(cfg["vocab_size"], cfg["d_model"], rng, 0.4)
    x = [embedding[token] for token in tokens]
    attn_norm_weight = vector(cfg["d_model"], rng, 0.2)
    attn_norm_weight = [1.0 + v for v in attn_norm_weight]
    ffn_norm_weight = vector(cfg["d_model"], rng, 0.2)
    ffn_norm_weight = [1.0 + v for v in ffn_norm_weight]
    final_norm_weight = vector(cfg["d_model"], rng, 0.2)
    final_norm_weight = [1.0 + v for v in final_norm_weight]
    wq = matrix(cfg["d_model"], cfg["d_model"], rng, 0.25)
    wk = matrix(cfg["d_model"], cfg["d_model"], rng, 0.25)
    wv = matrix(cfg["d_model"], cfg["d_model"], rng, 0.25)
    wo = matrix(cfg["d_model"], cfg["d_model"], rng, 0.25)
    router_w = matrix(cfg["d_model"], cfg["n_experts"], rng, 0.3)
    experts = []
    for _ in range(cfg["n_experts"]):
        experts.append(
            {
                "w_gate": matrix(cfg["d_model"], cfg["expert_hidden"], rng, 0.25),
                "w_up": matrix(cfg["d_model"], cfg["expert_hidden"], rng, 0.25),
                "w_down": matrix(cfg["expert_hidden"], cfg["d_model"], rng, 0.25),
            }
        )
    lm_head = matrix(cfg["d_model"], cfg["vocab_size"], rng, 0.3)

    x_attn_norm = [rms_norm(row, attn_norm_weight, cfg["rms_eps"]) for row in x]
    attn = causal_attention(
        x_attn_norm, wq, wk, wv, wo, cfg["n_heads"], cfg["head_dim"]
    )
    post_attention_residual = [add(row, out) for row, out in zip(x, attn["attention_output"])]
    x_ffn_norm = [rms_norm(row, ffn_norm_weight, cfg["rms_eps"]) for row in post_attention_residual]

    router_logits = [matvec(row, router_w) for row in x_ffn_norm]
    router_ids = [top_k(row, cfg["top_k"]) for row in router_logits]
    router_scores = [softmax([router_logits[pos][i] for i in ids]) for pos, ids in enumerate(router_ids)]

    all_expert_outputs: list[list[list[float]]] = []
    selected_expert_outputs: list[list[list[float]]] = []
    moe_output: list[list[float]] = []
    for pos, row in enumerate(x_ffn_norm):
        per_expert: list[list[float]] = []
        for expert in experts:
            gate = [clamp(v, cfg["swiglu_limit"]) for v in matvec(row, expert["w_gate"])]
            up = [clamp(v, cfg["swiglu_limit"]) for v in matvec(row, expert["w_up"])]
            hidden = [silu(g) * u for g, u in zip(gate, up)]
            per_expert.append(matvec(hidden, expert["w_down"]))
        all_expert_outputs.append(per_expert)
        selected = [per_expert[i] for i in router_ids[pos]]
        selected_expert_outputs.append(selected)
        weighted = [0.0] * cfg["d_model"]
        for score, output in zip(router_scores[pos], selected):
            weighted = [acc + score * value for acc, value in zip(weighted, output)]
        moe_output.append(weighted)

    layer_output = [add(row, moe) for row, moe in zip(post_attention_residual, moe_output)]
    final_norm = [rms_norm(row, final_norm_weight, cfg["rms_eps"]) for row in layer_output]
    logits = [matvec(row, lm_head) for row in final_norm]

    fixture = {
        "schema_version": 1,
        "name": "tiny_gpt_oss_shape_v1",
        "reference": {
            "source": "repo C00 deterministic Python reference shaped by docs/13_GPT_OSS_20B_MODEL_CONTRACT.md",
            "semantics": [
                "RMSNorm before attention and MoE",
                "causal scaled dot-product attention",
                "router top-k expert selection with softmax over selected logits",
                "SwiGLU expert path with symmetric activation clamp",
                "residual attention and MoE updates followed by output logits",
            ],
        },
        "config": cfg,
        "tokens": tokens,
        "weights": {
            "embedding": embedding,
            "attn_norm_weight": attn_norm_weight,
            "ffn_norm_weight": ffn_norm_weight,
            "final_norm_weight": final_norm_weight,
            "wq": wq,
            "wk": wk,
            "wv": wv,
            "wo": wo,
            "router_w": router_w,
            "experts": experts,
            "lm_head": lm_head,
        },
        "golden": {
            "input_embeddings": x,
            "attention_norm": x_attn_norm,
            "attention_scores": attn["attention_scores"],
            "attention_probs": attn["attention_probs"],
            "attention_output": attn["attention_output"],
            "post_attention_residual": post_attention_residual,
            "ffn_norm": x_ffn_norm,
            "router_logits": router_logits,
            "router_ids": router_ids,
            "router_scores": router_scores,
            "expert_outputs_all": all_expert_outputs,
            "expert_outputs_selected": selected_expert_outputs,
            "moe_output": moe_output,
            "layer_output": layer_output,
            "final_norm": final_norm,
            "logits": logits,
        },
    }
    rounded_fixture = round_nested(fixture)
    payload = json.dumps(rounded_fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rounded_fixture["sha256"] = hashlib.sha256(payload).hexdigest()
    return rounded_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("tests/fixtures/tiny_gpt_oss"))
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture()
    (out_dir / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Tiny gpt-oss-shaped fixture\n\n"
        "Deterministic C00 fixture covering RMSNorm, causal attention, router top-k, "
        "weighted MoE expert outputs, SwiGLU/clamping, residuals, final norm and logits.\n",
        encoding="utf-8",
    )
    print(fixture["sha256"])


if __name__ == "__main__":
    main()
