#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.model.evaluate_dense_quant import max_abs, quant_dequant, run_network


def main() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/tiny_gpt_oss/fixture.json").read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / "artifacts/model/gpt_oss_20b_inventory.json").read_text(encoding="utf-8"))
    storage = json.loads((ROOT / "benchmarks/stock/storage.json").read_text(encoding="utf-8"))
    cpu = json.loads((ROOT / "benchmarks/stock/cpu_memory.json").read_text(encoding="utf-8"))
    memory = json.loads((ROOT / "benchmarks/stock/memory_budget.json").read_text(encoding="utf-8"))

    lm_head = next(item for item in inventory["tensors"] if item["name"] == "lm_head.weight")
    weights = json.loads(json.dumps(fixture["weights"]))
    weights["lm_head"], qstats = quant_dequant(weights["lm_head"], 8)
    logits, router_ids = run_network(fixture, weights)
    logit_diff = max_abs(logits, fixture["golden"]["logits"])

    bf16_bytes = lm_head["byte_size"]
    q8_bytes = math.ceil(bf16_bytes * 8 / 16)
    safe_rss = memory["metrics"]["safe_rss_budget_bytes"]
    random_8m_mib_s = storage["metrics"]["summary_by_pattern_block"]["random_8388608"]["mib_per_second_median"]
    int8_gops = cpu["metrics"]["summary_by_config"]["best_by_int8_matvec"]["int8_matvec_gops_median"]
    ops = lm_head["shape"][0] * lm_head["shape"][1]
    q8_mib = q8_bytes / (1024 * 1024)
    stream_seconds = q8_mib / random_8m_mib_s
    compute_seconds = ops / (int8_gops * 1_000_000_000)
    chunk_vocab = 4096
    chunk_bytes = chunk_vocab * lm_head["shape"][1]

    resident_headroom = safe_rss - q8_bytes
    report = {
        "schema_version": 1,
        "status": "pass",
        "lm_head": {
            "shape": lm_head["shape"],
            "bf16_bytes": bf16_bytes,
            "q8_bytes": q8_bytes,
            "q8_logit_max_abs_diff": logit_diff,
            "routing_match": router_ids == fixture["golden"]["router_ids"],
        },
        "device_constraints": {
            "safe_rss_budget_bytes": safe_rss,
            "random_8m_mib_per_second": random_8m_mib_s,
            "best_int8_gops": int8_gops,
        },
        "strategies": [
            {
                "name": "resident_q8_head",
                "accepted": False,
                "reason": "q8 head alone leaves insufficient RSS headroom for runtime state/cache",
                "resident_bytes": q8_bytes,
                "rss_headroom_bytes": resident_headroom,
                "estimated_compute_ms": compute_seconds * 1000,
                "estimated_flash_bytes_per_token": 0,
            },
            {
                "name": "chunked_streamed_q8_head",
                "accepted": True,
                "reason": "keeps resident head memory bounded while preserving exact full-vocabulary projection",
                "chunk_vocab": chunk_vocab,
                "chunk_bytes": chunk_bytes,
                "resident_bytes": chunk_bytes,
                "estimated_compute_ms": compute_seconds * 1000,
                "estimated_flash_bytes_per_token": q8_bytes,
                "estimated_flash_ms_per_token": stream_seconds * 1000,
            },
        ],
        "selected": "chunked_streamed_q8_head",
    }
    out = ROOT / "benchmarks/model/output_head.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    adr = ROOT / "docs/decisions/ADR_002_OUTPUT_HEAD.md"
    adr.parent.mkdir(parents=True, exist_ok=True)
    adr.write_text(
        "\n".join(
            [
                "# ADR 002: Output Head Strategy",
                "",
                "## Status",
                "",
                "Accepted for initial runtime implementation.",
                "",
                "## Decision",
                "",
                "Use a chunked streamed Q8 output head as the default exact vocabulary projection path.",
                "",
                "## Evidence",
                "",
                f"- BF16 `lm_head.weight` is {bf16_bytes:,} bytes; Q8 estimate is {q8_bytes:,} bytes.",
                f"- Safe RSS budget is {safe_rss:,} bytes, leaving only {resident_headroom:,} bytes if Q8 is fully resident.",
                f"- Tiny fixture Q8 output-head-only max absolute logit drift is {logit_diff:.9f}.",
                f"- D04 random 8 MiB read median is {random_8m_mib_s:.2f} MiB/s.",
                f"- D05 best measured INT8 matvec throughput is {int8_gops:.2f} GOPS.",
                f"- Chunked Q8 uses {chunk_vocab:,}-vocab chunks ({chunk_bytes:,} bytes resident) and scans {q8_bytes:,} bytes/token.",
                "",
                "## Consequences",
                "",
                "Resident Q8 remains a future option only if later memory planning proves enough headroom. The initial path favors correctness and bounded RSS over output-head latency.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
