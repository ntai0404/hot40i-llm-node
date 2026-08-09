#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAYOUT = ROOT / "artifacts/model/h40m/expert_arena.layout.json"
BENCHMARK = ROOT / "benchmarks/model/h40m_layout.json"
ARENA = ROOT / "artifacts/model/h40m/expert_arena.bin"


def main() -> None:
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    records = layout["records"]
    rng = random.Random(4002)
    sample_indexes = sorted(rng.sample(range(len(records)), 16))
    samples = []
    total_bytes = 0
    digest = hashlib.sha256()
    started = time.perf_counter()
    with ARENA.open("rb") as handle:
        for index in sample_indexes:
            record = records[index]
            handle.seek(record["offset"])
            data = handle.read(record["length"])
            if len(data) != record["length"]:
                raise EOFError(f"short read for record {index}")
            record_sha = hashlib.sha256(data).hexdigest()
            matched = record_sha == record["sha256"]
            samples.append(
                {
                    "layer": record["layer"],
                    "expert_id": record["expert_id"],
                    "offset": record["offset"],
                    "length": record["length"],
                    "sha256": record_sha,
                    "matched": matched,
                }
            )
            digest.update(data)
            total_bytes += len(data)
    elapsed = time.perf_counter() - started
    mib_per_second = total_bytes / (1024 * 1024) / elapsed

    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    benchmark["arena_read_probe"] = {
        "sample_count": len(samples),
        "bytes_read": total_bytes,
        "seconds": elapsed,
        "mib_per_second": mib_per_second,
        "combined_sha256": digest.hexdigest(),
        "all_record_hashes_match": all(sample["matched"] for sample in samples),
        "samples": samples,
        "limitation": "Host filesystem read probe over generated arena; device flash behavior is represented by D04 until the full arena is pushed to Android storage.",
    }
    benchmark["status"] = "pass" if benchmark["arena_read_probe"]["all_record_hashes_match"] else "fail"
    BENCHMARK.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(benchmark["arena_read_probe"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
