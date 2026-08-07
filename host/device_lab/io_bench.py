from __future__ import annotations

import datetime as dt
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .adb import AdbClient

ROOT = Path(__file__).resolve().parents[2]


BLOCK_SPECS = (
    (4 * 1024, 4096),
    (64 * 1024, 2048),
    (256 * 1024, 1024),
    (1024 * 1024, 256),
    (8 * 1024 * 1024, 64),
    (16 * 1024 * 1024, 32),
    (32 * 1024 * 1024, 16),
)
PATTERNS = ("sequential", "random")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        key = f"{sample['pattern']}_{sample['block_bytes']}"
        grouped.setdefault(key, []).append(sample)

    summary: dict[str, Any] = {}
    for key, rows in grouped.items():
        summary[key] = {
            "repeats": len(rows),
            "mib_per_second_median": statistics.median(row["mib_per_second"] for row in rows),
            "mib_per_second_min": min(row["mib_per_second"] for row in rows),
            "latency_ms_p50_median": statistics.median(row["latency_ms_p50"] for row in rows),
            "latency_ms_p95_median": statistics.median(row["latency_ms_p95"] for row in rows),
            "iops_median": statistics.median(row["iops"] for row in rows),
        }
    return summary


def run_storage_benchmark(
    *,
    serial: str,
    local_binary: Path,
    manifest_path: Path,
    samples_jsonl: Path,
    out: Path,
    repeats: int = 3,
    file_size_mib: int = 512,
    remote_binary: str = "/data/local/tmp/h40_io_bench",
    remote_file: str = "/data/local/tmp/h40_d04_storage.bin",
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    client = AdbClient(serial=serial)
    local_binary = local_binary if local_binary.is_absolute() else ROOT / local_binary
    samples_jsonl = samples_jsonl if samples_jsonl.is_absolute() else ROOT / samples_jsonl
    out = out if out.is_absolute() else ROOT / out

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    client.push(local_binary, remote_binary).check()
    client.make_tmp_executable(remote_binary).check()
    create_result = client.create_tmp_zero_file(remote_file, file_size_mib, timeout=900).check()

    samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    with samples_jsonl.open("w", encoding="utf-8") as handle:
        for repeat in range(1, repeats + 1):
            for pattern in PATTERNS:
                for block_bytes, reads in BLOCK_SPECS:
                    result = client.run_tmp_binary(
                        remote_binary,
                        [
                            "--file",
                            remote_file,
                            "--pattern",
                            pattern,
                            "--block-bytes",
                            str(block_bytes),
                            "--reads",
                            str(reads),
                            "--seed",
                            str(0x4040 + repeat),
                        ],
                        timeout=600,
                    ).check()
                    row = json.loads(result.stdout)
                    row.update(
                        {
                            "repeat": repeat,
                            "remote_binary": remote_binary,
                            "remote_file": remote_file,
                            "exit_code": result.returncode,
                        }
                    )
                    samples.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    document = {
        "schema_version": 1,
        "benchmark": "stock_expert_shaped_flash_access",
        "device": {
            "serial": serial,
            "manifest_summary": manifest.get("summary", {}),
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": _git_commit(),
        "config": {
            "remote_binary": remote_binary,
            "remote_file": remote_file,
            "file_size_mib": file_size_mib,
            "repeats": repeats,
            "patterns": list(PATTERNS),
            "block_specs": [
                {"block_bytes": block_bytes, "reads": reads}
                for block_bytes, reads in BLOCK_SPECS
            ],
            "shape_note": "Expert-shaped accesses use random block reads over a fixed precreated /data/local/tmp file; sequential rows provide the baseline.",
        },
        "metrics": {
            "sample_count": len(samples),
            "summary_by_pattern_block": _summarize(samples),
            "benchmark_file_create_exit_code": create_result.returncode,
        },
        "samples": samples,
        "artifacts": {
            "samples_jsonl": str(samples_jsonl.relative_to(ROOT)),
        },
        "limitations": [
            "The benchmark uses direct userspace file reads on stock Android /data/local/tmp and does not bypass kernel page cache.",
            "The expert-shaped pattern approximates expert page/block fetches with seeded random fixed-size reads; final model layout may shift absolute latency.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document
