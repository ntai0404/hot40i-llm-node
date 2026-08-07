from __future__ import annotations

import datetime as dt
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .adb import AdbClient
from .thermal import _parse_battery, _parse_cpu_freq

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def parse_cpu_topology(text: str) -> list[dict[str, int | None]]:
    cpus: list[dict[str, int | None]] = []
    current: dict[str, int | None] | None = None
    fields = ("core_id", "package_id", "max_freq_khz", "cur_freq_khz", "capacity")
    index = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("==== "):
            if current:
                cpus.append(current)
            cpu_name = line.rsplit("/", 1)[-1]
            current = {"cpu": int(cpu_name.removeprefix("cpu"))}
            index = 0
            continue
        if current is None or index >= len(fields):
            continue
        current[fields[index]] = int(line) if line.isdigit() else None
        index += 1
    if current:
        cpus.append(current)
    return cpus


def select_configs(cpus: list[dict[str, int | None]]) -> list[dict[str, Any]]:
    online = sorted(
        (cpu for cpu in cpus if cpu.get("cpu") is not None),
        key=lambda item: int(item["cpu"]),
    )
    if not online:
        online = [{"cpu": i, "max_freq_khz": None, "capacity": None} for i in range(8)]
    ranked = sorted(
        online,
        key=lambda item: (
            int(item.get("max_freq_khz") or 0),
            int(item.get("capacity") or 0),
            int(item["cpu"]),
        ),
    )
    high_freq = ranked[-1].get("max_freq_khz")
    big = [item for item in ranked if item.get("max_freq_khz") == high_freq][-2:] if high_freq else ranked[-2:]
    little = [item for item in ranked if item not in big]
    if not little:
        little = ranked[:-2] or ranked
    return [
        {"name": "a75_focused", "cpus": [int(item["cpu"]) for item in big], "threads": len(big)},
        {"name": "a55_focused", "cpus": [int(item["cpu"]) for item in little], "threads": len(little)},
        {"name": "mixed_all_core", "cpus": [int(item["cpu"]) for item in online], "threads": len(online)},
    ]


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for config in sorted({row["config_name"] for row in samples}):
        rows = [row for row in samples if row["config_name"] == config]
        summary[config] = {
            "repeats": len(rows),
            "cpus": rows[0]["cpus"],
            "threads": rows[0]["threads"],
            "memcpy_mib_per_second_median": statistics.median(row["memcpy_mib_per_second"] for row in rows),
            "read_mib_per_second_median": statistics.median(row["read_mib_per_second"] for row in rows),
            "int8_matvec_gops_median": statistics.median(row["int8_matvec_gops"] for row in rows),
            "int4_matvec_gops_median": statistics.median(row["int4_matvec_gops"] for row in rows),
        }
    best = max(summary.items(), key=lambda item: item[1]["int8_matvec_gops_median"])
    summary["best_by_int8_matvec"] = {"config_name": best[0], **best[1]}
    return summary


def run_cpu_memory_benchmark(
    *,
    serial: str,
    local_binary: Path,
    out: Path,
    samples_jsonl: Path,
    repeats: int = 3,
    seconds: float = 0.35,
    mem_bytes: int = 32 * 1024 * 1024,
    remote_binary: str = "/data/local/tmp/h40_cpu_memory_bench",
) -> dict[str, Any]:
    client = AdbClient(serial=serial)
    local_binary = local_binary if local_binary.is_absolute() else ROOT / local_binary
    out = out if out.is_absolute() else ROOT / out
    samples_jsonl = samples_jsonl if samples_jsonl.is_absolute() else ROOT / samples_jsonl
    samples_jsonl.parent.mkdir(parents=True, exist_ok=True)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    topology_result = client.probe("cpu_topology").check()
    topology = parse_cpu_topology(topology_result.stdout)
    configs = select_configs(topology)
    client.push(local_binary, remote_binary).check()
    client.make_tmp_executable(remote_binary).check()

    rows: list[dict[str, Any]] = []
    with samples_jsonl.open("w", encoding="utf-8") as handle:
        for repeat in range(1, repeats + 1):
            for config in configs:
                freq_before = _parse_cpu_freq(client.probe("cpu_freq").stdout)
                battery_before = _parse_battery(client.probe("battery").stdout)
                result = client.run_tmp_binary(
                    remote_binary,
                    [
                        "--threads",
                        str(config["threads"]),
                        "--cpus",
                        ",".join(str(cpu) for cpu in config["cpus"]),
                        "--seconds",
                        str(seconds),
                        "--mem-bytes",
                        str(mem_bytes),
                    ],
                    timeout=300,
                ).check()
                freq_after = _parse_cpu_freq(client.probe("cpu_freq").stdout)
                battery_after = _parse_battery(client.probe("battery").stdout)
                row = json.loads(result.stdout)
                row.update(
                    {
                        "repeat": repeat,
                        "config_name": config["name"],
                        "exit_code": result.returncode,
                        "cpu_freq_before_khz": freq_before,
                        "cpu_freq_after_khz": freq_after,
                        "battery_before": battery_before,
                        "battery_after": battery_after,
                    }
                )
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    document = {
        "schema_version": 1,
        "benchmark": "stock_cpu_memory",
        "device": {"serial": serial},
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": {
            "remote_binary": remote_binary,
            "repeats": repeats,
            "seconds_per_kernel": seconds,
            "mem_bytes": mem_bytes,
            "configs": configs,
            "cpu_topology": topology,
            "topology_probe_stdout": topology_result.stdout,
        },
        "metrics": {
            "sample_count": len(rows),
            "summary_by_config": _summarize(rows),
        },
        "samples": rows,
        "artifacts": {"samples_jsonl": str(samples_jsonl.relative_to(ROOT))},
        "limitations": [
            "Affinity is requested per worker with sched_setaffinity; Android may still enforce scheduler and thermal policies.",
            "Matvec kernels are representative scalar INT8 and packed INT4 loops, not vendor-optimized GEMM kernels.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document
