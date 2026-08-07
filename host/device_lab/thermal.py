from __future__ import annotations

import datetime as dt
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from host.device_lab.adb import AdbClient
from host.device_lab.parsers import parse_meminfo

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _parse_cpu_freq(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("==== "):
            current = line.rsplit("/", 1)[-1]
            continue
        if current and line.strip().isdigit():
            result[current] = int(line.strip())
            current = None
    return result


def _parse_battery(text: str) -> dict[str, str | int | bool]:
    result: dict[str, str | int | bool] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().replace(" ", "_")
        value = value.strip()
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def _parse_thermal_headers(text: str) -> list[str]:
    return [line.removeprefix("==== ").strip() for line in text.splitlines() if line.startswith("==== ")]


def collect_thermal_baseline(
    *,
    serial: str,
    duration_seconds: int,
    interval_seconds: float,
    thermal_jsonl: Path,
    summary_out: Path,
) -> None:
    client = AdbClient(serial=serial)
    thermal_path = thermal_jsonl if thermal_jsonl.is_absolute() else ROOT / thermal_jsonl
    summary_path = summary_out if summary_out.is_absolute() else ROOT / summary_out
    thermal_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    workload_result: dict[str, Any] = {}

    def _workload() -> None:
        try:
            result = client.run_harmless_workload(duration_seconds)
            workload_result.update(
                {
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        except Exception as exc:  # pragma: no cover - recorded in artifact
            workload_result.update({"exit_code": 1, "error": str(exc)})

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start_monotonic = time.monotonic()
    worker = threading.Thread(target=_workload, daemon=True)
    worker.start()

    samples: list[dict[str, Any]] = []
    with thermal_path.open("w", encoding="utf-8") as handle:
        while True:
            elapsed = time.monotonic() - start_monotonic
            mem = client.probe("meminfo")
            cpu = client.probe("cpu_freq")
            thermal = client.probe("thermal_zones")
            battery = client.probe("battery")
            sample = {
                "schema_version": 1,
                "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "monotonic_seconds": elapsed,
                "meminfo": parse_meminfo(mem.stdout) if mem.returncode == 0 else {},
                "cpu_freq_khz": _parse_cpu_freq(cpu.stdout),
                "thermal_zone_paths": _parse_thermal_headers(thermal.stdout),
                "thermal_probe_exit_code": thermal.returncode,
                "battery": _parse_battery(battery.stdout) if battery.returncode == 0 else {},
                "probe_exit_codes": {
                    "meminfo": mem.returncode,
                    "cpu_freq": cpu.returncode,
                    "thermal_zones": thermal.returncode,
                    "battery": battery.returncode,
                },
            }
            samples.append(sample)
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            handle.flush()
            if elapsed >= duration_seconds:
                break
            time.sleep(interval_seconds)

    worker.join(timeout=30)
    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    cpu_sample_count = sum(1 for sample in samples if sample["cpu_freq_khz"])
    mem_available_values = [
        sample["meminfo"].get("MemAvailable")
        for sample in samples
        if sample["meminfo"].get("MemAvailable") is not None
    ]
    summary = {
        "schema_version": 1,
        "benchmark": "stock_thermal_baseline",
        "device": {"serial": serial},
        "started_at": started_at,
        "finished_at": finished_at,
        "git_commit": _git_commit(),
        "config": {
            "duration_seconds": duration_seconds,
            "interval_seconds": interval_seconds,
            "workload": "read-only meminfo/build.prop/cpufreq loop via adb shell",
        },
        "metrics": {
            "sample_count": len(samples),
            "duration_observed_seconds": samples[-1]["monotonic_seconds"] if samples else 0,
            "cpu_freq_sample_count": cpu_sample_count,
            "thermal_zone_count_max": max((len(sample["thermal_zone_paths"]) for sample in samples), default=0),
            "mem_available_min_bytes": min(mem_available_values) if mem_available_values else None,
            "workload_exit_code": workload_result.get("exit_code"),
        },
        "samples": [
            {
                "thermal_jsonl": str(thermal_path.relative_to(ROOT)),
                "first_monotonic_seconds": samples[0]["monotonic_seconds"] if samples else None,
                "last_monotonic_seconds": samples[-1]["monotonic_seconds"] if samples else None,
            }
        ],
        "workload": workload_result,
        "limitations": [
            "Thermal zone type/temp contents are not readable through the current non-root stock Android probe; zone paths and exit code are retained.",
            "Process RSS for the shell workload is not attributable with the current safe wrapper; MemAvailable is sampled instead.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
