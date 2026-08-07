from __future__ import annotations

import datetime as dt
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from host.device_lab.adb import AdbClient
from host.device_lab.parsers import parse_meminfo

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "PROJECT_STATE.yaml"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state() -> dict[str, Any]:
    return yaml.safe_load(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(yaml.safe_dump(state, sort_keys=False, width=110), encoding="utf-8")


def _sample(client: AdbClient, label: str) -> dict[str, Any]:
    result = client.probe("meminfo").check()
    values = parse_meminfo(result.stdout)
    return {
        "label": label,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mem_total_bytes": values.get("MemTotal"),
        "mem_available_bytes": values.get("MemAvailable"),
        "mem_free_bytes": values.get("MemFree"),
        "cached_bytes": values.get("Cached"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
        "raw": result.stdout,
    }


def measure_memory_budget(
    *,
    out: Path,
    samples_count: int,
    interval_seconds: float,
    serial: str | None,
    manifest_path: Path,
) -> int:
    manifest = _load_json(manifest_path)
    selected_serial = serial or manifest.get("serial")
    if not selected_serial:
        raise SystemExit("no serial supplied and manifest.serial is empty")

    client = AdbClient(serial=selected_serial)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    samples: list[dict[str, Any]] = []
    for index in range(samples_count):
        samples.append(_sample(client, f"settled_idle_{index + 1}"))
        if index + 1 < samples_count:
            time.sleep(interval_seconds)

    pressure_before = _sample(client, "cache_pressure_before")
    pressure_result = client.run_bounded_cache_pressure(
        "/data/local/tmp/h40_d02_cache_pressure.bin", 128
    )
    pressure_after = _sample(client, "cache_pressure_after")

    available = [item["mem_available_bytes"] for item in samples if item["mem_available_bytes"] is not None]
    if len(available) < 3:
        raise SystemExit("expected at least three MemAvailable samples")

    min_available = min(available)
    median_available = int(statistics.median(available))
    safe_rss_budget = min(min_available // 2, 768 * 1024 * 1024)
    headroom = min_available - safe_rss_budget

    result = {
        "schema_version": 1,
        "benchmark": "stock_memory_budget",
        "device": {
            "serial": selected_serial,
            "manifest_summary": manifest.get("summary", {}),
        },
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": {
            "samples": samples_count,
            "interval_seconds": interval_seconds,
            "budget_rule": "min(MemAvailable_samples) / 2 capped at 768MiB",
            "native_allocator_probe": "not_run_ndk_toolchain_unavailable",
            "bounded_pressure_probe": "128MiB /data/local/tmp file write/read/remove",
        },
        "metrics": {
            "mem_available_min_bytes": int(min_available),
            "mem_available_median_bytes": int(median_available),
            "safe_rss_budget_bytes": int(safe_rss_budget),
            "headroom_bytes": int(headroom),
            "swap_total_bytes": samples[-1].get("swap_total_bytes"),
            "swap_free_bytes": samples[-1].get("swap_free_bytes"),
            "pressure_probe_exit_code": pressure_result.returncode,
            "pressure_after_mem_available_bytes": pressure_after.get("mem_available_bytes"),
        },
        "samples": [*samples, pressure_before, pressure_after],
        "pressure_probe": {
            "argv": pressure_result.argv,
            "exit_code": pressure_result.returncode,
            "stdout": pressure_result.stdout,
            "stderr": pressure_result.stderr,
            "size_mib": 128,
        },
        "limitations": [
            "Native process RSS allocation pressure probe was not run because no Android NDK/aarch64 toolchain is installed on the host; the bounded pressure probe exercises storage/page-cache pressure only.",
            "Cold-reboot and screen-off variants were not used for this first budget to avoid losing an active authorized ADB session mid-DAG.",
        ],
    }
    if pressure_result.returncode != 0:
        raise SystemExit("bounded pressure probe failed; refusing to record safe_rss_budget_bytes")

    state = _load_state()
    state.setdefault("decisions", {})["safe_rss_budget_bytes"] = int(safe_rss_budget)
    _save_state(state)
    output = out if out.is_absolute() else ROOT / out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return int(safe_rss_budget)
