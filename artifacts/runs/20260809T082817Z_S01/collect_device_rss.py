#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ADB = r"C:\mobile-remote-tools\platform-tools\adb.exe"
BENCHMARK = ROOT / "benchmarks/custom/memory_arena.json"


def run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise SystemExit(
            f"command failed {result.returncode}: {args}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def parse_status_kb(status: str, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}:\s+(\d+)\s+kB$", status, re.MULTILINE)
    return int(match.group(1)) if match else None


def main() -> None:
    devices = run([ADB, "devices", "-l"]).stdout
    has_target = re.search(r"^112193741U000563\s+device\b", devices, re.MULTILINE) is not None

    toolchain_checks = {
        "device_clangpp": run([ADB, "shell", "which", "clang++"], check=False).returncode == 0,
        "device_gpp": run([ADB, "shell", "which", "g++"], check=False).returncode == 0,
    }
    native_ls = run(
        [ADB, "shell", "ls", "-l", "/data/local/tmp/h40_cpu_memory_bench"],
        check=False,
    )
    native_benchmark_present = native_ls.returncode == 0

    status = ""
    probe_exit = None
    if has_target and native_benchmark_present:
        script = (
            "/data/local/tmp/h40_cpu_memory_bench --threads 1 --cpus 0 "
            "--seconds 3 --mem-bytes 67108864 >/dev/null & "
            "pid=$!; sleep 1; cat /proc/$pid/status; wait $pid; echo PROBE_EXIT:$?"
        )
        probe = run([ADB, "shell", script], check=False)
        status = probe.stdout
        probe_exit_match = re.search(r"PROBE_EXIT:(\d+)", status)
        probe_exit = int(probe_exit_match.group(1)) if probe_exit_match else probe.returncode

    document = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    document["device_rss_probe"] = {
        "adb_devices": devices.strip(),
        "target_device_present": has_target,
        "native_benchmark_present": native_benchmark_present,
        "native_benchmark_ls": native_ls.stdout.strip() or native_ls.stderr.strip(),
        "probe_command": "h40_cpu_memory_bench --threads 1 --cpus 0 --seconds 3 --mem-bytes 67108864",
        "probe_exit_code": probe_exit,
        "vmrss_kb": parse_status_kb(status, "VmRSS"),
        "vmhwm_kb": parse_status_kb(status, "VmHWM"),
        "raw_status": status,
        "toolchain_checks": toolchain_checks,
        "limitation": (
            "S01 arena test was not Android-built because no host Android NDK/cross compiler or device C++ "
            "compiler is available in this environment; RSS probe uses the deployed native benchmark path."
        ),
    }
    BENCHMARK.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document["device_rss_probe"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
