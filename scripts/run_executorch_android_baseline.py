#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ADB = Path(r"C:\mobile-remote-tools\platform-tools\adb.exe")


def run(argv: list[str], *, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


def adb(serial: str, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return run([str(ADB), "-s", serial, *args], timeout=timeout)


def ensure_device(serial: str) -> str:
    result = run([str(ADB), "devices", "-l"], timeout=20)
    if any(line.split()[:2] == [serial, "device"] for line in result.stdout.splitlines()):
        return result.stdout

    run(["taskkill", "/F", "/IM", "adb.exe"], timeout=30)
    for args in (["kill-server"], ["start-server"], ["devices", "-l"]):
        result = run([str(ADB), *args], timeout=30)
    if not any(line.split()[:2] == [serial, "device"] for line in result.stdout.splitlines()):
        raise SystemExit(f"ADB device {serial} is not available:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def shell(serial: str, command: str, *, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return adb(serial, "shell", command, timeout=timeout)


def read_remote(serial: str, path: str) -> str:
    return shell(serial, f"cat {path} 2>/dev/null", timeout=60).stdout


def excerpt(text: str, *, head: int = 20, tail: int = 20) -> dict[str, Any]:
    lines = text.splitlines()
    return {
        "line_count": len(lines),
        "head": lines[:head],
        "tail": lines[-tail:] if len(lines) > head else [],
    }


def parse_duration_ms(stdout: str, stderr: str) -> float | None:
    matches = re.findall(
        r"Model executed successfully\s+\d+\s+time\(s\)\s+in\s+([0-9.]+)\s+ms",
        stdout + "\n" + stderr,
    )
    if matches:
        return float(matches[-1])
    for text in (stdout, stderr):
        for line in text.splitlines():
            lower = line.lower()
            if "time" not in lower and "duration" not in lower and "elapsed" not in lower:
                continue
            numbers = [token.strip(",:;()[]") for token in line.split()]
            for index, token in enumerate(numbers):
                try:
                    value = float(token)
                except ValueError:
                    continue
                suffix = numbers[index + 1].lower() if index + 1 < len(numbers) else ""
                if suffix in {"ms", "millisecond", "milliseconds"}:
                    return value
                if suffix in {"s", "sec", "secs", "second", "seconds"}:
                    return value * 1000.0
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--runner", type=Path, default=Path("artifacts/build/executorch-android-arm64/executor_runner"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/runs/20260807T172453Z_B02/linear_xnnpack_fp32.pte"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/runtimes/executorch.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--executions", type=int, default=1000)
    args = parser.parse_args()

    runner = args.runner if args.runner.is_absolute() else ROOT / args.runner
    model = args.model if args.model.is_absolute() else ROOT / args.model
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if not runner.exists():
        raise SystemExit(f"missing executor_runner: {runner}")
    if not model.exists():
        raise SystemExit(f"missing model: {model}")

    adb_devices = ensure_device(args.serial)
    remote_dir = "/data/local/tmp/b02_executorch"
    remote_runner = f"{remote_dir}/executor_runner"
    remote_model = f"{remote_dir}/linear_xnnpack_fp32.pte"
    shell(args.serial, f"rm -rf {remote_dir}; mkdir -p {remote_dir}", timeout=30).check_returncode()
    adb(args.serial, "push", str(runner), remote_runner, timeout=180).check_returncode()
    adb(args.serial, "push", str(model), remote_model, timeout=60).check_returncode()
    shell(args.serial, f"chmod 700 {remote_runner}", timeout=30).check_returncode()

    help_probe = shell(args.serial, f"{remote_runner} --help", timeout=60)
    meminfo_before = shell(args.serial, "cat /proc/meminfo | head -n 8", timeout=30)
    thermal_before = shell(
        args.serial,
        "for z in /sys/class/thermal/thermal_zone*/temp; do echo $z=$(cat $z 2>/dev/null); done | head -n 12",
        timeout=30,
    )

    samples: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        prefix = f"{remote_dir}/run_{index}"
        shell(args.serial, f"rm -f {prefix}.out {prefix}.err {prefix}.rc {prefix}.rss", timeout=30)
        command = (
            f"cd {remote_dir}; "
            f"(./executor_runner --model_path={remote_model} --num_executions={args.executions} --print_output=none > {prefix}.out 2> {prefix}.err & "
            f"pid=$!; echo $pid > {prefix}.pid; peak=0; "
            f"while kill -0 $pid 2>/dev/null; do "
            f"rss=$(awk '/VmRSS/{{print $2}}' /proc/$pid/status 2>/dev/null); "
            f"if [ -n \"$rss\" ] && [ \"$rss\" -gt \"$peak\" ]; then peak=$rss; fi; "
            f"sleep 0.02; "
            f"done; wait $pid; rc=$?; echo $peak > {prefix}.rss; echo $rc > {prefix}.rc)"
        )
        shell(args.serial, command, timeout=120).check_returncode()
        stdout = read_remote(args.serial, f"{prefix}.out")
        stderr = read_remote(args.serial, f"{prefix}.err")
        stdout_path = run_dir / f"executorch_run_{index}.stdout.log"
        stderr_path = run_dir / f"executorch_run_{index}.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        rc_text = read_remote(args.serial, f"{prefix}.rc").strip()
        rss_text = read_remote(args.serial, f"{prefix}.rss").strip()
        sample = {
            "run_index": index,
            "exit_code": int(rc_text) if rc_text.lstrip("-").isdigit() else None,
            "rss_peak_kib": int(rss_text) if rss_text.isdigit() else None,
            "duration_ms": parse_duration_ms(stdout, stderr),
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
            "stdout_excerpt": excerpt(stdout),
            "stderr_excerpt": excerpt(stderr),
        }
        (run_dir / f"executorch_run_{index}.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        samples.append(sample)

    meminfo_after = shell(args.serial, "cat /proc/meminfo | head -n 8", timeout=30)
    thermal_after = shell(
        args.serial,
        "for z in /sys/class/thermal/thermal_zone*/temp; do echo $z=$(cat $z 2>/dev/null); done | head -n 12",
        timeout=30,
    )
    rss_peaks = [row["rss_peak_kib"] for row in samples if row["rss_peak_kib"] is not None]
    durations = [row["duration_ms"] for row in samples if row["duration_ms"] is not None]
    document = {
        "schema_version": 1,
        "benchmark": "executorch_xnnpack_android_backend_evaluation",
        "status": "pass" if all(row["exit_code"] == 0 for row in samples) else "blocked",
        "runnable": all(row["exit_code"] == 0 for row in samples),
        "device": {"serial": args.serial, "adb_devices": adb_devices},
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "executorch_ref": "812c7f0227d04fd343042fdf376aa90631c25995",
            "runner": str(runner.relative_to(ROOT)),
            "model": str(model.relative_to(ROOT)),
            "model_description": "ExecuTorch examples.xnnpack linear toy model, fp32, XNNPACK delegated",
            "runs": args.runs,
            "executions_per_run": args.executions,
        },
        "metrics": {
            "sample_count": len(samples),
            "exit_codes": [row["exit_code"] for row in samples],
            "duration_ms_median": statistics.median(durations) if durations else None,
            "single_execution_ms_median": (statistics.median(durations) / args.executions) if durations else None,
            "rss_peak_kib_max": max(rss_peaks) if rss_peaks else None,
        },
        "device_state": {
            "meminfo_before": meminfo_before.stdout,
            "meminfo_after": meminfo_after.stdout,
            "thermal_before": thermal_before.stdout,
            "thermal_after": thermal_after.stdout,
        },
        "binary_help": {
            "exit_code": help_probe.returncode,
            "stdout": help_probe.stdout,
            "stderr": help_probe.stderr,
        },
        "samples": samples,
        "limitations": [
            "Fixture is a tiny upstream linear model, not a gpt-oss/GGUF LLM; it evaluates Android ExecuTorch/XNNPACK build and delegate runtime viability only.",
            "executor_runner output does not expose kernel throughput counters for this fixture; RSS and process success are measured directly.",
        ],
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(samples)} runs")


if __name__ == "__main__":
    main()
