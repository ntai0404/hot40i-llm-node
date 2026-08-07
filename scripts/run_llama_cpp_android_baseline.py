#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
import time
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
    for args in (["kill-server"], ["start-server"], ["devices", "-l"]):
        result = run([str(ADB), *args], timeout=30)
    if not any(line.split()[:2] == [serial, "device"] for line in result.stdout.splitlines()):
        raise SystemExit(f"ADB device {serial} is not available:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def shell(serial: str, command: str, *, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return adb(serial, "shell", command, timeout=timeout)


def parse_rss_kib(status: str) -> int | None:
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_bench_json(text: str) -> list[dict[str, Any]]:
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        return parsed["results"]
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("unexpected llama-bench JSON output")


def rates(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    prompt = []
    decode = []
    for row in rows:
        if row.get("n_prompt", 0):
            prompt.append(float(row.get("avg_ts") or row.get("t/s") or 0))
        if row.get("n_gen", 0):
            decode.append(float(row.get("avg_ts") or row.get("t/s") or 0))
    return (
        statistics.median(prompt) if prompt else None,
        statistics.median(decode) if decode else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--binary", type=Path, default=Path("artifacts/build/llama-cpp-android-arm64/bin/llama-bench"))
    parser.add_argument("--model", type=Path, default=Path(r"C:\tmp\stories260K.gguf"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/runtimes/llama_cpp.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-prompt", type=int, default=32)
    parser.add_argument("--n-gen", type=int, default=16)
    args = parser.parse_args()

    binary = args.binary if args.binary.is_absolute() else ROOT / args.binary
    out = args.out if args.out.is_absolute() else ROOT / args.out
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if not binary.exists():
        raise SystemExit(f"missing binary: {binary}")
    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")

    adb_devices = ensure_device(args.serial)
    remote_bin = "/data/local/tmp/b00_llama_bench"
    remote_model = "/data/local/tmp/b00_stories260K.gguf"
    adb(args.serial, "push", str(binary), remote_bin, timeout=180).check_returncode()
    adb(args.serial, "push", str(args.model), remote_model, timeout=180).check_returncode()
    shell(args.serial, f"chmod 700 {remote_bin}").check_returncode()
    help_probe = shell(args.serial, f"{remote_bin} --help", timeout=60)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    samples: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        prefix = f"/data/local/tmp/b00_llama_{index}"
        shell(args.serial, f"rm -f {prefix}.out {prefix}.err {prefix}.rc", timeout=30)
        before_battery = shell(args.serial, "dumpsys battery", timeout=30).stdout
        before_freq = shell(args.serial, "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo ==== $c; cat $c/cpufreq/scaling_cur_freq 2>/dev/null; done", timeout=30).stdout
        command = (
            f"cd /data/local/tmp; "
            f"(./b00_llama_bench -m {remote_model} -p {args.n_prompt} -n {args.n_gen} "
            f"-t {args.threads} -r 1 -o json > {prefix}.out 2> {prefix}.err & "
            f"pid=$!; echo $pid > {prefix}.pid; peak=0; "
            f"while kill -0 $pid 2>/dev/null; do "
            f"rss=$(awk '/VmRSS/{{print $2}}' /proc/$pid/status 2>/dev/null); "
            f"if [ -n \"$rss\" ] && [ \"$rss\" -gt \"$peak\" ]; then peak=$rss; fi; "
            f"sleep 0.05; "
            f"done; wait $pid; rc=$?; echo $peak > {prefix}.rss; echo $rc > {prefix}.rc)"
        )
        shell(args.serial, command, timeout=300).check_returncode()
        pid = shell(args.serial, f"cat {prefix}.pid", timeout=30).stdout.strip()
        stdout = shell(args.serial, f"cat {prefix}.out", timeout=60).stdout
        stderr = shell(args.serial, f"cat {prefix}.err", timeout=60).stdout
        rc_text = shell(args.serial, f"cat {prefix}.rc", timeout=60).stdout.strip()
        rss_text = shell(args.serial, f"cat {prefix}.rss", timeout=60).stdout.strip()
        after_battery = shell(args.serial, "dumpsys battery", timeout=30).stdout
        after_freq = shell(args.serial, "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo ==== $c; cat $c/cpufreq/scaling_cur_freq 2>/dev/null; done", timeout=30).stdout
        rows = parse_bench_json(stdout)
        pp_rate, tg_rate = rates(rows)
        sample = {
            "schema_version": 1,
            "run_index": index,
            "exit_code": int(rc_text),
            "pid": pid,
            "rss_peak_kib": int(rss_text) if rss_text.isdigit() else None,
            "prompt_tokens_per_second": pp_rate,
            "decode_tokens_per_second": tg_rate,
            "llama_bench_rows": rows,
            "stdout": stdout,
            "stderr": stderr,
            "battery_before": before_battery,
            "battery_after": after_battery,
            "cpu_freq_before": before_freq,
            "cpu_freq_after": after_freq,
        }
        (run_dir / f"llama_bench_run_{index}.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        samples.append(sample)

    prompt_rates = [row["prompt_tokens_per_second"] for row in samples if row["prompt_tokens_per_second"] is not None]
    decode_rates = [row["decode_tokens_per_second"] for row in samples if row["decode_tokens_per_second"] is not None]
    rss_peaks = [row["rss_peak_kib"] for row in samples if row["rss_peak_kib"] is not None]
    document = {
        "schema_version": 1,
        "benchmark": "llama_cpp_android_baseline",
        "device": {"serial": args.serial, "adb_devices": adb_devices},
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": {
            "llama_cpp_ref": "refs/tags/b10173",
            "llama_cpp_commit": "e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0",
            "binary": str(binary.relative_to(ROOT)),
            "remote_binary": remote_bin,
            "model_source": "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf",
            "model_sha256": "270CBA1BD5109F42D03350F60406024560464DB173C0E387D91F0426D3BD256D",
            "remote_model": remote_model,
            "runs": args.runs,
            "threads": args.threads,
            "n_prompt": args.n_prompt,
            "n_gen": args.n_gen,
            "output_format": "json",
        },
        "metrics": {
            "sample_count": len(samples),
            "exit_codes": [row["exit_code"] for row in samples],
            "prompt_tokens_per_second_median": statistics.median(prompt_rates) if prompt_rates else None,
            "decode_tokens_per_second_median": statistics.median(decode_rates) if decode_rates else None,
            "rss_peak_kib_max": max(rss_peaks) if rss_peaks else None,
        },
        "binary_help": {"exit_code": help_probe.returncode, "stdout": help_probe.stdout, "stderr": help_probe.stderr},
        "samples": samples,
        "limitations": [
            "Baseline uses llama-bench with the tiny stories260K fixture, not a gpt-oss production-size model.",
            "RSS is sampled from /proc while the process is running; very short peaks between samples may be missed.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(samples)} runs")


if __name__ == "__main__":
    main()
