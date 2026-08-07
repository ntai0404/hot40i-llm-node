#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
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

    for process in ("adb.exe",):
        run(["taskkill", "/F", "/IM", process], timeout=30)
    for args in (["kill-server"], ["start-server"], ["devices", "-l"]):
        result = run([str(ADB), *args], timeout=30)
    if not any(line.split()[:2] == [serial, "device"] for line in result.stdout.splitlines()):
        raise SystemExit(f"ADB device {serial} is not available:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def shell(serial: str, command: str, *, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return adb(serial, "shell", command, timeout=timeout)


def find_binary(path: Path) -> Path:
    candidates = [
        path,
        path / "llm_bench",
        path / "transformers" / "llm" / "engine" / "tools" / "llm_bench",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(path.rglob("llm_bench")) if path.exists() else []
    if matches:
        return matches[0]
    raise SystemExit(f"missing llm_bench under {path}")


def parse_json_lines(stdout: str, json_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in (json_text, stdout):
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def rates(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    prefill: list[float] = []
    decode: list[float] = []
    for row in rows:
        for result in row.get("results", []):
            if "prefill_tps" in result:
                prefill.append(float(result["prefill_tps"]))
            if "decode_tps" in result:
                decode.append(float(result["decode_tps"]))
    return (
        statistics.median(prefill) if prefill else None,
        statistics.median(decode) if decode else None,
    )


def write_blocked(args: argparse.Namespace, reason: str, details: dict[str, Any]) -> None:
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "benchmark": "mnn_android_backend_evaluation",
        "status": "blocked",
        "runnable": False,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": reason,
        "details": details,
        "config": {
            "mnn_ref": "3c97f7e19b3da8454bf9b46855d9c0c29e40658f",
            "requested_equivalence": {
                "fixture": str(args.gguf),
                "threads": args.threads,
                "n_prompt": args.n_prompt,
                "n_gen": args.n_gen,
                "runs": args.runs,
            },
        },
        "limitations": [
            "MNN gguf2mnn requires an existing MNN LLM model directory containing llm.mnn.json and llm_config.json; the llama.cpp tiny GGUF fixture alone is not a complete runnable MNN model.",
        ],
    }
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote blocked MNN evaluation to {out}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("artifacts/build/mnn-android-arm64"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/runs/20260807T165722Z_B01/stories260K_mnn"))
    parser.add_argument("--gguf", type=Path, default=Path(r"C:\tmp\stories260K.gguf"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/runtimes/mnn.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-prompt", type=int, default=512)
    parser.add_argument("--n-gen", type=int, default=128)
    args = parser.parse_args()

    build_dir = args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir
    model_dir = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    missing_template = [
        str(path)
        for path in (model_dir / "llm.mnn.json", model_dir / "llm_config.json")
        if not path.exists()
    ]
    if missing_template:
        write_blocked(
            args,
            "mnn_gguf_conversion_requires_existing_template_model",
            {
                "missing_template_files": missing_template,
                "model_dir": str(model_dir),
                "converter": "third_party/MNN/transformers/llm/export/gguf2mnn.py",
            },
        )
        return

    binary = find_binary(build_dir)
    if not (model_dir / "config.json").exists():
        write_blocked(
            args,
            "converted_mnn_model_missing_config_json",
            {"model_dir": str(model_dir), "binary": str(binary)},
        )
        return

    adb_devices = ensure_device(args.serial)
    remote_bin = "/data/local/tmp/b01_llm_bench"
    remote_model_dir = "/data/local/tmp/b01_mnn_model"
    adb(args.serial, "push", str(binary), remote_bin, timeout=180).check_returncode()
    shell(args.serial, f"rm -rf {remote_model_dir}", timeout=30).check_returncode()
    adb(args.serial, "push", str(model_dir), remote_model_dir, timeout=300).check_returncode()
    shell(args.serial, f"chmod 700 {remote_bin}", timeout=30).check_returncode()
    help_probe = shell(args.serial, f"{remote_bin} --help", timeout=60)

    samples: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        prefix = f"/data/local/tmp/b01_mnn_{index}"
        shell(args.serial, f"rm -f {prefix}.out {prefix}.err {prefix}.json {prefix}.rc {prefix}.rss", timeout=30)
        command = (
            f"cd /data/local/tmp; "
            f"(./b01_llm_bench -m {remote_model_dir}/config.json -a cpu -t {args.threads} "
            f"-pg {args.n_prompt},{args.n_gen} -rep 1 -j {prefix}.json "
            f"> {prefix}.out 2> {prefix}.err & "
            f"pid=$!; echo $pid > {prefix}.pid; peak=0; "
            f"while kill -0 $pid 2>/dev/null; do "
            f"rss=$(awk '/VmRSS/{{print $2}}' /proc/$pid/status 2>/dev/null); "
            f"if [ -n \"$rss\" ] && [ \"$rss\" -gt \"$peak\" ]; then peak=$rss; fi; "
            f"sleep 0.05; "
            f"done; wait $pid; rc=$?; echo $peak > {prefix}.rss; echo $rc > {prefix}.rc)"
        )
        shell(args.serial, command, timeout=600).check_returncode()
        stdout = shell(args.serial, f"cat {prefix}.out", timeout=60).stdout
        stderr = shell(args.serial, f"cat {prefix}.err", timeout=60).stdout
        json_text = shell(args.serial, f"cat {prefix}.json 2>/dev/null", timeout=60).stdout
        rc_text = shell(args.serial, f"cat {prefix}.rc", timeout=60).stdout.strip()
        rss_text = shell(args.serial, f"cat {prefix}.rss", timeout=60).stdout.strip()
        rows = parse_json_lines(stdout, json_text)
        pp_rate, tg_rate = rates(rows)
        sample = {
            "run_index": index,
            "exit_code": int(rc_text),
            "rss_peak_kib": int(rss_text) if rss_text.isdigit() else None,
            "prompt_tokens_per_second": pp_rate,
            "decode_tokens_per_second": tg_rate,
            "mnn_bench_rows": rows,
            "stdout": stdout,
            "stderr": stderr,
        }
        (run_dir / f"mnn_bench_run_{index}.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        samples.append(sample)

    prompt_rates = [row["prompt_tokens_per_second"] for row in samples if row["prompt_tokens_per_second"] is not None]
    decode_rates = [row["decode_tokens_per_second"] for row in samples if row["decode_tokens_per_second"] is not None]
    rss_peaks = [row["rss_peak_kib"] for row in samples if row["rss_peak_kib"] is not None]
    document = {
        "schema_version": 1,
        "benchmark": "mnn_android_backend_evaluation",
        "status": "pass",
        "runnable": True,
        "device": {"serial": args.serial, "adb_devices": adb_devices},
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "mnn_ref": "3c97f7e19b3da8454bf9b46855d9c0c29e40658f",
            "binary": str(binary.relative_to(ROOT)),
            "model_dir": str(model_dir),
            "runs": args.runs,
            "threads": args.threads,
            "n_prompt": args.n_prompt,
            "n_gen": args.n_gen,
            "backend": "cpu",
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
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(samples)} runs")


if __name__ == "__main__":
    main()
