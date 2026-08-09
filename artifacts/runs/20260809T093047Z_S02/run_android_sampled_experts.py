from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADB = Path(r"C:\mobile-remote-tools\platform-tools\adb.exe")
PROBE = ROOT / "build-android-direct/h40_expert_loader_probe"
SAMPLE_BIN = ROOT / "artifacts/runs/20260809T093047Z_S02/h40m_sampled_experts.bin"
SAMPLE_JSON = ROOT / "artifacts/runs/20260809T093047Z_S02/h40m_sampled_experts.json"
OUT_JSONL = ROOT / "artifacts/runs/20260809T093047Z_S02/android_expert_loader_results.jsonl"
DEVICE_PROBE = "/data/local/tmp/h40_expert_loader_probe"
DEVICE_SAMPLE = "/data/local/tmp/h40m_sampled_experts.bin"


def run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def adb(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return run([str(ADB), *args], check=check)


def ensure_device() -> str:
    devices = adb(["devices", "-l"], check=False)
    has_device = any(line.split()[1:2] == ["device"] for line in devices.stdout.splitlines())
    if not has_device:
        adb(["kill-server"], check=False)
        adb(["start-server"], check=False)
        devices = adb(["devices", "-l"], check=False)
        has_device = any(line.split()[1:2] == ["device"] for line in devices.stdout.splitlines())
    if devices.returncode != 0 or not has_device:
        raise RuntimeError(f"no adb device available\nstdout={devices.stdout}\nstderr={devices.stderr}")
    return devices.stdout


def main() -> None:
    device_listing = ensure_device()
    adb(["push", str(PROBE), DEVICE_PROBE])
    adb(["shell", "chmod", "755", DEVICE_PROBE])
    adb(["push", str(SAMPLE_BIN), DEVICE_SAMPLE])
    sample = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
    rows = []
    with OUT_JSONL.open("w", encoding="utf-8") as output:
        for record in sample["records"]:
            proc = adb(
                [
                    "shell",
                    DEVICE_PROBE,
                    "--file",
                    DEVICE_SAMPLE,
                    "--layer",
                    str(record["layer"]),
                    "--expert",
                    str(record["expert_id"]),
                    "--offset",
                    str(record["sample_offset"]),
                    "--length",
                    str(record["length"]),
                    "--sha256",
                    record["sha256"],
                ]
            )
            payload = json.loads(proc.stdout)
            payload["exit_code"] = proc.returncode
            rows.append(payload)
            output.write(json.dumps(payload, separators=(",", ":")) + "\n")
    summary = {
        "device_listing": device_listing.strip().splitlines(),
        "record_count": len(rows),
        "all_verified": all(row["checksum_verified"] for row in rows),
        "bytes_loaded": sum(row["loaded_bytes"] for row in rows),
        "provider_operations": sum(row["provider_operations"] for row in rows),
    }
    print(json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
