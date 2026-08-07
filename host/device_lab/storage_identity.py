from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from host.device_lab.adb import AdbClient
from host.device_lab.parsers import parse_getprop

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _classify_storage(props: dict[str, str], block_by_name: str, storage_sysfs: str) -> tuple[str, list[str]]:
    evidence: list[str] = []
    boot_devices = props.get("ro.boot.boot_devices", "")
    persist_type = props.get("persist.storage.type", "")

    if ".ufs" in boot_devices or "/ufs" in boot_devices or re.search(r"\bsd[a-z]\b", block_by_name):
        evidence.append(f"ro.boot.boot_devices={boot_devices}")
        if re.search(r"==== /sys/block/sd[a-z]", storage_sysfs):
            evidence.append("sysfs exposes sd* block roots")
        if persist_type:
            evidence.append(f"persist.storage.type={persist_type}")
        return "UFS", evidence

    if "mmc" in boot_devices or "mmcblk" in block_by_name or re.search(r"==== /sys/block/mmcblk", storage_sysfs):
        evidence.append(f"ro.boot.boot_devices={boot_devices}")
        if persist_type:
            evidence.append(f"persist.storage.type={persist_type}")
        return "eMMC", evidence

    if boot_devices:
        evidence.append(f"ro.boot.boot_devices={boot_devices}")
    if persist_type:
        evidence.append(f"persist.storage.type={persist_type}")
    return "UNKNOWN", evidence


def identify_storage(*, out: Path, serial: str | None, manifest_path: Path | None = None) -> str:
    client = AdbClient(serial=serial)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    probes = {
        "getprop": client.probe("getprop").check(),
        "block_by_name": client.probe("block_by_name").check(),
        "storage_sysfs": client.probe("storage_sysfs").check(),
        "df": client.probe("df").check(),
        "mounts": client.probe("mounts").check(),
    }

    props = parse_getprop(probes["getprop"].stdout)
    storage_class, class_evidence = _classify_storage(
        props, probes["block_by_name"].stdout, probes["storage_sysfs"].stdout
    )
    data_fs = None
    data_device = None
    for line in probes["mounts"].stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "/data":
            data_device = parts[0]
            data_fs = parts[2]
            break

    model_location = "/data/local/tmp/h40_models"
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "stock_storage_identity",
        "device": {
            "serial": serial,
            "product": props.get("ro.product.name"),
            "model": props.get("ro.product.model"),
            "hardware": props.get("ro.boot.hardware"),
            "board_platform": props.get("ro.board.platform"),
        },
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": {
            "model_benchmark_location": model_location,
            "manifest": str(manifest_path) if manifest_path else None,
        },
        "metrics": {
            "storage_classification": storage_class,
            "classification_evidence_count": len(class_evidence),
            "data_filesystem": data_fs,
            "data_device": data_device,
            "boot_devices": props.get("ro.boot.boot_devices"),
            "dynamic_partitions": props.get("ro.boot.dynamic_partitions"),
        },
        "samples": [
            {
                "name": name,
                "command": result.argv,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            for name, result in probes.items()
        ],
        "classification_evidence": class_evidence,
        "limitations": [],
    }
    output = out if out.is_absolute() else ROOT / out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return storage_class
