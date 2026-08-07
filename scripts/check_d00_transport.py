#!/usr/bin/env python3
"""D00 stock-Android transport check.

This harness performs only non-destructive ADB operations:
- require exactly one authorized device;
- push/pull a harmless nonce under /data/local/tmp;
- configure adb forward localhost:18080 -> device:8080;
- write a machine-readable transport summary and update PROJECT_STATE.yaml.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
from pathlib import Path
from typing import Any

import yaml

from host.device_lab.adb import AdbClient

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "PROJECT_STATE.yaml"


def _load_state() -> dict[str, Any]:
    return yaml.safe_load(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(yaml.safe_dump(state, sort_keys=False, width=110), encoding="utf-8")


def _authorized_devices(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in raw.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        fields = {"serial": serial, "state": state}
        for token in parts[2:]:
            if ":" in token:
                key, value = token.split(":", 1)
                fields[key] = value
        if state == "device":
            rows.append(fields)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host-port", type=int, default=18080)
    parser.add_argument("--device-port", type=int, default=8080)
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    base = AdbClient()
    first_devices = base.devices().check().stdout
    devices = _authorized_devices(first_devices)
    if len(devices) != 1:
        raise SystemExit(f"expected exactly one authorized device, found {len(devices)}")

    serial = devices[0]["serial"]
    client = AdbClient(serial=serial)
    second_devices = client.devices().check().stdout

    nonce = secrets.token_hex(16)
    local_nonce = run_dir / "nonce.txt"
    pulled_nonce = run_dir / "nonce.pulled.txt"
    local_nonce.write_text(nonce + "\n", encoding="utf-8")

    remote_nonce = f"/data/local/tmp/h40_d00_nonce_{nonce[:8]}.txt"
    push = client.push(local_nonce, remote_nonce).check()
    pull = client.pull(remote_nonce, pulled_nonce).check()
    pulled_text = pulled_nonce.read_text(encoding="utf-8").strip()
    if pulled_text != nonce:
        raise SystemExit("pulled nonce did not match pushed nonce")

    forward = client.forward(args.host_port, args.device_port).check()
    forward_list = client._run(["forward", "--list"]).check()
    expected_forward = f"tcp:{args.host_port} tcp:{args.device_port}"
    forward_verified = serial in forward_list.stdout and expected_forward in forward_list.stdout
    if not forward_verified:
        raise SystemExit("adb forward entry was not visible in adb forward --list")

    state = _load_state()
    state.setdefault("decisions", {})["device_serial"] = serial
    _save_state(state)

    transport = {
        "schema_version": 1,
        "task_id": "D00",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "pass",
        "serial": serial,
        "device": devices[0],
        "adb_devices_first": first_devices,
        "adb_devices_second": second_devices,
        "push": {
            "remote": remote_nonce,
            "exit_code": push.returncode,
            "stdout": push.stdout,
            "stderr": push.stderr,
        },
        "pull": {
            "exit_code": pull.returncode,
            "stdout": pull.stdout,
            "stderr": pull.stderr,
            "nonce_match": True,
        },
        "forward": {
            "host_port": args.host_port,
            "device_port": args.device_port,
            "exit_code": forward.returncode,
            "verified": True,
            "forward_list": forward_list.stdout,
        },
    }
    output = run_dir / "transport.json"
    output.write_text(json.dumps(transport, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"D00_TRANSPORT_OK serial={serial} transport={output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
