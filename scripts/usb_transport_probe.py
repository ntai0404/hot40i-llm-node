#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

from host.transport.adb_forward import AdbForwardSupervisor, TransportUnavailable, resolve_adb_path


def request_health(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3.0) as response:
        if response.status != 200:
            raise RuntimeError(f"health returned HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"health payload is not ok: {payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial")
    parser.add_argument("--host-port", type=int, default=18080)
    parser.add_argument("--device-port", type=int, default=8080)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--drop-forward-at", type=int, default=34)
    parser.add_argument("--restart-server-at", type=int, default=67)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/service/usb_transport.json"))
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    supervisor = AdbForwardSupervisor(
        expected_serial=args.serial,
        host_port=args.host_port,
        device_port=args.device_port,
        adb_path=resolve_adb_path(),
    )
    initial = supervisor.recover()
    events: list[dict] = []
    latencies_ms: list[float] = []
    failures: list[dict] = []

    for request_index in range(1, args.requests + 1):
        if request_index == args.drop_forward_at:
            supervisor.remove_forward()
            detected: dict
            try:
                supervisor.inspect()
                detected = {"status": "unexpected_success"}
                failures.append({"request": request_index, "error": "removed forward was not detected"})
            except TransportUnavailable as exc:
                detected = exc.to_dict()
            recovered = supervisor.recover()
            events.append(
                {
                    "type": "forward_removed_and_recovered",
                    "before_request": request_index,
                    "detected": detected,
                    "recovered": recovered.to_dict(),
                }
            )
        if request_index == args.restart_server_at:
            supervisor.restart_adb_server()
            detected = None
            try:
                supervisor.inspect()
            except TransportUnavailable as exc:
                detected = exc.to_dict()
            recovered = supervisor.recover()
            events.append(
                {
                    "type": "adb_server_restarted_and_recovered",
                    "before_request": request_index,
                    "detected": detected,
                    "recovered": recovered.to_dict(),
                }
            )
        started = time.perf_counter_ns()
        try:
            payload = request_health(supervisor.endpoint)
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            latencies_ms.append(latency_ms)
            events.append(
                {
                    "type": "request",
                    "index": request_index,
                    "status": "pass",
                    "latency_ms": latency_ms,
                    "payload": payload,
                }
            )
        except Exception as exc:
            failure = {"request": request_index, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            events.append({"type": "request", "index": request_index, "status": "fail", **failure})

    final = supervisor.inspect()
    sorted_latency = sorted(latencies_ms)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latency)) - 1)
    document = {
        "schema_version": 1,
        "task_id": "A02",
        "status": "pass" if not failures and len(latencies_ms) == args.requests else "fail",
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "transport": {
            "kind": "USB ADB forward",
            "adb_path": supervisor.adb,
            "serial": final.serial,
            "host_port": final.host_port,
            "device_port": final.device_port,
            "endpoint": final.endpoint,
            "initial": initial.to_dict(),
            "final": final.to_dict(),
        },
        "requests": {
            "attempted": args.requests,
            "successful": len(latencies_ms),
            "failed": len(failures),
            "latency_ms_median": statistics.median(latencies_ms),
            "latency_ms_p95": sorted_latency[p95_index],
            "latency_ms_max": max(latencies_ms),
        },
        "reconnect_scenarios": [event for event in events if event["type"] != "request"],
        "failures": failures,
        "events_artifact": str(args.events),
    }
    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, sort_keys=True))
    return 0 if document["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
