#!/usr/bin/env python3
"""Validate A00 device-local HTTP responses and bounded RSS behavior."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_http(path: Path) -> tuple[int, dict[str, object]]:
    data = path.read_bytes()
    separator = b"\r\n\r\n" if b"\r\n\r\n" in data else b"\n\n"
    header, body = data.split(separator, 1)
    status = int(header.splitlines()[0].split()[1])
    return status, json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    device = args.run_dir / "device"
    responses = [read_http(device / f"a00_infer_{index}.http") for index in range(1, 4)]
    metrics = [read_http(device / f"a00_metrics_{index}.http")[1] for index in range(4)]
    health_status, health = read_http(device / "a00_final_health.http")
    final_metrics_status, final_metrics = read_http(device / "a00_final_metrics.http")

    assert all(status == 200 for status, _ in responses)
    payloads = [payload for _, payload in responses]
    assert all(payload["status"] == "pass" for payload in payloads)
    assert all(payload["input_tokens"] == 1 and payload["layers_run"] == 24 for payload in payloads)
    assert len({payload["emitted_token_id"] for payload in payloads}) == 1
    assert len({payload["emitted_token_logit"] for payload in payloads}) == 1

    assert metrics[0]["inference_requests"] == 0
    for index in range(1, 4):
        assert metrics[index]["inference_requests"] == index
        assert metrics[index]["completed_inference_requests"] == index
        assert metrics[index]["failures"] == 0
    budget_kib = int(metrics[-1]["rss_budget_kib"])
    service_rss = [int(sample["service_rss_kib"]) for sample in metrics[1:]]
    child_rss = [int(payload["peak_rss_kib"]) for payload in payloads]
    assert max(service_rss) <= budget_kib
    assert max(child_rss) <= budget_kib
    assert max(service_rss) - min(service_rss) <= 64

    assert health_status == 200 and health == {"status": "ok"}
    assert final_metrics_status == 200
    assert final_metrics["failures"] == 0
    assert final_metrics["service_rss_kib"] <= budget_kib
    assert (device / "a00_clean_stop.txt").read_text(encoding="utf-8").strip() == "stopped_cleanly"

    elapsed_ms = [int(payload["elapsed_ms"]) for payload in payloads]
    median_ms = int(statistics.median(elapsed_ms))
    benchmark = {
        "schema_version": 1,
        "task_id": "A00",
        "status": "pass",
        "device": {
            "serial": "112193741U000563",
            "model": "Infinix_X6528",
            "adb_transport": "usb",
        },
        "service": {
            "bind": "127.0.0.1:8080",
            "rss_budget_kib": budget_kib,
            "initial_rss_kib": int(metrics[0]["service_rss_kib"]),
            "post_request_rss_kib": service_rss,
            "post_warmup_rss_delta_kib": service_rss[-1] - service_rss[0],
            "post_warmup_rss_spread_kib": max(service_rss) - min(service_rss),
            "clean_stop": True,
            "clean_restart": True,
            "final_rss_kib": int(final_metrics["service_rss_kib"]),
        },
        "requests": {
            "count": len(payloads),
            "successful": len(payloads),
            "failures": 0,
            "input_token_id": 12194,
            "emitted_token_id": int(payloads[0]["emitted_token_id"]),
            "emitted_token_logit": float(payloads[0]["emitted_token_logit"]),
            "token_and_logit_agreement": True,
            "elapsed_ms": elapsed_ms,
            "median_elapsed_ms": median_ms,
            "median_tokens_per_second": 1000.0 / median_ms,
            "child_peak_rss_kib": child_rss,
            "max_child_peak_rss_kib": max(child_rss),
            "dense_threads": int(payloads[0]["dense_threads"]),
            "io_overlap_enabled": bool(payloads[0]["io_overlap_enabled"]),
            "cache_policy": payloads[0]["cache_policy"],
        },
        "endpoints": {
            "health_http_status": health_status,
            "metrics_http_status": final_metrics_status,
            "inference_http_statuses": [status for status, _ in responses],
        },
        "artifacts": {
            "telemetry": str(device / "a00_telemetry.log"),
            "thermal": str(device / "a00_final_thermal.txt"),
            "oom_scan": str(device / "a00_final_oom_scan.txt"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(benchmark, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
