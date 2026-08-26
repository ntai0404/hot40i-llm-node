from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


RUN = Path("artifacts/runs/20260827T022000Z_F01")
OUT = Path("benchmarks/final/sustained_30m.json")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def http_json(path: Path) -> dict:
    raw = read_text(path)
    separator = "\r\n\r\n" if "\r\n\r\n" in raw else "\n\n"
    body = raw.split(separator, 1)[1] if separator in raw else raw
    return json.loads(body)


def parse_requests() -> list[dict[str, int]]:
    rows: dict[int, dict[str, int]] = {}
    pattern = re.compile(r"request=(\d+) (start|end)_epoch=(\d+)(?: rc=(\d+) elapsed_seconds=(\d+))?")
    for line in read_text(RUN / "f01_requests.log").splitlines():
        match = pattern.fullmatch(line.strip())
        if not match:
            continue
        request, phase, epoch, rc, elapsed = match.groups()
        row = rows.setdefault(int(request), {})
        row[f"{phase}_epoch"] = int(epoch)
        if rc is not None:
            row["rc"] = int(rc)
            row["elapsed_seconds"] = int(elapsed)
    return [rows[index] | {"request": index} for index in sorted(rows)]


def parse_vmstat(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in read_text(path).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


def main() -> int:
    requests = parse_requests()
    assert len(requests) == 59, len(requests)
    assert all(row.get("rc") == 0 for row in requests)
    assert all(row["end_epoch"] >= row["start_epoch"] for row in requests)

    responses = [http_json(RUN / f"f01_request_{row['request']}.http") for row in requests]
    metrics = [http_json(RUN / f"f01_metrics_{row['request']}.http") for row in requests]
    final_metrics = http_json(RUN / "f01_metrics_final.http")
    final_health = http_json(RUN / "f01_health_final.http")

    assert all(response["status"] == "pass" for response in responses)
    assert all(response["input_tokens"] == 1 for response in responses)
    assert all(response["layers_run"] == 24 for response in responses)
    assert all(response["emitted_token_id"] == 366 for response in responses)
    assert all(metric["inference_requests"] == index for index, metric in enumerate(metrics, 1))
    assert all(metric["completed_inference_requests"] == index for index, metric in enumerate(metrics, 1))
    assert all(metric["failures"] == 0 for metric in metrics)
    assert final_health["status"] == "ok"
    assert final_metrics["inference_requests"] == 59
    assert final_metrics["completed_inference_requests"] == 59
    assert final_metrics["failures"] == 0

    thermal = [json.loads(line) for line in read_text(RUN / "f01_thermal.jsonl").splitlines()]
    assert thermal[0]["label"] == "start"
    assert thermal[-1]["label"] == "finish"
    observed_duration = thermal[-1]["captured_at_epoch"] - thermal[0]["captured_at_epoch"]
    assert observed_duration >= 1800, observed_duration

    request_seconds = [row["elapsed_seconds"] for row in requests]
    decoder_rss = [response["peak_rss_kib"] for response in responses]
    dense_flash = [response["dense_flash_bytes"] for response in responses]
    expert_flash = [response["expert_flash_bytes"] for response in responses]
    cache_hits = [response["cache_hits"] for response in responses]
    cache_misses = [response["cache_misses"] for response in responses]
    prefetch_reads = [response["prefetch_read_ns"] for response in responses]
    service_rss = [sample.get("service_rss_kib", 0) for sample in thermal if sample.get("service_alive")]
    mem_available = [sample["mem_available_kib"] for sample in thermal]
    swap_free = [sample["swap_free_kib"] for sample in thermal]
    frequencies = [value for sample in thermal for value in sample.get("cpu_freq_khz", {}).values()]
    thermal_statuses = [int(value) for sample in thermal for value in re.findall(r"Thermal Status:\s*(\d+)", sample.get("thermal", ""))]
    battery_temperatures = [int(value) for sample in thermal for value in re.findall(r"temperature:\s*(\d+)", sample.get("battery", ""))]

    vmstat = parse_vmstat(RUN / "f01_vmstat_final.txt")
    failure_path = RUN / "f01_failure.log"
    failure_text = read_text(failure_path) if failure_path.exists() else ""
    logcat = read_text(RUN / "f01_logcat_final.txt")
    inference_logcat_matches = [
        line for line in logcat.splitlines()
        if re.search(r"inference_service_a00|minimal_decoder", line, re.IGNORECASE)
        and re.search(r"oom|killed|fatal|crash", line, re.IGNORECASE)
    ]
    assert not failure_text.strip()
    assert vmstat.get("oom_kill", 0) == 0
    assert not inference_logcat_matches
    assert all(status == 0 for status in thermal_statuses)

    total_seconds = sum(request_seconds)
    total_flash = sum(dense_flash) + sum(expert_flash)
    total_cache_lookups = sum(cache_hits) + sum(cache_misses)
    benchmark = {
        "schema_version": 1,
        "task": "F01",
        "status": "pass",
        "device": {
            "serial": "192.168.100.189:5555",
            "model": "Infinix X6528",
            "adb_transport": "wifi_adb",
            "service_pid": 20742,
        },
        "duration": {
            "required_seconds": 1800,
            "configured_seconds": 1860,
            "observed_seconds": observed_duration,
            "request_count": len(requests),
            "telemetry_samples": len(thermal),
            "request_success_count": len(requests),
            "request_failure_count": 0,
        },
        "throughput": {
            "tokens_emitted": len(requests),
            "tokens_per_second_wall": len(requests) / observed_duration,
            "tokens_per_second_request_compute": len(requests) / total_seconds,
            "seconds_per_token_mean": statistics.mean(request_seconds),
            "seconds_per_token_min": min(request_seconds),
            "seconds_per_token_max": max(request_seconds),
        },
        "correctness": {
            "all_http_status_pass": True,
            "all_layers_run_24": True,
            "emitted_token_ids": sorted(set(response["emitted_token_id"] for response in responses)),
            "emitted_token_agreement": len({response["emitted_token_id"] for response in responses}) == 1,
            "emitted_logit_agreement": len({response["emitted_token_logit"] for response in responses}) == 1,
        },
        "decoder": {
            "dense_threads": sorted(set(response["dense_threads"] for response in responses)),
            "io_overlap_enabled": sorted(set(response["io_overlap_enabled"] for response in responses)),
            "lm_head_chunk_rows": sorted(set(response["lm_head_chunk_rows"] for response in responses)),
            "peak_decoder_rss_kib": max(decoder_rss),
            "rss_budget_kib": 630938,
            "rss_headroom_kib": 630938 - max(decoder_rss),
            "flash_bytes_total": total_flash,
            "flash_bytes_per_token": total_flash / len(requests),
            "dense_flash_bytes_per_token": sum(dense_flash) / len(requests),
            "expert_flash_bytes_per_token": sum(expert_flash) / len(requests),
            "cache_hits_total": sum(cache_hits),
            "cache_misses_total": sum(cache_misses),
            "cache_hit_rate": sum(cache_hits) / total_cache_lookups if total_cache_lookups else 0.0,
            "prefetch_read_ns_total": sum(prefetch_reads),
            "prefetch_read_ms_per_token": sum(prefetch_reads) / len(requests) / 1e6,
        },
        "system_telemetry": {
            "service_rss_kib_min": min(service_rss),
            "service_rss_kib_max": max(service_rss),
            "service_hwm_kib_max": max(sample.get("service_hwm_kib", 0) for sample in thermal),
            "mem_available_kib_min": min(mem_available),
            "mem_available_kib_final": mem_available[-1],
            "swap_free_kib_min": min(swap_free),
            "swap_free_kib_final": swap_free[-1],
            "swap_total_kib": thermal[-1]["swap_total_kib"],
            "cpu_frequency_khz_min": min(frequencies),
            "cpu_frequency_khz_max": max(frequencies),
            "thermal_status_max": max(thermal_statuses),
            "battery_temperature_deci_c_min": min(battery_temperatures),
            "battery_temperature_deci_c_max": max(battery_temperatures),
            "oom_kill_counter": vmstat.get("oom_kill", 0),
            "inference_logcat_crash_or_oom_matches": len(inference_logcat_matches),
            "zram_mm_stat": read_text(RUN / "f01_zram_mm_stat_final.txt").strip(),
            "zram_bd_stat": read_text(RUN / "f01_zram_bd_stat_final.txt").strip(),
            "psi": read_text(RUN / "f01_psi_final.txt").strip(),
        },
        "service_metrics_final": final_metrics,
        "raw_artifacts": {
            "thermal": "artifacts/runs/20260827T022000Z_F01/thermal.jsonl",
            "request_log": "artifacts/runs/20260827T022000Z_F01/f01_requests.log",
            "raw_bundle": "artifacts/runs/20260827T022000Z_F01/f01_raw.tar",
            "final_metrics": "artifacts/runs/20260827T022000Z_F01/f01_metrics_final.http",
            "final_health": "artifacts/runs/20260827T022000Z_F01/f01_health_final.http",
        },
        "limitations": [
            "Stock Android shell access denied zram mm_stat/bd_stat and PSI reads; the denied probes are retained verbatim.",
            "Battery temperature is reported by dumpsys battery in deci-degrees Celsius; thermal HAL exposes status but no readable sensor list.",
            "Each request emits one deterministic token, so F01 measures repeated service usability and decode stability rather than a continuous multi-token stream."
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "observed_seconds": observed_duration,
        "requests": len(requests),
        "mean_request_seconds": statistics.mean(request_seconds),
        "peak_decoder_rss_kib": max(decoder_rss),
        "mem_available_min_kib": min(mem_available),
        "swap_free_min_kib": min(swap_free),
        "thermal_status_max": max(thermal_statuses),
        "battery_temperature_deci_c_max": max(battery_temperatures),
        "oom_kill_counter": vmstat.get("oom_kill", 0),
        "output": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
