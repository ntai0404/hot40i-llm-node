from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from openai_harmony import HarmonyEncodingName, load_harmony_encoding

MODEL = "openai/gpt-oss-20b"


def _get_json(url: str) -> dict[str, Any]:
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()


def _trace_sha256(steps: list[dict[str, Any]]) -> str:
    canonical = json.dumps(steps, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _run_scenario(
    client: OpenAI,
    gateway_url: str,
    *,
    name: str,
    input_text: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    transport_before = _get_json(f"{gateway_url}/transport/health")
    metrics_before = _get_json(f"{gateway_url}/device/metrics")
    started = time.perf_counter()
    raw = client.responses.with_raw_response.create(
        model=MODEL,
        input=input_text,
        max_output_tokens=1,
        reasoning={"effort": "low"},
        tools=tools,
    )
    response = raw.parse()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    metrics_after = _get_json(f"{gateway_url}/device/metrics")

    token_header = raw.headers.get("x-hot40-output-token-ids", "")
    token_ids = [int(value) for value in token_header.split(",") if value]
    trace_sha256 = raw.headers.get("x-hot40-device-trace-sha256", "")
    if not trace_sha256:
        raise RuntimeError(f"{name}: gateway response omitted device trace identity")
    trace = _get_json(f"{gateway_url}/device/traces/{trace_sha256}")
    steps = trace.get("steps", [])

    if response.status not in {"completed", "incomplete"}:
        raise RuntimeError(f"{name}: unexpected response status {response.status}")
    if response.usage is None or response.usage.output_tokens != 1:
        raise RuntimeError(f"{name}: response did not account for one output token")
    if len(token_ids) != 1 or len(steps) != 1:
        raise RuntimeError(f"{name}: expected exactly one client and device token")
    if response.usage.input_tokens != steps[0].get("input_tokens"):
        raise RuntimeError(f"{name}: client/device input-token counts do not match")
    if token_ids[0] != int(steps[0].get("emitted_token_id", -1)):
        raise RuntimeError(f"{name}: response token does not match device trace")
    if trace.get("sha256") != trace_sha256 or _trace_sha256(steps) != trace_sha256:
        raise RuntimeError(f"{name}: device trace digest mismatch")
    if metrics_after["inference_requests"] - metrics_before["inference_requests"] != 1:
        raise RuntimeError(f"{name}: service inference counter did not advance once")
    if (
        metrics_after["completed_inference_requests"]
        - metrics_before["completed_inference_requests"]
        != 1
    ):
        raise RuntimeError(f"{name}: service completion counter did not advance once")
    if metrics_after["failures"] != metrics_before["failures"]:
        raise RuntimeError(f"{name}: device service failure counter increased")
    if elapsed_ms < float(steps[0]["elapsed_ms"]):
        raise RuntimeError(f"{name}: device elapsed time exceeds client round trip")

    incomplete_reason = None
    if response.incomplete_details is not None:
        incomplete_reason = response.incomplete_details.reason
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return {
        "name": name,
        "transport": transport_before,
        "request": {
            "input": input_text,
            "model": MODEL,
            "max_output_tokens": 1,
            "reasoning_effort": "low",
            "tool_names": [tool["name"] for tool in tools],
        },
        "response": {
            "id": response.id,
            "status": response.status,
            "incomplete_reason": incomplete_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "output": [item.model_dump(mode="json") for item in response.output],
            "token_ids": token_ids,
            "decoded_token_text": encoding.decode(token_ids),
        },
        "device_trace": trace,
        "service_metrics_before": metrics_before,
        "service_metrics_after": metrics_after,
        "round_trip_ms": elapsed_ms,
        "gateway_overhead_ms": round(elapsed_ms - float(steps[0]["elapsed_ms"]), 3),
        "correlation": {
            "trace_digest_match": True,
            "token_id_match": True,
            "input_token_count_match": True,
            "inference_counter_delta": 1,
            "completion_counter_delta": 1,
            "failure_counter_delta": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the OpenAI Python client through the Hot40i local gateway."
    )
    parser.add_argument("--gateway-url", default="http://127.0.0.1:18081")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gateway_url = args.gateway_url.rstrip("/")
    base_url = f"{gateway_url}/v1"

    health = {
        "gateway": _get_json(f"{gateway_url}/health"),
        "transport": _get_json(f"{gateway_url}/transport/health"),
        "device": _get_json(f"{gateway_url}/device/health"),
    }
    client = OpenAI(
        api_key="local-device",
        base_url=base_url,
        timeout=None,
        max_retries=0,
    )
    document = {
        "schema_version": 1,
        "task_id": "A03",
        "status": "in_progress",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "runner_pid": os.getpid(),
        "active_scenario": None,
        "client": {
            "package": "openai",
            "version": importlib.metadata.version("openai"),
            "base_url": base_url,
            "api_key": "local-device (non-secret placeholder)",
            "timeout": None,
            "max_retries": 0,
        },
        "health": health,
        "scenarios": [],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("task_id") != "A03":
            raise RuntimeError("cannot resume an artifact belonging to another task")
        document["scenarios"] = previous.get("scenarios", [])
        document["recorded_at"] = previous.get("recorded_at", document["recorded_at"])
        document["health"] = health
    _write_document(args.output, document)
    scenario_specs = [
        {"name": "deterministic_simple", "input_text": "OK", "tools": []},
        {
            "name": "function_tool_format",
            "input_text": "Echo.",
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ],
        },
    ]
    completed_names = {item["name"] for item in document["scenarios"]}
    try:
        for spec in scenario_specs:
            if spec["name"] in completed_names:
                print(f"resume skip {spec['name']}", flush=True)
                continue
            document["active_scenario"] = spec["name"]
            _write_document(args.output, document)
            print(f"starting {spec['name']}", flush=True)
            result = _run_scenario(client, gateway_url, **spec)
            document["scenarios"].append(result)
            document["active_scenario"] = None
            _write_document(args.output, document)
            print(
                f"completed {spec['name']}: token={result['response']['token_ids'][0]} "
                f"round_trip_ms={result['round_trip_ms']}",
                flush=True,
            )
    except Exception as exc:
        document["status"] = "failed"
        document["error"] = f"{type(exc).__name__}: {exc}"
        _write_document(args.output, document)
        raise

    document["status"] = "pass"
    document["active_scenario"] = None
    document["summary"] = {
        "round_trips": len(document["scenarios"]),
        "successful_round_trips": len(document["scenarios"]),
        "tool_format_supported": True,
        "all_trace_correlations_passed": True,
    }
    _write_document(args.output, document)
    print(
        f"wrote {args.output}: {len(document['scenarios'])} real round trips passed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
