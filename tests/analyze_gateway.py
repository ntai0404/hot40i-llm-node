#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from collections.abc import AsyncIterator, Collection, Sequence
from pathlib import Path

from fastapi.testclient import TestClient
from openai_harmony import Message, RenderOptions, Role

from host.gateway.app import create_app
from host.gateway.harmony_adapter import HarmonyAdapter


class FixtureDevice:
    def __init__(self, completion: Sequence[int]) -> None:
        self.completion = list(completion)

    async def health(self) -> dict:
        return {"status": "ok"}

    async def generate(
        self,
        prompt_tokens: Sequence[int],
        max_output_tokens: int,
        stop_tokens: Collection[int],
    ) -> AsyncIterator[int]:
        for token in self.completion[:max_output_tokens]:
            yield token
            if token in stop_tokens:
                return


def completion_tokens(adapter: HarmonyAdapter, message: Message) -> list[int]:
    return adapter.encoding.render(
        message, RenderOptions(conversation_has_function_tools=True)
    )[2:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--gateway-health", type=Path, required=True)
    parser.add_argument("--device-health", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    gateway_health = json.loads(args.gateway_health.read_text(encoding="utf-8"))
    device_health = json.loads(args.device_health.read_text(encoding="utf-8"))
    adapter = HarmonyAdapter(fixture["conversation_start_date"])

    render_ms: list[float] = []
    prompt_tokens: list[int] = []
    for _ in range(25):
        started = time.perf_counter_ns()
        prompt_tokens = adapter.render_request(
            fixture["input"],
            instructions=fixture["instructions"],
            tools=fixture["tools"],
            reasoning_effort=fixture["reasoning_effort"],
        )
        render_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    token_hash = hashlib.sha256(",".join(map(str, prompt_tokens)).encode("ascii")).hexdigest()
    assert prompt_tokens == fixture["tokens"]
    assert token_hash == fixture["token_csv_sha256"]

    final_message = Message.from_role_and_content(Role.ASSISTANT, "Gateway pass.").with_channel(
        "final"
    )
    completion = completion_tokens(adapter, final_message)
    client = TestClient(create_app(device_client=FixtureDevice(completion), adapter=adapter))
    request = {"input": fixture["input"], "max_output_tokens": 32}

    started = time.perf_counter_ns()
    non_stream = client.post("/v1/responses", json=request)
    non_stream_ms = (time.perf_counter_ns() - started) / 1_000_000
    assert non_stream.status_code == 200
    assert non_stream.json()["output"][0]["content"][0]["text"] == "Gateway pass."

    started = time.perf_counter_ns()
    stream = client.post("/v1/responses", json={**request, "stream": True})
    stream_ms = (time.perf_counter_ns() - started) / 1_000_000
    events = [
        json.loads(line.removeprefix("data: "))
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    deltas = [event["delta"] for event in events if event["type"] == "response.output_text.delta"]
    assert "".join(deltas) == "Gateway pass."
    assert events[-1]["type"] == "response.completed"
    assert gateway_health["status"] == "ok"
    assert device_health["status"] == "ok"

    benchmark = {
        "schema_version": 1,
        "task_id": "A01",
        "status": "pass",
        "harmony": {
            "package": "openai-harmony",
            "version": importlib.metadata.version("openai-harmony"),
            "encoding": adapter.encoding.name,
            "upstream_commit": "abd677f7ac962629c808197caa1feb9e3e95d2b0",
            "golden_prompt_tokens": len(prompt_tokens),
            "golden_token_csv_sha256": token_hash,
            "render_ms_median": statistics.median(render_ms),
            "render_ms_max": max(render_ms),
        },
        "responses_api": {
            "non_stream_status": non_stream.status_code,
            "non_stream_fixture_latency_ms": non_stream_ms,
            "stream_status": stream.status_code,
            "stream_fixture_latency_ms": stream_ms,
            "stream_event_count": len(events),
            "stream_event_types": [event["type"] for event in events],
            "streamed_text_match": True,
        },
        "live_restart": {
            "gateway_health": gateway_health,
            "device_health": device_health,
            "device_transport": "adb forward tcp:18080 -> tcp:8080 over USB",
        },
        "limitations": [
            (
                "Gateway latency uses an in-process deterministic device token fixture; official "
                "device decode latency is recorded by A00/P01/P02."
            ),
            (
                "Live A01 verification proxies A00 health but deliberately does not repeat the "
                "multi-hour full Harmony prompt decode."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(benchmark, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
