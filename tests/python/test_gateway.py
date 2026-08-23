from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
from collections.abc import AsyncIterator, Collection, Sequence
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from openai_harmony import Message, RenderOptions, Role

from host.gateway.app import create_app
from host.gateway.device_client import HttpDeviceTokenClient
from host.gateway.harmony_adapter import HarmonyAdapter

FIXTURE = Path("tests/fixtures/harmony_gateway_golden.json")


class FakeDevice:
    def __init__(self, completion: Sequence[int]) -> None:
        self.completion = list(completion)
        self.prompts: list[list[int]] = []

    async def health(self) -> dict:
        return {"status": "ok"}

    async def generate(
        self,
        prompt_tokens: Sequence[int],
        max_output_tokens: int,
        stop_tokens: Collection[int],
    ) -> AsyncIterator[int]:
        self.prompts.append(list(prompt_tokens))
        for token in self.completion[:max_output_tokens]:
            yield token
            if token in stop_tokens:
                return


def completion_tokens(adapter: HarmonyAdapter, message: Message) -> list[int]:
    rendered = adapter.encoding.render(
        message, RenderOptions(conversation_has_function_tools=True)
    )
    assert rendered[:2] == [200006, 173781]
    return rendered[2:]


def test_official_harmony_golden_render() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert importlib.metadata.version("openai-harmony") == fixture["version"]
    adapter = HarmonyAdapter(fixture["conversation_start_date"])
    tokens = adapter.render_request(
        fixture["input"],
        instructions=fixture["instructions"],
        tools=fixture["tools"],
        reasoning_effort=fixture["reasoning_effort"],
    )
    assert tokens == fixture["tokens"]
    assert adapter.encoding.decode(tokens) == fixture["decoded"]
    token_csv = ",".join(str(token) for token in tokens).encode("ascii")
    assert hashlib.sha256(token_csv).hexdigest() == fixture["token_csv_sha256"]


def test_strict_harmony_tool_call_parse() -> None:
    adapter = HarmonyAdapter("2026-08-24")
    message = (
        Message.from_role_and_content(Role.ASSISTANT, '{"city":"Saigon"}')
        .with_channel("analysis")
        .with_recipient("functions.weather")
        .with_content_type("json")
    )
    parsed = adapter.parse_completion(completion_tokens(adapter, message))
    assert len(parsed) == 1
    assert parsed[0].channel == "analysis"
    assert parsed[0].recipient == "functions.weather"
    assert parsed[0].content_type == "json"
    assert adapter.message_text(parsed[0]) == '{"city":"Saigon"}'
    item = adapter.response_items(parsed, "resp_fixture")[0]
    assert item["type"] == "function_call"
    assert item["name"] == "weather"
    assert item["arguments"] == '{"city":"Saigon"}'


def test_function_output_reuses_prior_call_name() -> None:
    adapter = HarmonyAdapter("2026-08-24")
    tokens = adapter.render_request(
        [
            {
                "type": "function_call",
                "call_id": "call_weather",
                "name": "weather",
                "arguments": '{"city":"Saigon"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": '{"temperature_c":31}',
            },
        ]
    )
    rendered = adapter.encoding.decode(tokens)
    assert "<|start|>assistant to=functions.weather<|channel|>analysis json<|message|>" in rendered
    assert "<|start|>functions.weather<|message|>{\"temperature_c\":31}<|end|>" in rendered


def test_http_device_client_appends_tokens_and_stops() -> None:
    bodies: list[str] = []
    emitted = iter([42, 200007, 99])

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append((await request.aread()).decode("ascii"))
        return httpx.Response(200, json={"status": "pass", "emitted_token_id": next(emitted)})

    async def run() -> list[int]:
        client = HttpDeviceTokenClient("http://device", transport=httpx.MockTransport(handler))
        return [token async for token in client.generate([1, 2], 8, {200007})]

    assert asyncio.run(run()) == [42, 200007]
    assert bodies == ["1,2", "1,2,42"]


def test_responses_endpoint_non_stream_and_device_health() -> None:
    adapter = HarmonyAdapter("2026-08-24")
    final = Message.from_role_and_content(Role.ASSISTANT, "Hello from Hot 40i.").with_channel(
        "final"
    )
    fake = FakeDevice(completion_tokens(adapter, final))
    client = TestClient(create_app(device_client=fake, adapter=adapter))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["harmony"]["version"] == "0.0.8"
    assert client.get("/device/health").json() == {"status": "ok"}

    response = client.post(
        "/v1/responses",
        json={"model": "openai/gpt-oss-20b", "input": "Say hi.", "max_output_tokens": 32},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "Hello from Hot 40i."
    assert payload["usage"]["output_tokens"] == len(fake.completion)
    assert fake.prompts[0] == adapter.render_request("Say hi.")


def test_responses_endpoint_streams_official_parser_deltas() -> None:
    adapter = HarmonyAdapter("2026-08-24")
    final = Message.from_role_and_content(Role.ASSISTANT, "Streamed.").with_channel("final")
    fake = FakeDevice(completion_tokens(adapter, final))
    client = TestClient(create_app(device_client=fake, adapter=adapter))

    response = client.post(
        "/v1/responses",
        json={"input": "Stream it.", "stream": True, "max_output_tokens": 32},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    types = [event["type"] for event in events]
    assert types[0] == "response.created"
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"
    deltas = [event["delta"] for event in events if event["type"] == "response.output_text.delta"]
    assert "".join(deltas) == "Streamed."
    assert [event["sequence_number"] for event in events] == list(range(len(events)))


def test_responses_endpoint_returns_function_call() -> None:
    adapter = HarmonyAdapter("2026-08-24")
    call = (
        Message.from_role_and_content(Role.ASSISTANT, '{"city":"Saigon"}')
        .with_channel("analysis")
        .with_recipient("functions.weather")
        .with_content_type("json")
    )
    fake = FakeDevice(completion_tokens(adapter, call))
    client = TestClient(create_app(device_client=fake, adapter=adapter))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    response = client.post(
        "/responses",
        json={
            "input": fixture["input"],
            "tools": fixture["tools"],
            "max_output_tokens": 32,
        },
    )
    assert response.status_code == 200
    item = response.json()["output"][0]
    assert item["type"] == "function_call"
    assert item["name"] == "weather"
    assert json.loads(item["arguments"]) == {"city": "Saigon"}
