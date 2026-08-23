from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from importlib.metadata import version
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai_harmony import Role, StreamableParser
from pydantic import BaseModel, ConfigDict, Field

from host.gateway.device_client import DeviceTokenSource, HttpDeviceTokenClient
from host.gateway.harmony_adapter import HarmonyAdapter
from host.transport.adb_forward import AdbForwardSupervisor, TransportUnavailable

DEVICE_URL = os.getenv("HOT40_DEVICE_URL", "http://127.0.0.1:18080")
LOCAL_MODEL = "openai/gpt-oss-20b"


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = LOCAL_MODEL
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int = Field(default=32, ge=1, le=4096)
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


def _response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _response_payload(
    request: ResponsesRequest,
    response_id: str,
    output: list[dict[str, Any]],
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "model": request.model,
        "output": output,
        "parallel_tool_calls": True,
        "tools": request.tools,
        "metadata": request.metadata,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


async def _non_stream_response(
    request: ResponsesRequest,
    prompt_tokens: list[int],
    response_id: str,
    adapter: HarmonyAdapter,
    device: DeviceTokenSource,
) -> dict[str, Any]:
    completion: list[int] = []
    async for token in device.generate(prompt_tokens, request.max_output_tokens, adapter.stop_tokens):
        completion.append(token)
        if token in adapter.stop_tokens:
            break
    messages = adapter.parse_completion(completion)
    output = adapter.response_items(messages, response_id)
    return _response_payload(request, response_id, output, len(prompt_tokens), len(completion))


async def _stream_response(
    request: ResponsesRequest,
    prompt_tokens: list[int],
    response_id: str,
    adapter: HarmonyAdapter,
    device: DeviceTokenSource,
) -> AsyncIterator[str]:
    sequence = 0
    in_progress = _response_payload(request, response_id, [], len(prompt_tokens), 0)
    in_progress["status"] = "in_progress"
    in_progress["usage"] = None
    yield _sse({"type": "response.created", "sequence_number": sequence, "response": in_progress})
    sequence += 1
    yield _sse(
        {"type": "response.in_progress", "sequence_number": sequence, "response": in_progress}
    )
    sequence += 1

    parser = StreamableParser(adapter.encoding, Role.ASSISTANT, strict=True)
    completion: list[int] = []
    streamed_text = ""
    message_started = False
    message_id = f"msg_{response_id.removeprefix('resp_')}_0"
    try:
        async for token in device.generate(prompt_tokens, request.max_output_tokens, adapter.stop_tokens):
            completion.append(token)
            parser.process(token)
            if parser.current_channel == "final" and not parser.current_recipient:
                current_text = parser.current_content
                delta = current_text[len(streamed_text) :] if current_text.startswith(streamed_text) else ""
                streamed_text = current_text
                if delta:
                    if not message_started:
                        item = {
                            "id": message_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        }
                        yield _sse(
                            {
                                "type": "response.output_item.added",
                                "sequence_number": sequence,
                                "output_index": 0,
                                "item": item,
                            }
                        )
                        sequence += 1
                        yield _sse(
                            {
                                "type": "response.content_part.added",
                                "sequence_number": sequence,
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []},
                            }
                        )
                        sequence += 1
                        message_started = True
                    yield _sse(
                        {
                            "type": "response.output_text.delta",
                            "sequence_number": sequence,
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": delta,
                            "logprobs": [],
                        }
                    )
                    sequence += 1
            if token in adapter.stop_tokens:
                break
        parser.process_eos()
        messages = parser.messages
        output = adapter.response_items(messages, response_id)
        if message_started:
            yield _sse(
                {
                    "type": "response.output_text.done",
                    "sequence_number": sequence,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": streamed_text,
                    "logprobs": [],
                }
            )
            sequence += 1
            yield _sse(
                {
                    "type": "response.content_part.done",
                    "sequence_number": sequence,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": streamed_text,
                        "annotations": [],
                    },
                }
            )
            sequence += 1
        for index, item in enumerate(output):
            if message_started and index == 0 and item["type"] == "message":
                yield _sse(
                    {
                        "type": "response.output_item.done",
                        "sequence_number": sequence,
                        "output_index": index,
                        "item": item,
                    }
                )
            else:
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "sequence_number": sequence,
                        "output_index": index,
                        "item": item,
                    }
                )
                sequence += 1
                yield _sse(
                    {
                        "type": "response.output_item.done",
                        "sequence_number": sequence,
                        "output_index": index,
                        "item": item,
                    }
                )
            sequence += 1
        completed = _response_payload(
            request, response_id, output, len(prompt_tokens), len(completion)
        )
        yield _sse(
            {"type": "response.completed", "sequence_number": sequence, "response": completed}
        )
    except Exception as exc:
        failed = _response_payload(request, response_id, [], len(prompt_tokens), len(completion))
        failed["status"] = "failed"
        failed["error"] = {"code": "device_error", "message": str(exc)}
        yield _sse({"type": "response.failed", "sequence_number": sequence, "response": failed})


def create_app(
    *,
    device_client: DeviceTokenSource | None = None,
    adapter: HarmonyAdapter | None = None,
) -> FastAPI:
    app = FastAPI(title="Hot40i Harmony gateway", version="0.2.0")
    supervisor = None
    if device_client is None:
        manage_forward = os.getenv("HOT40_MANAGE_ADB_FORWARD", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if manage_forward and DEVICE_URL == "http://127.0.0.1:18080":
            supervisor = AdbForwardSupervisor(
                expected_serial=os.getenv("HOT40_DEVICE_SERIAL"),
                host_port=18080,
                device_port=8080,
            )
        device_client = HttpDeviceTokenClient(DEVICE_URL, transport_guard=supervisor)
    app.state.device_client = device_client
    app.state.transport_supervisor = supervisor
    app.state.harmony = adapter or HarmonyAdapter()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "device_url": DEVICE_URL,
            "model": LOCAL_MODEL,
            "transport_managed": app.state.transport_supervisor is not None,
            "harmony": {"package": "openai-harmony", "version": version("openai-harmony")},
        }

    @app.get("/transport/health")
    async def transport_health():
        if app.state.transport_supervisor is None:
            return {"status": "unmanaged", "device_url": DEVICE_URL}
        try:
            status = await asyncio.to_thread(app.state.transport_supervisor.inspect)
            return {"status": "ok", **status.to_dict()}
        except TransportUnavailable as exc:
            return JSONResponse(status_code=503, content=exc.to_dict())

    @app.get("/device/health")
    async def device_health() -> dict:
        try:
            return await app.state.device_client.health()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"device runtime unavailable: {exc}") from exc

    async def create_response(request: ResponsesRequest):
        try:
            effort = request.reasoning.get("effort") if request.reasoning else None
            prompt_tokens = app.state.harmony.render_request(
                request.input,
                instructions=request.instructions,
                tools=request.tools,
                reasoning_effort=effort,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response_id = _response_id()
        if request.stream:
            return StreamingResponse(
                _stream_response(
                    request,
                    prompt_tokens,
                    response_id,
                    app.state.harmony,
                    app.state.device_client,
                ),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )
        try:
            return await _non_stream_response(
                request,
                prompt_tokens,
                response_id,
                app.state.harmony,
                app.state.device_client,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"device inference failed: {exc}") from exc

    app.post("/v1/responses")(create_response)
    app.post("/responses", include_in_schema=False)(create_response)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("host.gateway.app:app", host="127.0.0.1", port=18081, reload=False)
