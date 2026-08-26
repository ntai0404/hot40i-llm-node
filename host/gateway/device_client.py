from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Collection, Sequence
from contextvars import ContextVar
from typing import Protocol

import httpx


class DeviceTokenSource(Protocol):
    async def health(self) -> dict: ...

    async def metrics(self) -> dict: ...

    def generate(
        self,
        prompt_tokens: Sequence[int],
        max_output_tokens: int,
        stop_tokens: Collection[int],
    ) -> AsyncIterator[int]: ...


class TransportGuard(Protocol):
    def ensure(self) -> object: ...

    def recover(self) -> object: ...


class HttpDeviceTokenClient:
    """Iterate A00's deterministic one-token endpoint into a token stream."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        transport_guard: TransportGuard | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.transport_guard = transport_guard
        self._generation_trace: ContextVar[tuple[dict, ...]] = ContextVar(
            "hot40_generation_trace", default=()
        )

    def _client(self, timeout: float | None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self.transport)

    async def _prepare_transport(self, *, recover: bool = False) -> None:
        if self.transport_guard is None:
            return
        action = self.transport_guard.recover if recover else self.transport_guard.ensure
        await asyncio.to_thread(action)

    async def health(self) -> dict:
        await self._prepare_transport()
        async with self._client(3.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
            except httpx.TransportError:
                await self._prepare_transport(recover=True)
                response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def metrics(self) -> dict:
        await self._prepare_transport()
        async with self._client(3.0) as client:
            try:
                response = await client.get(f"{self.base_url}/metrics")
            except httpx.TransportError:
                await self._prepare_transport(recover=True)
                response = await client.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.json()

    def last_generation_trace(self) -> tuple[dict, ...]:
        return self._generation_trace.get()

    async def generate(
        self,
        prompt_tokens: Sequence[int],
        max_output_tokens: int,
        stop_tokens: Collection[int],
    ) -> AsyncIterator[int]:
        context = list(prompt_tokens)
        stop = set(stop_tokens)
        trace: list[dict] = []
        self._generation_trace.set(())
        await self._prepare_transport()
        async with self._client(None) as client:
            for _ in range(max_output_tokens):
                body = ",".join(str(token) for token in context)
                try:
                    response = await client.post(
                        f"{self.base_url}/infer",
                        content=body.encode("ascii"),
                        headers={"content-type": "text/plain"},
                    )
                except httpx.TransportError:
                    await self._prepare_transport(recover=True)
                    response = await client.post(
                        f"{self.base_url}/infer",
                        content=body.encode("ascii"),
                        headers={"content-type": "text/plain"},
                    )
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") != "pass" or "emitted_token_id" not in payload:
                    raise RuntimeError("device returned an invalid inference payload")
                trace.append(payload)
                self._generation_trace.set(tuple(trace))
                token = int(payload["emitted_token_id"])
                yield token
                context.append(token)
                if token in stop:
                    break
