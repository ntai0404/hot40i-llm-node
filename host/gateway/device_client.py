from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Sequence
from typing import Protocol

import httpx


class DeviceTokenSource(Protocol):
    async def health(self) -> dict: ...

    def generate(
        self,
        prompt_tokens: Sequence[int],
        max_output_tokens: int,
        stop_tokens: Collection[int],
    ) -> AsyncIterator[int]: ...


class HttpDeviceTokenClient:
    """Iterate A00's deterministic one-token endpoint into a token stream."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    def _client(self, timeout: float | None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self.transport)

    async def health(self) -> dict:
        async with self._client(3.0) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def generate(
        self,
        prompt_tokens: Sequence[int],
        max_output_tokens: int,
        stop_tokens: Collection[int],
    ) -> AsyncIterator[int]:
        context = list(prompt_tokens)
        stop = set(stop_tokens)
        async with self._client(None) as client:
            for _ in range(max_output_tokens):
                body = ",".join(str(token) for token in context)
                response = await client.post(
                    f"{self.base_url}/infer",
                    content=body.encode("ascii"),
                    headers={"content-type": "text/plain"},
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") != "pass" or "emitted_token_id" not in payload:
                    raise RuntimeError("device returned an invalid inference payload")
                token = int(payload["emitted_token_id"])
                yield token
                context.append(token)
                if token in stop:
                    break
