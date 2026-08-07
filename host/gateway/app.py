from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Hot40i laptop gateway", version="0.1.0")
DEVICE_URL = os.getenv("HOT40_DEVICE_URL", "http://127.0.0.1:18080")


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=4096)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "device_url": DEVICE_URL}


@app.get("/device/health")
async def device_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{DEVICE_URL}/health")
            response.raise_for_status()
            return response.json()
    except Exception as exc:  # boundary converts transport failures to API error
        raise HTTPException(status_code=502, detail=f"device runtime unavailable: {exc}") from exc


@app.post("/experimental/infer")
async def infer(request: InferRequest) -> dict:
    """Temporary transport contract.

    The final OpenAI/Harmony gateway belongs on the laptop, but this endpoint is
    intentionally generic until the device runtime can generate correct tokens.
    """
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{DEVICE_URL}/infer", json=request.model_dump())
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"device inference failed: {exc}") from exc


def main() -> None:
    import uvicorn

    uvicorn.run("host.gateway.app:app", host="127.0.0.1", port=18081, reload=False)
