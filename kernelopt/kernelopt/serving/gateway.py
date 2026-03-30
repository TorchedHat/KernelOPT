"""FastAPI gateway: translates OpenAI /v1/chat/completions to TorchServe predictions API."""

from __future__ import annotations

import json
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="kernelopt gateway")

# Model name → TorchServe prediction endpoint
MODEL_ROUTES: dict[str, str] = {}


def configure_routes(
    ts_inference_url: str = "http://localhost:8085",
    planner_model: str = "Qwen/Qwen3-30B-A3B",
    executor_model: str = "Qwen/Qwen3-Coder-Next-FP8",
):
    """Set up model-to-TorchServe routing."""
    MODEL_ROUTES[planner_model] = f"{ts_inference_url}/predictions/qwen3-30b-a3b"
    MODEL_ROUTES[executor_model] = f"{ts_inference_url}/predictions/qwen3-coder-next-fp8"


@app.on_event("startup")
async def _auto_configure():
    """Auto-configure routes from env vars when started as a subprocess."""
    import os
    ts_url = os.environ.get("KERNELOPT_TS_URL")
    if ts_url and not MODEL_ROUTES:
        configure_routes(
            ts_inference_url=ts_url,
            planner_model=os.environ.get("KERNELOPT_PLANNER_MODEL", "Qwen/Qwen3-30B-A3B"),
            executor_model=os.environ.get("KERNELOPT_EXECUTOR_MODEL", "Qwen/Qwen3-Coder-Next-FP8"),
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "")

    ts_url = MODEL_ROUTES.get(model)
    if not ts_url:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model: {model}. Available: {list(MODEL_ROUTES.keys())}"},
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(ts_url, json=body)

        if resp.status_code != 200:
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": f"TorchServe error: {resp.text[:500]}"},
            )

        ts_result = resp.json()

    # TorchServe returns the handler's postprocess output
    # Parse generated_text from the response
    generated = ""
    if isinstance(ts_result, str):
        parsed = json.loads(ts_result)
        generated = parsed.get("generated_text", "")
    elif isinstance(ts_result, dict):
        generated = ts_result.get("generated_text", "")
    elif isinstance(ts_result, list) and ts_result:
        first = ts_result[0]
        if isinstance(first, str):
            parsed = json.loads(first)
            generated = parsed.get("generated_text", "")
        elif isinstance(first, dict):
            generated = first.get("generated_text", "")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated},
            "finish_reason": "stop",
        }],
        "model": model,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model"} for name in MODEL_ROUTES
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
