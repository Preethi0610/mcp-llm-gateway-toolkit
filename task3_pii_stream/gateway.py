"""Task 3 - LLM gateway endpoint that streams the provider's SSE response through the PII redactor.

POST /v1/chat/completions (OpenAI-compatible chat format, stream forced on)
  -> upstream provider (httpx streaming) -> per-choice StreamRedactor -> client, event by event.

Nothing accumulates: each upstream line is processed and yielded immediately; the only state is one
bounded StreamRedactor per choice index. Upstream connection is closed when the client disconnects.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from redactor import StreamRedactor

log = logging.getLogger("task3.gateway")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


async def redact_sse(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Transform upstream SSE lines into redacted SSE events. Pure async-iterator => easy to test."""
    redactors: dict[int, StreamRedactor] = {}
    template: dict = {}
    async for line in lines:
        if not line.strip():
            continue  # we re-frame events ourselves
        if not line.startswith("data:"):
            yield line + "\n"  # id:/event:/comments pass through
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            # Safety net: flush anything still held for choices that never sent finish_reason.
            for idx, r in redactors.items():
                tail = r.flush()
                if tail:
                    yield _sse({**template, "choices": [{"index": idx, "delta": {"content": tail}, "finish_reason": None}]})
            yield "data: [DONE]\n\n"
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            log.warning("dropping unparseable upstream event")  # never forward unredacted unknown data
            continue
        template = {k: v for k, v in obj.items() if k != "choices"}
        emit = False
        for choice in obj.get("choices", []):
            idx = choice.get("index", 0)
            delta = choice.get("delta") or {}
            red = redactors.setdefault(idx, StreamRedactor())
            content = delta.get("content")
            if isinstance(content, str):
                delta["content"] = red.feed(content)
            if choice.get("finish_reason") is not None:  # end of this choice: release held tail first
                delta["content"] = delta.get("content", "") + red.flush()
            choice["delta"] = delta
            if delta.get("content") != "" or choice.get("finish_reason") is not None or set(delta) - {"content"}:
                emit = True
        if emit or not obj.get("choices"):
            yield _sse(obj)  # fully-held chunks are skipped; their text rides on the next event


def create_app(upstream: httpx.AsyncClient) -> FastAPI:
    app = FastAPI(title="LLM Gateway (PII redaction)")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        body["stream"] = True
        try:
            up = await upstream.send(upstream.build_request("POST", "/v1/chat/completions", json=body), stream=True)
        except httpx.HTTPError as exc:
            log.error("upstream connect failure: %r", exc)
            return JSONResponse({"error": {"type": "upstream_unavailable", "message": "Upstream provider unavailable"}}, status_code=502)
        if up.status_code != 200:
            await up.aclose()
            return JSONResponse({"error": {"type": "upstream_error", "message": "Upstream provider error"}}, status_code=502)

        async def body_iter():
            try:
                async for out in redact_sse(up.aiter_lines()):
                    yield out
            except httpx.HTTPError as exc:
                log.error("upstream stream failure: %r", exc)
                yield _sse({"error": {"type": "upstream_interrupted", "message": "Upstream stream interrupted"}})
            finally:
                await up.aclose()

        return StreamingResponse(
            body_iter(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def build_default_app() -> FastAPI:
    return create_app(httpx.AsyncClient(base_url=os.environ.get("UPSTREAM_URL", "http://127.0.0.1:9002"), timeout=httpx.Timeout(60.0, connect=5.0)))
