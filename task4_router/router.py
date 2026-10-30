"""Task 4 - resilient model router: rate limit -> primary (3s budget) -> fallback, sanitized errors.

Failover triggers on the primary: HTTP 429, a 3000 ms timeout, and (extra, sensible) 5xx / transport
errors. A non-retryable 4xx from the provider is NOT failed over (the fallback would reject it too).
Whatever happens, callers only ever see a GatewayError with a fixed, safe payload; upstream bodies,
exception text and stack traces go to the log under the request id, never to the client.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Protocol

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from limiter import SlidingWindowLimiter

log = logging.getLogger("task4.router")
PRIMARY_TIMEOUT_S = 3.0


# ------------------------------------------------------------------ provider layer
class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200_000)
    max_tokens: int = Field(default=256, ge=1, le=32_768)


@dataclass
class ProviderResult:
    text: str
    total_tokens: int | None  # provider-reported usage, if any
    model: str


class ProviderRateLimited(Exception): ...
class ProviderUnavailable(Exception): ...     # 5xx / network
class ProviderRejected(Exception): ...        # non-retryable 4xx


class Provider(Protocol):
    name: str
    async def complete(self, req: CompletionRequest) -> ProviderResult: ...


class HttpProvider:
    """POST {url} {"prompt","max_tokens"} -> {"text", "model", "usage": {"total_tokens"}}"""

    def __init__(self, name: str, client: httpx.AsyncClient, path: str = "/v1/complete") -> None:
        self.name, self.client, self.path = name, client, path

    async def complete(self, req: CompletionRequest) -> ProviderResult:
        try:
            r = await self.client.post(self.path, json=req.model_dump())
        except httpx.TimeoutException as exc:
            raise asyncio.TimeoutError() from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(type(exc).__name__) from exc
        if r.status_code == 429:
            raise ProviderRateLimited()
        if r.status_code >= 500:
            raise ProviderUnavailable(str(r.status_code))
        if r.status_code != 200:
            raise ProviderRejected(str(r.status_code))
        try:
            d = r.json()
            return ProviderResult(str(d["text"]), (d.get("usage") or {}).get("total_tokens"), str(d.get("model", self.name)))
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderUnavailable("malformed response") from exc


# ------------------------------------------------------------------ standardized errors
class GatewayError(Exception):
    def __init__(self, status: int, etype: str, message: str, retry_after_s: int | None = None):
        self.status, self.etype, self.message, self.retry_after_s = status, etype, message, retry_after_s

    def response(self, request_id: str) -> JSONResponse:
        headers = {"X-Gateway-Request-Id": request_id}
        if self.retry_after_s is not None:
            headers["Retry-After"] = str(self.retry_after_s)
        return JSONResponse({"error": {"type": self.etype, "message": self.message, "request_id": request_id}},
                            status_code=self.status, headers=headers)


# ------------------------------------------------------------------ router
def estimate_tokens(req: CompletionRequest) -> int:
    return math.ceil(len(req.prompt) / 4) + req.max_tokens  # ~4 chars/token + worst-case output


class ModelRouter:
    def __init__(self, primary: Provider, fallback: Provider, limiter: SlidingWindowLimiter, timeout_s: float = PRIMARY_TIMEOUT_S):
        self.primary, self.fallback, self.limiter, self.timeout_s = primary, fallback, limiter, timeout_s

    async def complete(self, api_key: str, req: CompletionRequest, request_id: str) -> dict:
        est = estimate_tokens(req)
        d = await self.limiter.reserve(api_key, est)
        if d.too_large:
            raise GatewayError(413, "request_too_large", "Request exceeds the per-minute token budget")
        if not d.allowed:
            raise GatewayError(429, "rate_limit_exceeded", "Token rate limit exceeded", math.ceil(d.retry_after_ms / 1000))
        assert d.reservation_id is not None
        try:
            result, served_by = await self._with_failover(req, request_id)
        except BaseException:
            await asyncio.shield(self.limiter.refund(d.reservation_id))  # no completion -> no charge (even if cancelled)
            raise
        await self.limiter.settle(d.reservation_id, result.total_tokens if result.total_tokens is not None else est)
        return {"text": result.text, "model": result.model, "served_by": served_by, "request_id": request_id}

    async def _with_failover(self, req: CompletionRequest, request_id: str) -> tuple[ProviderResult, str]:
        try:
            async with asyncio.timeout(self.timeout_s):  # cancels the in-flight call; a late reply is discarded
                return await self.primary.complete(req), "primary"
        except TimeoutError:
            log.warning("[%s] primary timed out after %.1fs; failing over", request_id, self.timeout_s)
        except ProviderRateLimited:
            log.warning("[%s] primary returned 429; failing over", request_id)
        except ProviderUnavailable as exc:
            log.warning("[%s] primary unavailable (%s); failing over", request_id, exc)
        except ProviderRejected as exc:
            log.error("[%s] primary rejected request (%s)", request_id, exc)
            raise GatewayError(502, "upstream_rejected", "The upstream provider could not process this request") from None
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self.fallback.complete(req), "fallback"
        except TimeoutError:
            log.error("[%s] fallback timed out", request_id)
        except (ProviderRateLimited, ProviderUnavailable, ProviderRejected) as exc:
            log.error("[%s] fallback failed (%s: %s)", request_id, type(exc).__name__, exc)
        raise GatewayError(503, "service_unavailable", "All model providers are currently unavailable", retry_after_s=5)


# ------------------------------------------------------------------ HTTP app
def create_app(router: ModelRouter, valid_keys: set[str] | None = None) -> FastAPI:
    """valid_keys=None accepts any non-empty bearer key (dev mode). Pass a set in production."""
    app = FastAPI(title="LLM Gateway Router")

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError):
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})  # names only, no input echo
        rid = uuid.uuid4().hex
        return GatewayError(422, "invalid_request", f"Invalid request fields: {', '.join(fields)}").response(rid)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = uuid.uuid4().hex
        log.exception("[%s] unhandled error", rid)
        return GatewayError(500, "internal_error", "Internal gateway error").response(rid)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, authorization: str | None = Header(default=None)):
        rid = uuid.uuid4().hex
        scheme, _, key = (authorization or "").partition(" ")
        key = key.strip()
        if scheme.lower() != "bearer" or not key or (valid_keys is not None and key not in valid_keys):
            return GatewayError(401, "unauthorized", "Invalid or missing API key").response(rid)
        try:
            out = await router.complete(key, req, rid)
        except GatewayError as e:
            return e.response(rid)
        except Exception:
            # Detail only in the log, keyed by request id. Handled here (not just by the app-level
            # handler) because Starlette re-raises after its catch-all handler responds.
            log.exception("[%s] unexpected error", rid)
            return GatewayError(500, "internal_error", "Internal gateway error").response(rid)
        return JSONResponse(out, headers={"X-Gateway-Request-Id": rid})

    return app


def build_default_app() -> FastAPI:
    mk = lambda env, default: httpx.AsyncClient(base_url=os.environ.get(env, default), timeout=httpx.Timeout(10.0, connect=2.0))
    router = ModelRouter(
        HttpProvider("primary", mk("PRIMARY_URL", "http://127.0.0.1:9101")),
        HttpProvider("fallback", mk("FALLBACK_URL", "http://127.0.0.1:9102")),
        SlidingWindowLimiter(os.environ.get("RATE_DB", "ratelimit.sqlite3")),
    )
    keys = os.environ.get("TENANT_KEYS")
    return create_app(router, set(keys.split(",")) if keys else None)
