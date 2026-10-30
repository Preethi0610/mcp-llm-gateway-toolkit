"""Task 2 - MCP security gateway: Bearer auth -> role -> method/tool-level authorization.

Flow per request (POST /mcp):
  1. Authenticate: ``Authorization: Bearer <token>`` -> role (HS256 JWT with a ``role`` claim,
     or a static token table). Missing/invalid -> HTTP 401, nothing forwarded.
  2. Parse JSON-RPC. Bad JSON -> -32700, non-object/batch -> -32600 (batches are rejected on
     purpose: MCP dropped them, and they would let an admin_ call hide inside an array).
  3. ``tools/call`` whose params.name starts with ``admin_`` requires role == admin, else the
     gateway answers -32001 itself and the downstream is never contacted.
  4. Everything else (tools/list, notifications, other methods) is forwarded transparently.
Downstream failures are mapped to a sanitized -32003; no upstream details leak.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

log = logging.getLogger("task2.gateway")

UNAUTHORIZED_TOOL_CALL = -32001
UPSTREAM_ERROR = -32003
AUTH_REQUIRED = -32002
PARSE_ERROR, INVALID_REQUEST, INVALID_PARAMS = -32700, -32600, -32602
ADMIN_PREFIX = "admin_"
HOP_BY_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade",
              "proxy-authorization", "proxy-authenticate", "trailer", "authorization"}


class Authenticator:
    """Token -> role. Static table for simple setups, HS256 JWT (role claim) for real ones."""

    def __init__(self, static_tokens: dict[str, str] | None = None, jwt_secret: str | None = None):
        self.static = static_tokens or {}
        self.secret = jwt_secret.encode() if jwt_secret else None

    @staticmethod
    def _b64d(s: str) -> bytes:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

    def role_for(self, token: str) -> str | None:
        for known, role in self.static.items():  # constant-time compare, no early exit leak
            if hmac.compare_digest(known.encode(), token.encode()):
                return role
        if self.secret and token.count(".") == 2:
            try:
                h, p, s = token.split(".")
                header = json.loads(self._b64d(h))
                if header.get("alg") != "HS256":  # blocks alg=none / confusion
                    return None
                expected = hmac.new(self.secret, f"{h}.{p}".encode(), hashlib.sha256).digest()
                if not hmac.compare_digest(expected, self._b64d(s)):
                    return None
                claims = json.loads(self._b64d(p))
                if "exp" in claims and float(claims["exp"]) < time.time():
                    return None
                role = claims.get("role")
                return role if isinstance(role, str) else None
            except Exception:
                return None
        return None


def make_jwt(secret: str, role: str, ttl: int = 3600) -> str:
    """Helper for tests/demos."""
    enc = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    h = enc(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = enc(json.dumps({"role": role, "exp": time.time() + ttl}).encode())
    sig = enc(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def rpc_error(rid: Any, code: int, message: str, status: int = 200) -> Response:
    body = {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}
    return Response(json.dumps(body), status_code=status, media_type="application/json")


def is_admin_tool(name: str) -> bool:
    # casefold+strip is stricter than the downstream's exact match -> no case/space bypass.
    return name.strip().casefold().startswith(ADMIN_PREFIX)


def create_app(downstream: httpx.AsyncClient, downstream_path: str = "/mcp", auth: Authenticator | None = None) -> FastAPI:
    auth = auth or Authenticator(
        static_tokens=json.loads(os.environ.get("GATEWAY_TOKENS", '{"admin-token":"admin","viewer-token":"viewer"}')),
        jwt_secret=os.environ.get("GATEWAY_JWT_SECRET"),
    )
    app = FastAPI(title="MCP Security Gateway")

    @app.post("/mcp")
    async def gateway(request: Request) -> Response:
        # 1. authenticate
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        role = auth.role_for(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
        if role is None:
            r = rpc_error(None, AUTH_REQUIRED, "Authentication required", status=401)
            r.headers["WWW-Authenticate"] = "Bearer"
            return r

        # 2. parse JSON-RPC
        raw = await request.body()
        try:
            msg = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return rpc_error(None, PARSE_ERROR, "Parse error", status=400)
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return rpc_error(None, INVALID_REQUEST, "Invalid Request", status=400)
        rid, method = msg.get("id"), msg["method"]

        # 3. method-level authorization
        if method == "tools/call":
            params = msg.get("params")
            name = params.get("name") if isinstance(params, dict) else None
            if not isinstance(name, str):
                return rpc_error(rid, INVALID_PARAMS, "params.name must be a string")
            if is_admin_tool(name) and role != "admin":
                log.warning("denied tool=%s role=%s", name, role)
                return rpc_error(rid, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call")

        # 4. forward everything else (tools/list included) transparently
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        try:
            up = await downstream.post(downstream_path, content=raw, headers=fwd_headers)
        except httpx.HTTPError as exc:
            log.error("downstream failure: %r", exc)
            return rpc_error(rid, UPSTREAM_ERROR, "Downstream MCP server unavailable", status=502)
        out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_BY_HOP | {"content-encoding"}}
        return Response(up.content, status_code=up.status_code, headers=out_headers)

    return app


def build_default_app() -> FastAPI:
    url = os.environ.get("DOWNSTREAM_URL", "http://127.0.0.1:9001")
    return create_app(httpx.AsyncClient(base_url=url, timeout=10.0))
