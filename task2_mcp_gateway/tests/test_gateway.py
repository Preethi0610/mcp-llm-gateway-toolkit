import httpx
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gateway, mock_downstream
from gateway import Authenticator, create_app, make_jwt

SECRET = "test-secret"


@pytest.fixture
async def env():
    mock_downstream.STATS.update({"tools/list": 0, "tools/call": 0, "other": 0, "called_tools": []})
    down = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_downstream.app), base_url="http://down")
    auth = Authenticator({"admin-token": "admin", "viewer-token": "viewer"}, jwt_secret=SECRET)
    gw = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(down, auth=auth)), base_url="http://gw")
    yield gw, mock_downstream.STATS
    await gw.aclose(); await down.aclose()


def call(name, rid=1): return {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": name, "arguments": {}}}
def H(tok): return {"Authorization": f"Bearer {tok}"}


async def test_tools_list_forwarded_for_viewer(env):
    gw, stats = env
    r = await gw.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=H("viewer-token"))
    assert r.status_code == 200 and len(r.json()["result"]["tools"]) == 2 and stats["tools/list"] == 1


async def test_viewer_blocked_on_admin_tool_and_downstream_untouched(env):
    gw, stats = env
    r = await gw.post("/mcp", json=call("admin_reset_key", rid=42), headers=H("viewer-token"))
    assert r.json() == {"jsonrpc": "2.0", "id": 42, "error": {"code": -32001, "message": "Unauthorized Tool Call"}}
    assert stats["tools/call"] == 0


async def test_admin_allowed_on_admin_tool(env):
    gw, stats = env
    r = await gw.post("/mcp", json=call("admin_reset_key"), headers=H("admin-token"))
    assert r.json()["result"]["content"][0]["text"] == "ok:admin_reset_key" and stats["called_tools"] == ["admin_reset_key"]


async def test_viewer_can_call_normal_tool(env):
    gw, stats = env
    r = await gw.post("/mcp", json=call("get_status"), headers=H("viewer-token"))
    assert "result" in r.json() and stats["called_tools"] == ["get_status"]


@pytest.mark.parametrize("name", ["ADMIN_reset_key", "Admin_x", "  admin_reset_key"])
async def test_case_and_whitespace_bypass_blocked(env, name):
    gw, stats = env
    r = await gw.post("/mcp", json=call(name), headers=H("viewer-token"))
    assert r.json()["error"]["code"] == -32001 and stats["tools/call"] == 0


async def test_batch_cannot_smuggle_admin_call(env):
    gw, stats = env
    r = await gw.post("/mcp", json=[call("admin_reset_key")], headers=H("viewer-token"))
    assert r.status_code == 400 and r.json()["error"]["code"] == -32600 and stats["tools/call"] == 0


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}])
async def test_missing_or_bad_token_is_401(env, headers):
    gw, stats = env
    r = await gw.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=headers)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer" and stats["tools/list"] == 0


async def test_jwt_roles(env):
    gw, stats = env
    assert (await gw.post("/mcp", json=call("admin_reset_key"), headers=H(make_jwt(SECRET, "admin")))).json().get("result")
    assert (await gw.post("/mcp", json=call("admin_reset_key"), headers=H(make_jwt(SECRET, "viewer")))).json()["error"]["code"] == -32001
    assert (await gw.post("/mcp", json=call("get_status"), headers=H(make_jwt("wrong", "admin")))).status_code == 401
    assert (await gw.post("/mcp", json=call("get_status"), headers=H(make_jwt(SECRET, "admin", ttl=-10)))).status_code == 401


async def test_alg_none_jwt_rejected(env):
    import base64, json
    gw, _ = env
    e = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    forged = f"{e({'alg':'none'})}.{e({'role':'admin'})}."
    assert (await gw.post("/mcp", json=call("admin_reset_key"), headers=H(forged))).status_code == 401


async def test_bad_json_and_bad_params(env):
    gw, _ = env
    r = await gw.post("/mcp", content=b"{not json", headers=H("admin-token"))
    assert r.status_code == 400 and r.json()["error"]["code"] == -32700
    r = await gw.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": 5}}, headers=H("admin-token"))
    assert r.json()["error"]["code"] == -32602 and r.json()["id"] == 3


async def test_notification_forwarded(env):
    gw, stats = env
    r = await gw.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=H("viewer-token"))
    assert r.status_code == 202 and stats["other"] == 1


async def test_downstream_down_is_sanitized():
    class Boom(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request): raise httpx.ConnectError("secret-host:5432 refused")
    down = httpx.AsyncClient(transport=Boom(), base_url="http://x")
    gw = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(down, auth=Authenticator({"t": "admin"}))), base_url="http://gw")
    r = await gw.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=H("t"))
    assert r.status_code == 502 and r.json()["error"]["code"] == -32003 and "secret-host" not in r.text
