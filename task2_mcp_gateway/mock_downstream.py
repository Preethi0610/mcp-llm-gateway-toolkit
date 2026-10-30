"""Mock downstream MCP server (HTTP JSON-RPC). Counts calls so tests can prove the gateway blocked something."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
STATS = {"tools/list": 0, "tools/call": 0, "other": 0, "called_tools": []}

TOOLS = [
    {"name": "get_status", "description": "Public status", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Reset an API key", "inputSchema": {"type": "object"}},
]


@app.get("/_stats")
async def stats():
    return STATS


@app.post("/_reset")
async def reset():
    STATS.update({"tools/list": 0, "tools/call": 0, "other": 0, "called_tools": []})
    return {}


@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()
    method, rid = body.get("method"), body.get("id")
    if method == "tools/list":
        STATS["tools/list"] += 1
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        STATS["tools/call"] += 1
        name = body["params"]["name"]
        STATS["called_tools"].append(name)
        return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": f"ok:{name}"}], "isError": False}}
    STATS["other"] += 1
    if rid is None:  # notification
        return JSONResponse(status_code=202, content=None)
    return {"jsonrpc": "2.0", "id": rid, "result": {}}
