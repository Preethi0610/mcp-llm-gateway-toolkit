import asyncio, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import httpx, pytest, uvicorn
import gateway, mock_llm
from gateway import redact_sse


async def _lines(events, log):
    for e in events:
        log.append(("up", e)); yield e; await asyncio.sleep(0)


def ev(content=None, finish=None, role=None):
    d = {}
    if content is not None: d["content"] = content
    if role: d["role"] = role
    return "data: " + json.dumps({"id": "1", "choices": [{"index": 0, "delta": d, "finish_reason": finish}]})


async def collect(events):
    log, out = [], []
    async for o in redact_sse(_lines(events, log)):
        out.append(o); log.append(("down", o))
    return out, log


def joined(out):
    return "".join(json.loads(o[5:])["choices"][0]["delta"].get("content", "") for o in out if o.startswith("data: {"))


async def test_split_email_across_deltas_is_redacted_and_framing_valid():
    out, _ = await collect([ev(role="assistant"), ev("write to jo"), ev("hn@exa"), ev("mple.com now"), ev(finish="stop"), "data: [DONE]"])
    assert joined(out) == "write to [REDACTED] now"
    assert out[-1] == "data: [DONE]\n\n" and all(o.endswith("\n\n") for o in out)


async def test_tail_released_before_finish_event():
    out, _ = await collect([ev("my ssn 123-45-67"), ev("89"), ev(finish="stop"), "data: [DONE]"])
    assert joined(out) == "my ssn [REDACTED]"
    assert json.loads(out[-2][5:])["choices"][0]["finish_reason"] == "stop"


async def test_first_token_forwarded_before_upstream_finishes():
    """TTFT: output for the first words is yielded while the upstream still has events pending."""
    _, log = await collect([ev("Hello "), ev("world "), ev("this is a long answer "), ev(finish="stop"), "data: [DONE]"])
    first_down = next(i for i, (k, _) in enumerate(log) if k == "down")
    last_up = max(i for i, (k, _) in enumerate(log) if k == "up")
    assert first_down < last_up


async def test_unparseable_upstream_event_is_dropped_not_forwarded():
    out, _ = await collect(["data: {not json 123-45-6789", ev("ok "), "data: [DONE]"])
    assert "123-45-6789" not in "".join(out)


# ---------------------------------------------------------------- end to end over real sockets
async def _start(app):
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.01)
    return srv, task, srv.servers[0].sockets[0].getsockname()[1]


@pytest.fixture
async def stack():
    s1, t1, p1 = await _start(mock_llm.app)
    up = httpx.AsyncClient(base_url=f"http://127.0.0.1:{p1}")
    s2, t2, p2 = await _start(gateway.create_app(up))
    yield f"http://127.0.0.1:{p2}"
    for s, t in ((s2, t2), (s1, t1)):
        s.should_exit = True; await t
    await up.aclose()


async def test_end_to_end_redacts_and_streams_incrementally(stack):
    stamps, text = [], ""
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", f"{stack}/v1/chat/completions", json={"x_chunk": 2, "x_delay": 0.01}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            async for line in r.aiter_lines():
                if line.startswith("data: {"):
                    stamps.append(time.perf_counter() - t0)
                    text += json.loads(line[5:])["choices"][0]["delta"].get("content", "")
    assert text == "Sure! Reach Jane at [REDACTED]. Her SSN is [REDACTED] and card [REDACTED]. Anything else?"
    assert stamps[0] < stamps[-1] * 0.25, "first token should arrive long before the stream ends"


async def test_upstream_down_gives_sanitized_502():
    up = httpx.AsyncClient(base_url="http://127.0.0.1:1")
    s, t, p = await _start(gateway.create_app(up))
    async with httpx.AsyncClient() as c:
        r = await c.post(f"http://127.0.0.1:{p}/v1/chat/completions", json={})
    s.should_exit = True; await t; await up.aclose()
    assert r.status_code == 502 and "127.0.0.1" not in r.text
