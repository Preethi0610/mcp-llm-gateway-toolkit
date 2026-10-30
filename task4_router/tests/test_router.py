import asyncio, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import httpx, pytest
from limiter import SlidingWindowLimiter
from router import (CompletionRequest, ModelRouter, ProviderRateLimited, ProviderRejected, ProviderResult,
                    ProviderUnavailable, create_app, PRIMARY_TIMEOUT_S)


class Fake:
    def __init__(self, name, behavior="ok", tokens=100, delay=0.0):
        self.name, self.behavior, self.tokens, self.delay = name, behavior, tokens, delay
        self.calls, self.cancelled, self.finished = 0, False, False

    async def complete(self, req):
        self.calls += 1
        try:
            if self.delay: await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True; raise
        self.finished = True
        if self.behavior == "429": raise ProviderRateLimited()
        if self.behavior == "5xx": raise ProviderUnavailable("503 secret-internal-host")
        if self.behavior == "4xx": raise ProviderRejected("400 bad")
        if self.behavior == "boom": raise RuntimeError("Traceback: /srv/internal/secret.py line 42 password=hunter2")
        return ProviderResult(f"hello from {self.name}", self.tokens, self.name)


def make(primary, fallback, tmp_path, timeout=0.2, limit=50_000):
    lim = SlidingWindowLimiter(str(tmp_path / "r.db"), limit=limit)
    router = ModelRouter(primary, fallback, lim, timeout_s=timeout)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(router)), base_url="http://gw")
    return client, lim


H = {"Authorization": "Bearer tenant-1"}
BODY = {"prompt": "hi there", "max_tokens": 100}


def test_default_timeout_is_3000ms():
    assert PRIMARY_TIMEOUT_S == 3.0


async def test_primary_success_no_fallback_call(tmp_path):
    p, f = Fake("primary"), Fake("fallback")
    c, _ = make(p, f, tmp_path)
    r = await c.post("/v1/completions", json=BODY, headers=H)
    assert r.status_code == 200 and r.json()["served_by"] == "primary" and f.calls == 0


@pytest.mark.parametrize("mode", ["429", "5xx"])
async def test_failover_on_429_and_5xx(tmp_path, mode):
    p, f = Fake("primary", mode), Fake("fallback")
    c, _ = make(p, f, tmp_path)
    r = await c.post("/v1/completions", json=BODY, headers=H)
    assert r.status_code == 200 and r.json()["served_by"] == "fallback" and "secret" not in r.text


async def test_failover_on_timeout_cancels_primary_and_discards_late_reply(tmp_path):
    p, f = Fake("primary", delay=5.0), Fake("fallback")
    c, _ = make(p, f, tmp_path, timeout=0.2)
    t0 = time.perf_counter()
    r = await c.post("/v1/completions", json=BODY, headers=H)
    assert r.json()["served_by"] == "fallback" and time.perf_counter() - t0 < 1.0
    assert p.cancelled and not p.finished


async def test_real_3s_budget_is_respected(tmp_path):
    p, f = Fake("primary", delay=10.0), Fake("fallback")
    c, _ = make(p, f, tmp_path, timeout=PRIMARY_TIMEOUT_S)
    t0 = time.perf_counter()
    r = await c.post("/v1/completions", json=BODY, headers=H)
    dt = time.perf_counter() - t0
    assert r.json()["served_by"] == "fallback" and 3.0 <= dt < 3.5


async def test_both_fail_gives_standard_503_without_leaks(tmp_path):
    c, _ = make(Fake("p", "5xx"), Fake("f", "429"), tmp_path)
    r = await c.post("/v1/completions", json=BODY, headers=H)
    e = r.json()["error"]
    assert r.status_code == 503 and set(e) == {"type", "message", "request_id"} and e["type"] == "service_unavailable"
    assert "secret" not in r.text and r.headers["retry-after"] == "5"


async def test_non_retryable_4xx_is_not_failed_over(tmp_path):
    p, f = Fake("p", "4xx"), Fake("f")
    c, _ = make(p, f, tmp_path)
    r = await c.post("/v1/completions", json=BODY, headers=H)
    assert r.status_code == 502 and f.calls == 0 and r.json()["error"]["type"] == "upstream_rejected"


async def test_unexpected_exception_never_leaks_traceback(tmp_path):
    c, _ = make(Fake("p", "boom"), Fake("f"), tmp_path)
    r = await c.post("/v1/completions", json=BODY, headers=H)
    assert r.status_code == 500 and r.json()["error"]["type"] == "internal_error"
    assert not any(s in r.text for s in ("Traceback", "hunter2", "secret.py", "/srv"))


async def test_rate_limit_returns_429_with_retry_after_and_skips_providers(tmp_path):
    p, f = Fake("p"), Fake("f")
    c, lim = make(p, f, tmp_path)
    await lim.reserve("tenant-1", 49_950)
    r = await c.post("/v1/completions", json=BODY, headers=H)   # estimate ~102 tokens
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1 and p.calls == 0
    assert r.json()["error"]["type"] == "rate_limit_exceeded"


async def test_usage_settled_to_actual_and_refunded_on_failure(tmp_path):
    c, lim = make(Fake("p", tokens=37), Fake("f"), tmp_path)
    await c.post("/v1/completions", json=BODY, headers=H)
    assert await lim.usage("tenant-1") == 37
    c2, lim2 = make(Fake("p", "5xx"), Fake("f", "429"), tmp_path)
    await c2.post("/v1/completions", json=BODY, headers={"Authorization": "Bearer tenant-2"})
    assert await lim2.usage("tenant-2") == 0


async def test_concurrent_requests_respect_budget(tmp_path):
    c, lim = make(Fake("p", delay=0.05, tokens=10_000), Fake("f"), tmp_path)
    body = {"prompt": "x" * 40, "max_tokens": 9_990}         # estimate 10_000
    rs = await asyncio.gather(*[c.post("/v1/completions", json=body, headers=H) for _ in range(12)])
    assert sorted(r.status_code for r in rs).count(200) == 5 and sum(r.status_code == 429 for r in rs) == 7


async def test_auth_and_validation_use_standard_payload(tmp_path):
    c, _ = make(Fake("p"), Fake("f"), tmp_path)
    r = await c.post("/v1/completions", json=BODY)
    assert r.status_code == 401 and r.json()["error"]["type"] == "unauthorized"
    r = await c.post("/v1/completions", json={"prompt": "", "max_tokens": -1, "secret": "x"}, headers=H)
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request" and "input" not in r.text
    r = await c.post("/v1/completions", json={"prompt": "x", "max_tokens": 32_768}, headers=H)  # est > 50k? no: 32_769 ok
    assert r.status_code == 200
    r = await c.post("/v1/completions", json={"prompt": "x" * 100_000, "max_tokens": 32_768}, headers=H)
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"
