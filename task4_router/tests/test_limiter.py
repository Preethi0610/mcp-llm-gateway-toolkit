import asyncio, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from limiter import SlidingWindowLimiter


class Clock:
    def __init__(self): self.t = 1_000_000
    def __call__(self): return self.t


@pytest.fixture
def env(tmp_path):
    clk = Clock()
    return SlidingWindowLimiter(str(tmp_path / "rl.sqlite3"), clock_ms=clk), clk, tmp_path / "rl.sqlite3"


async def test_admits_until_limit_then_denies(env):
    lim, _, _ = env
    assert (await lim.reserve("k", 30_000)).allowed
    assert (await lim.reserve("k", 20_000)).allowed          # exactly 50_000
    d = await lim.reserve("k", 1)
    assert not d.allowed and not d.too_large


async def test_window_eviction_and_retry_after(env):
    lim, clk, _ = env
    await lim.reserve("k", 40_000)                      # t0
    clk.t += 30_000
    await lim.reserve("k", 10_000)                      # t0+30s -> full
    clk.t += 29_999                                     # t0+59.999s: nothing expired yet
    d = await lim.reserve("k", 5_000)
    assert not d.allowed and d.retry_after_ms == 1      # first row (40k) expires 1ms from now
    clk.t += 1                                          # t0+60.000s: row at t0 is evicted
    assert (await lim.reserve("k", 5_000)).allowed
    assert await lim.usage("k") == 15_000


async def test_retry_after_waits_for_just_enough_budget(env):
    lim, clk, _ = env
    await lim.reserve("k", 10_000); clk.t += 10_000      # expires at +60s
    await lim.reserve("k", 10_000); clk.t += 10_000      # expires at +70s
    await lim.reserve("k", 30_000)                       # full (50k)
    d = await lim.reserve("k", 15_000)                   # needs 15k freed -> first two rows (20k) -> wait for 2nd row
    assert not d.allowed and d.retry_after_ms == 70_000 - 20_000
    d2 = await lim.reserve("k", 5_000)                   # only needs 5k -> first row suffices
    assert d2.retry_after_ms == 60_000 - 20_000


async def test_tenants_are_isolated(env):
    lim, _, _ = env
    assert (await lim.reserve("a", 50_000)).allowed
    assert (await lim.reserve("b", 50_000)).allowed
    assert not (await lim.reserve("a", 1)).allowed


async def test_too_large_request_flagged_permanent(env):
    lim, _, _ = env
    d = await lim.reserve("k", 50_001)
    assert d.too_large and not d.allowed


async def test_settle_and_refund_adjust_usage(env):
    lim, _, _ = env
    d = await lim.reserve("k", 10_000)
    await lim.settle(d.reservation_id, 2_500)
    assert await lim.usage("k") == 2_500
    d2 = await lim.reserve("k", 7_000)
    await lim.refund(d2.reservation_id)
    assert await lim.usage("k") == 2_500


async def test_concurrent_reservations_never_oversubscribe(env):
    lim, _, _ = env
    results = await asyncio.gather(*[lim.reserve("k", 10_000) for _ in range(40)])
    assert sum(r.allowed for r in results) == 5
    assert await lim.usage("k") == 50_000


async def test_state_is_on_disk_and_shared_between_instances(env):
    lim, clk, path = env
    await lim.reserve("k", 50_000)
    other = SlidingWindowLimiter(str(path), clock_ms=clk)     # e.g. another worker / restart
    assert not (await other.reserve("k", 1)).allowed
    assert path.exists() and path.stat().st_size > 0


async def test_raw_api_key_never_stored(env):
    lim, _, path = env
    await lim.reserve("sk-super-secret-key", 100)
    blob = b"".join(p.read_bytes() for p in path.parent.glob("rl.sqlite3*"))
    assert b"sk-super-secret-key" not in blob


async def test_table_does_not_grow_past_one_window(env):
    lim, clk, path = env
    import sqlite3
    for _ in range(50):
        await lim.reserve("k", 1); clk.t += 5_000
    n = sqlite3.connect(path).execute("SELECT COUNT(*) FROM usage").fetchone()[0]
    assert n <= 13
