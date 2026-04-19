# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import time

import pytest

from verl.experimental.agent_loop.load_balance import (
    LeastKVCacheStrategy,
    LeastRequestsStrategy,
)


class TestLeastRequestsStrategy:
    def test_pick_returns_least_loaded(self):
        strategy = LeastRequestsStrategy()
        inflight = {"s0": 3, "s1": 1, "s2": 2}
        strategy.register_servers(["s0", "s1", "s2"], inflight)
        assert strategy.pick("req") == "s1"

    def test_tiebreak_insertion_order(self):
        strategy = LeastRequestsStrategy()
        inflight = {"s0": 0, "s1": 0, "s2": 0}
        strategy.register_servers(["s0", "s1", "s2"], inflight)
        assert strategy.pick("req") == "s0"


def _make_kv_strategy(servers, **kwargs):
    inflight = {s: 0 for s in servers}
    strategy = LeastKVCacheStrategy(**kwargs)
    strategy.register_servers(servers, inflight)
    for s in servers:
        strategy._state[s] = LeastKVCacheStrategy.HEALTHY
        strategy._last_update[s] = time.monotonic()
    return strategy, inflight


class TestLeastKVCachePick:
    def test_picks_lowest_kv_usage(self):
        strategy, _ = _make_kv_strategy(["s0", "s1", "s2"])
        strategy._kv_usage["s0"] = 0.3
        strategy._kv_usage["s1"] = 0.1
        strategy._kv_usage["s2"] = 0.5
        assert strategy.pick("req") == "s1"

    def test_tiebreak_by_inflight(self):
        strategy, inflight = _make_kv_strategy(["s0", "s1", "s2"])
        strategy._kv_usage["s0"] = 0.2
        strategy._kv_usage["s1"] = 0.2
        strategy._kv_usage["s2"] = 0.5
        inflight["s0"] = 5
        inflight["s1"] = 1
        assert strategy.pick("req") == "s1"

    def test_unhealthy_excluded_even_if_lowest_usage(self):
        strategy, _ = _make_kv_strategy(["s0", "s1"])
        strategy._kv_usage["s0"] = 0.1
        strategy._kv_usage["s1"] = 0.5
        strategy._state["s0"] = LeastKVCacheStrategy.UNHEALTHY
        assert strategy.pick("req") == "s1"

    def test_falls_back_to_stale_when_no_healthy(self):
        strategy, inflight = _make_kv_strategy(["s0", "s1"])
        strategy._state["s0"] = LeastKVCacheStrategy.STALE
        strategy._state["s1"] = LeastKVCacheStrategy.STALE
        inflight["s0"] = 3
        inflight["s1"] = 1
        assert strategy.pick("req") == "s1"

    def test_all_unhealthy_degrades_to_inflight(self):
        strategy, inflight = _make_kv_strategy(["s0", "s1"])
        strategy._state["s0"] = LeastKVCacheStrategy.UNHEALTHY
        strategy._state["s1"] = LeastKVCacheStrategy.UNHEALTHY
        inflight["s0"] = 5
        inflight["s1"] = 2
        # Runtime all-UNHEALTHY must not raise; fall back to least inflight.
        assert strategy.pick("req") == "s1"


class TestLeastKVCacheParser:
    def test_parses_single_engine(self):
        text = (
            "# HELP vllm:kv_cache_usage_perc KV-cache usage.\n"
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            'vllm:kv_cache_usage_perc{model_name="x",engine="0"} 0.4231\n'
        )
        assert LeastKVCacheStrategy._parse_kv_cache_usage(text) == pytest.approx(0.4231)

    def test_averages_across_engines(self):
        text = (
            "# HELP vllm:kv_cache_usage_perc KV-cache usage.\n"
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            'vllm:kv_cache_usage_perc{model_name="x",engine="0"} 0.2\n'
            'vllm:kv_cache_usage_perc{model_name="x",engine="1"} 0.6\n'
        )
        assert LeastKVCacheStrategy._parse_kv_cache_usage(text) == pytest.approx(0.4)


class _FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttpxClient:
    def __init__(self, responder):
        self._responder = responder
        self.closed = False

    async def get(self, url: str):
        result = self._responder(url)
        if isinstance(result, BaseException):
            raise result
        return result

    async def aclose(self):
        self.closed = True


def _metric_text(usage: float) -> str:
    return (
        "# HELP vllm:kv_cache_usage_perc KV-cache usage.\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        f'vllm:kv_cache_usage_perc{{model_name="x",engine="0"}} {usage}\n'
    )


class TestLeastKVCachePollLoop:
    @pytest.mark.asyncio
    async def test_scrape_marks_healthy_and_records_usage(self):
        strategy = LeastKVCacheStrategy(poll_interval_s=0.05, http_timeout_s=0.05)
        strategy.register_servers(["s0"], {"s0": 0})
        strategy._client = _FakeHttpxClient(lambda url: _FakeResponse(200, _metric_text(0.42)))
        await strategy._scrape_one("s0")
        assert strategy._state["s0"] == LeastKVCacheStrategy.HEALTHY
        assert strategy._kv_usage["s0"] == pytest.approx(0.42)
        assert strategy._failures["s0"] == 0

    @pytest.mark.asyncio
    async def test_scrape_failure_marks_unhealthy_after_threshold(self):
        strategy = LeastKVCacheStrategy(failure_threshold=2, poll_interval_s=0.05, http_timeout_s=0.05)
        strategy.register_servers(["s0"], {"s0": 0})
        strategy._client = _FakeHttpxClient(lambda url: RuntimeError("boom"))
        await strategy._scrape_one("s0")
        assert strategy._state["s0"] == LeastKVCacheStrategy.STALE
        assert strategy._failures["s0"] == 1
        await strategy._scrape_one("s0")
        assert strategy._state["s0"] == LeastKVCacheStrategy.UNHEALTHY
        assert strategy._failures["s0"] == 2

    @pytest.mark.asyncio
    async def test_recovery_after_failures(self):
        strategy = LeastKVCacheStrategy(failure_threshold=2, poll_interval_s=0.05)
        strategy.register_servers(["s0"], {"s0": 0})
        strategy._client = _FakeHttpxClient(lambda url: RuntimeError("boom"))
        await strategy._scrape_one("s0")
        await strategy._scrape_one("s0")
        assert strategy._state["s0"] == LeastKVCacheStrategy.UNHEALTHY
        strategy._client = _FakeHttpxClient(lambda url: _FakeResponse(200, _metric_text(0.1)))
        await strategy._scrape_one("s0")
        assert strategy._state["s0"] == LeastKVCacheStrategy.HEALTHY
        assert strategy._failures["s0"] == 0

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_stop_other_servers(self):
        strategy = LeastKVCacheStrategy(failure_threshold=1, poll_interval_s=0.05)
        strategy.register_servers(["s0", "s1"], {"s0": 0, "s1": 0})

        def responder(url):
            return RuntimeError("boom") if "s0" in url else _FakeResponse(200, _metric_text(0.7))

        strategy._client = _FakeHttpxClient(responder)
        await strategy._poll_once()
        assert strategy._state["s0"] == LeastKVCacheStrategy.UNHEALTHY
        assert strategy._state["s1"] == LeastKVCacheStrategy.HEALTHY
        assert strategy._kv_usage["s1"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_ttl_demotes_to_stale(self):
        strategy = LeastKVCacheStrategy(poll_interval_s=0.01, http_timeout_s=0.01, failure_threshold=100)
        strategy.register_servers(["s0"], {"s0": 0})
        strategy._state["s0"] = LeastKVCacheStrategy.HEALTHY
        strategy._last_update["s0"] = time.monotonic() - 10.0
        strategy._client = _FakeHttpxClient(lambda url: RuntimeError("transient"))
        await strategy._poll_once()
        assert strategy._state["s0"] == LeastKVCacheStrategy.STALE

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, monkeypatch):
        from verl.experimental.agent_loop import load_balance as lb_mod

        calls: list[str] = []

        def responder(url):
            calls.append(url)
            return _FakeResponse(200, _metric_text(0.1))

        def fake_async_client(**kwargs):
            return _FakeHttpxClient(responder)

        monkeypatch.setattr(lb_mod.httpx, "AsyncClient", fake_async_client)

        strategy = LeastKVCacheStrategy(poll_interval_s=0.01, http_timeout_s=0.01)
        strategy.register_servers(["s0"], {"s0": 0})
        await strategy.start()
        assert strategy._state["s0"] == LeastKVCacheStrategy.HEALTHY
        await asyncio.sleep(0.05)
        assert len(calls) >= 2
        await strategy.stop()
        assert strategy._poll_task is None
        assert strategy._client is None

    @pytest.mark.asyncio
    async def test_start_raises_when_metric_missing(self, monkeypatch):
        from verl.experimental.agent_loop import load_balance as lb_mod

        def fake_async_client(**kwargs):
            return _FakeHttpxClient(lambda url: _FakeResponse(200, "# HELP other_metric\nother_metric 1.0\n"))

        monkeypatch.setattr(lb_mod.httpx, "AsyncClient", fake_async_client)

        strategy = LeastKVCacheStrategy(poll_interval_s=0.01, http_timeout_s=0.01)
        strategy.register_servers(["s0"], {"s0": 0})
        with pytest.raises(RuntimeError, match="no server exposed"):
            await strategy.start()


class TestLeastKVCacheConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_picks_with_metric_refresh(self):
        strategy = LeastKVCacheStrategy(poll_interval_s=0.005, http_timeout_s=0.01)
        servers = ["s0", "s1", "s2"]
        strategy.register_servers(servers, {s: 0 for s in servers})
        for s in servers:
            strategy._state[s] = LeastKVCacheStrategy.HEALTHY
            strategy._last_update[s] = time.monotonic()

        async def flip_usages():
            for i in range(50):
                for s in servers:
                    strategy._kv_usage[s] = (i + hash(s)) % 10 / 10.0
                await asyncio.sleep(0.001)

        async def do_pick():
            return strategy.pick("req")

        flipper = asyncio.create_task(flip_usages())
        results = await asyncio.gather(*[do_pick() for _ in range(200)])
        await flipper
        assert all(r in servers for r in results)


class TestRolloutConfigValidation:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            name="vllm",
            mode="async",
            disable_log_stats=True,
            load_balance_strategy="least_requests",
        )
        kwargs.update(overrides)
        return kwargs

    def test_invalid_strategy_raises(self):
        from verl.workers.config import RolloutConfig

        with pytest.raises(ValueError, match="not supported"):
            RolloutConfig(**self._base_kwargs(load_balance_strategy="round_robin"))

    def test_least_kv_cache_requires_log_stats_on(self):
        from verl.workers.config import RolloutConfig

        with pytest.raises(ValueError, match="disable_log_stats"):
            RolloutConfig(**self._base_kwargs(load_balance_strategy="least_kv_cache", disable_log_stats=True))

    def test_least_kv_cache_rejects_sglang(self):
        from verl.workers.config import RolloutConfig

        with pytest.raises(ValueError, match="vllm"):
            RolloutConfig(
                **self._base_kwargs(name="sglang", load_balance_strategy="least_kv_cache", disable_log_stats=False)
            )

    def test_least_kv_cache_with_vllm_and_log_stats_on_is_valid(self):
        from verl.workers.config import RolloutConfig

        cfg = RolloutConfig(**self._base_kwargs(load_balance_strategy="least_kv_cache", disable_log_stats=False))
        assert cfg.load_balance_strategy == "least_kv_cache"


@pytest.fixture(scope="module")
def ray_for_integration():
    import ray

    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestGlobalRequestLoadBalancerStrategyIntegration:
    def test_default_uses_least_requests(self, ray_for_integration):
        import ray

        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        s = ray.get(lb.acquire_server.remote(request_id="r0"))
        assert s == "s0"

    def test_lb_stop_is_idempotent(self, ray_for_integration):
        import ray

        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0"))
        ray.get(lb.stop.remote())
        ray.get(lb.stop.remote())
