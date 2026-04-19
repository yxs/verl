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
import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(ABC):
    """Strategy for picking a server for a new request.

    Sticky-session reuse is handled by ``GlobalRequestLoadBalancer``; ``pick`` is
    invoked only for fresh request ids. The LB shares its in-flight counter
    read-only at registration; ``pick`` must be sync and non-blocking.
    """

    @abstractmethod
    def register_servers(self, server_addresses: list[str], inflight: dict[str, int]) -> None: ...

    @abstractmethod
    def pick(self, request_id: str) -> str: ...

    def on_acquire(self, server_id: str) -> None:  # noqa: B027
        pass

    def on_release(self, server_id: str) -> None:  # noqa: B027
        pass

    async def start(self) -> None:  # noqa: B027
        pass

    async def stop(self) -> None:  # noqa: B027
        pass


class LeastRequestsStrategy(LoadBalanceStrategy):
    """Route new requests to the server with the fewest in-flight requests."""

    def __init__(self) -> None:
        self._inflight: dict[str, int] = {}

    def register_servers(self, server_addresses: list[str], inflight: dict[str, int]) -> None:
        self._inflight = inflight

    def pick(self, request_id: str) -> str:
        return min(self._inflight, key=self._inflight.get)


_VLLM_KV_CACHE_METRIC = "vllm:kv_cache_usage_perc"


class LeastKVCacheStrategy(LoadBalanceStrategy):
    """Route new requests to the vLLM server with the lowest ``vllm:kv_cache_usage_perc``.

    Polls ``/metrics`` every ``poll_interval_s``. Per-server state: ``HEALTHY`` /
    ``STALE`` (past TTL) / ``UNHEALTHY`` (past ``failure_threshold`` consecutive
    failures). ``start`` raises if no server exposes the metric; runtime
    all-``UNHEALTHY`` degrades to inflight-only routing with a WARN.
    """

    HEALTHY = "healthy"
    STALE = "stale"
    UNHEALTHY = "unhealthy"

    def __init__(
        self,
        poll_interval_s: float = 1.0,
        http_timeout_s: float = 0.5,
        failure_threshold: int = 3,
        backoff_initial_s: float = 1.0,
        backoff_max_s: float = 30.0,
        url_scheme: str = "http",
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be positive, got {poll_interval_s}")
        if http_timeout_s <= 0:
            raise ValueError(f"http_timeout_s must be positive, got {http_timeout_s}")
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")

        self._poll_interval_s = poll_interval_s
        self._http_timeout_s = http_timeout_s
        self._failure_threshold = failure_threshold
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._url_scheme = url_scheme

        self._addresses: list[str] = []
        self._inflight: dict[str, int] = {}
        self._kv_usage: dict[str, float] = {}
        self._last_update: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        self._state: dict[str, str] = {}

        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._rng = random.Random()
        self._all_unhealthy_warned = False

    def register_servers(self, server_addresses: list[str], inflight: dict[str, int]) -> None:
        self._addresses = list(server_addresses)
        self._inflight = inflight
        for addr in self._addresses:
            self._kv_usage[addr] = 0.0
            self._last_update[addr] = 0.0
            self._failures[addr] = 0
            self._state[addr] = self.STALE

    async def start(self) -> None:
        if self._poll_task is not None:
            return
        self._client = httpx.AsyncClient(timeout=self._http_timeout_s)
        # Startup sanity: one poll round; raise if nothing exposes the metric.
        await self._poll_once()
        if not any(self._state[a] == self.HEALTHY for a in self._addresses):
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"LeastKVCacheStrategy: no server exposed {_VLLM_KV_CACHE_METRIC} at startup; "
                f"check vLLM version and that rollout.disable_log_stats is False. "
                f"scrape failures={dict(self._failures)}"
            )
        self._poll_task = asyncio.create_task(self._poll_loop(), name="kv_cache_lb_poller")

    async def stop(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _poll_loop(self) -> None:
        backoff = self._backoff_initial_s
        while True:
            try:
                await self._poll_once()
                backoff = self._backoff_initial_s
                await asyncio.sleep(self._poll_interval_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("KV-cache LB poll iteration failed: %s; backing off %.1fs", exc, backoff)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2.0, self._backoff_max_s)

    async def _poll_once(self) -> None:
        await asyncio.gather(*[self._scrape_one(addr) for addr in self._addresses])
        ttl = 2.0 * self._poll_interval_s
        now = time.monotonic()
        for addr in self._addresses:
            if self._state[addr] == self.HEALTHY and (now - self._last_update[addr]) > ttl:
                logger.warning(
                    "KV-cache LB: server %s stale (last update %.2fs ago); demoting to STALE",
                    addr,
                    now - self._last_update[addr],
                )
                self._state[addr] = self.STALE

    async def _scrape_one(self, addr: str) -> None:
        assert self._client is not None
        url = f"{self._url_scheme}://{addr}/metrics"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            usage = self._parse_kv_cache_usage(response.text)
            if usage is None:
                raise ValueError(f"metric {_VLLM_KV_CACHE_METRIC} not found at {url}")
            self._kv_usage[addr] = usage
            self._last_update[addr] = time.monotonic()
            self._failures[addr] = 0
            if self._state[addr] != self.HEALTHY:
                logger.info("KV-cache LB: server %s HEALTHY (kv_cache_usage=%.3f)", addr, usage)
                self._state[addr] = self.HEALTHY
                self._all_unhealthy_warned = False
        except Exception as exc:
            self._failures[addr] += 1
            if self._failures[addr] >= self._failure_threshold and self._state[addr] != self.UNHEALTHY:
                logger.warning(
                    "KV-cache LB: server %s UNHEALTHY after %d consecutive failures: %s",
                    addr,
                    self._failures[addr],
                    exc,
                )
                self._state[addr] = self.UNHEALTHY

    @staticmethod
    def _parse_kv_cache_usage(text: str) -> Optional[float]:
        from prometheus_client.parser import text_string_to_metric_families

        for family in text_string_to_metric_families(text):
            if family.name != _VLLM_KV_CACHE_METRIC:
                continue
            values = [s.value for s in family.samples]
            if not values:
                return None
            return sum(values) / len(values)
        return None

    def pick(self, request_id: str) -> str:
        healthy = [a for a in self._addresses if self._state[a] == self.HEALTHY]
        if healthy:
            min_usage = min(self._kv_usage[a] for a in healthy)
            candidates = [a for a in healthy if self._kv_usage[a] == min_usage]
            if len(candidates) > 1:
                min_inflight = min(self._inflight[a] for a in candidates)
                candidates = [a for a in candidates if self._inflight[a] == min_inflight]
            return self._rng.choice(candidates) if len(candidates) > 1 else candidates[0]

        stale = [a for a in self._addresses if self._state[a] == self.STALE]
        if stale:
            min_inflight = min(self._inflight[a] for a in stale)
            candidates = [a for a in stale if self._inflight[a] == min_inflight]
            return self._rng.choice(candidates) if len(candidates) > 1 else candidates[0]

        # All UNHEALTHY: degrade to inflight-only routing so we don't hard-fail
        # the whole rollout on metric drift. WARN once per transition.
        if not self._all_unhealthy_warned:
            logger.warning(
                "KV-cache LB: all %d servers UNHEALTHY; degrading to inflight-only routing. failures=%s",
                len(self._addresses),
                dict(self._failures),
            )
            self._all_unhealthy_warned = True
        return min(self._inflight, key=self._inflight.get)
