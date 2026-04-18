from __future__ import annotations

import asyncio
import logging
import inspect
import os
import time
from typing import Any, Awaitable, Callable, Generic, TypeVar

from conveyor.types import _SENTINEL, IStage
from conveyor.metrics import StageMetrics
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T")


class Stage(Generic[T], IStage):
    """A single processing stage in the pipeline.

    Runs *fn* on each item independently, with *workers* concurrent tasks
    pulling from an input queue and pushing to the next stage.
    """

    def __init__(
        self,
        fns: list[Callable[..., Awaitable[Any] | Any]],
        queue_size_per_worker: int,
        stage_name: str | None = None,
    ):
        self._fns: list[Callable[..., Awaitable[Any] | Any]] = fns
        self._in_q: asyncio.Queue = asyncio.Queue(maxsize=len(fns) * queue_size_per_worker)
        self._stage_name: str = stage_name or os.urandom(2).hex()
        self._metrics: StageMetrics | None = None

    @property
    def in_q(self) -> asyncio.Queue:
        return self._in_q

    @property
    def stage_name(self) -> str:
        return self._stage_name

    @property
    def fns(self) -> list[Callable[..., Awaitable[Any] | Any]]:
        return self._fns

    def _run_fn(
        self,
        fn: Callable[..., Awaitable[Any] | Any],
        args: tuple[Any, ...],
        executor: ThreadPoolExecutor | None = None,
    ) -> Awaitable[Any]:
        loop = asyncio.get_event_loop()

        if inspect.iscoroutinefunction(fn):
            return fn(*args)

        assert executor is not None, "Executor is required for non-async functions"
        return loop.run_in_executor(executor, fn, *args)

    async def _worker(
        self,
        next_q: asyncio.Queue | None,
        results: dict[int, asyncio.Future],
        runner_index: int,
        metrics: StageMetrics,
        executor: ThreadPoolExecutor | None = None,
    ):
        fn = self._fns[runner_index]

        logger = logging.getLogger(f"{self._stage_name}:{runner_index}")
        logger.info("Starting worker")

        while True:
            item = await self._in_q.get()

            if item is _SENTINEL:
                self._in_q.task_done()
                break

            req_id, payload = item
            t0 = time.perf_counter()

            try:
                result = await self._run_fn(fn, (payload,), executor=executor)
                elapsed = time.perf_counter() - t0

                logger.info("Finished request %s in %.2fs", req_id, elapsed)

                metrics.record_success(1, elapsed)

                if next_q is not None:
                    await next_q.put((req_id, result))
                elif req_id in results:
                    results[req_id].set_result(result)

            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error("Error processing request %s: %s", req_id, e)

                metrics.record_failure(1, elapsed)

                if req_id in results and not results[req_id].done():
                    results[req_id].set_exception(e)

            finally:
                self._in_q.task_done()
