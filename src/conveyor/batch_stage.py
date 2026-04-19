from __future__ import annotations

import asyncio
import logging
import inspect
import os
import time
from typing import Any, Awaitable, Callable, Generic, TypeVar

from conveyor.types import _SENTINEL, IStage, PipelineRuntime
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T")


class BatchStage(Generic[T], IStage):
    """A stage that collects items into batches before processing.

    Waits up to *timeout_s* or until *max_batch_size* items arrive,
    then calls *fn* with the full batch list at once.
    """

    def __init__(
        self,
        fns: list[Callable[..., Awaitable[Any] | Any]],
        worker_queue_size: int,
        max_batch_size: int,
        timeout_s: float,
        stage_name: str | None = None,
    ):
        self._fns: list[Callable[..., Awaitable[Any] | Any]] = fns
        self._in_q: asyncio.Queue = asyncio.Queue(maxsize=len(fns) * worker_queue_size)
        self._stage_name: str = stage_name or os.urandom(2).hex()
        self._max_batch_size = max_batch_size
        self._timeout_s = timeout_s

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
        runtime: PipelineRuntime,
        runner_index: int,
    ):
        fn = self._fns[runner_index]
        logger = logging.getLogger(f"batch:{self._stage_name}:{runner_index}")
        logger.info("Starting worker")

        while True:
            batch: list[tuple[int, Any]] = []

            first = await self._in_q.get()
            self._in_q.task_done()

            if first is _SENTINEL:
                break

            batch.append(first)
            deadline = asyncio.get_event_loop().time() + self._timeout_s

            while len(batch) < self._max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._in_q.get(), timeout=remaining)
                    self._in_q.task_done()
                    if item is _SENTINEL:
                        break
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            req_ids = [rid for rid, _ in batch]
            payloads = [p for _, p in batch]
            batch_len = len(payloads)

            t0 = time.perf_counter()
            try:
                outputs = await self._run_fn(fn, (payloads,), executor=runtime.pool)
                elapsed = time.perf_counter() - t0

                logger.info("Batch of %d finished in %.2fs", batch_len, elapsed)
                runtime.metrics.record_success(batch_len, elapsed)

                for rid, out in zip(req_ids, outputs):
                    if next_q is not None:
                        await next_q.put((rid, out))
                    elif rid in runtime.futures:
                        runtime.futures[rid].set_result(out)

            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error("Batch error: %s", e)

                runtime.metrics.record_failure(batch_len, elapsed)

                for rid in req_ids:
                    if rid in runtime.futures and not runtime.futures[rid].done():
                        runtime.futures[rid].set_exception(e)
