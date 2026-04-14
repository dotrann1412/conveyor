from __future__ import annotations

import asyncio
import logging
import inspect
import time
from typing import Any, Awaitable, Callable, Generic, TypeVar

from conveyor.types import _SENTINEL
from conveyor.config import BatchConfig, StageConfig
from conveyor.progress import ProgressReporter, accepts_progress
from conveyor.tracker import SimpleTracker
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T")

class BatchStage(Generic[T]):
    """A stage that collects items into batches before processing.

    Waits up to *timeout_s* or until *max_batch_size* items arrive,
    then calls *fn* with the full batch list at once.
    """

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any] | Any],
        config: BatchConfig | None = None,
        stage_config: StageConfig | None = None,
    ):
        self.batch_config = config or BatchConfig()
        self.config = stage_config or StageConfig(workers=1)
        self._fns: list[Callable[..., Awaitable[Any] | Any]] = [fn] * self.config.workers
        self._in_q: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._tracker = SimpleTracker(
            identifier=self.config.stage_name,
            max_records=self.config.max_log_records,
            in_progress=[ProgressReporter() for _ in range(self.config.workers)],
            batch_configs=[self.batch_config for _ in range(self.config.workers)],
        )

    @property
    def fns(self) -> list[Callable[..., Awaitable[Any] | Any]]:
        return self._fns

    @classmethod
    def from_factory(
        cls,
        fn_factory: Callable[[int], Callable[..., Awaitable[Any] | Any]],
        device_ids: list[int],
        batch_config: BatchConfig | None = None,
        stage_config: StageConfig | None = None,
    ) -> BatchStage[T]:
        stage_config = stage_config or StageConfig()
        stage_config.device_ids = device_ids
        stage_config.workers = len(device_ids)
        batch_config = batch_config or BatchConfig()
        instance = cls.__new__(cls)
        instance.batch_config = batch_config
        instance.config = stage_config
        fns = [fn_factory(did) for did in device_ids]
        instance._fns = fns
        instance._in_q = asyncio.Queue(maxsize=stage_config.max_queue_size)
        instance._tracker = SimpleTracker(
            identifier=stage_config.stage_name,
            max_records=stage_config.max_log_records,
            in_progress=[ProgressReporter() for _ in range(stage_config.workers)],
            batch_configs=[batch_config for _ in range(stage_config.workers)],
        )
        return instance

    @property
    def tracker(self) -> SimpleTracker:
        return self._tracker

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
        executor: ThreadPoolExecutor | None = None,
    ):
        fn = self._fns[runner_index]
        reporter = self._tracker.in_progress[runner_index]
        use_progress = accepts_progress(fn)

        log = logging.getLogger(f"conveyor.batch/{self.config.stage_name}:{runner_index}")
        log.info("Starting worker")

        while True:
            batch: list[tuple[int, Any]] = []

            first = await self._in_q.get()
            self._in_q.task_done()

            if first is _SENTINEL:
                break

            batch.append(first)
            deadline = asyncio.get_event_loop().time() + self.batch_config.timeout_s

            while len(batch) < self.batch_config.max_batch_size:
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

            try:
                t0 = time.perf_counter()
                log.info("Start batch processing %d requests", len(payloads))

                args = (payloads, reporter) if use_progress else (payloads,)
                outputs = await self._run_fn(fn, args, executor=executor)

                log.info("Finished batch %d requests in %.2fs", len(payloads), time.perf_counter() - t0)
                self._tracker.on_process_done(time.perf_counter() - t0, success=True)

                for rid, out in zip(req_ids, outputs):
                    if next_q is not None:
                        await next_q.put((rid, out))
                    elif rid in results:
                        results[rid].set_result(out)

            except Exception as e:
                self._tracker.on_process_done(time.perf_counter() - t0, success=False)
                for rid in req_ids:
                    if rid in results and not results[rid].done():
                        results[rid].set_exception(e)

            finally:
                reporter.reset()
