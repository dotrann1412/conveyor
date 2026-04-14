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

class Stage(Generic[T]):
    """A single processing stage in the pipeline.

    Runs *fn* on each item independently, with *workers* concurrent tasks
    pulling from an input queue and pushing to the next stage.
    """

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any] | Any],
        config: StageConfig | None = None,
    ):
        self.config = config or StageConfig()
        self._fns: list[Callable[..., Awaitable[Any] | Any]] = [fn] * self.config.workers
        self._in_q: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._tracker = SimpleTracker(
            identifier=self.config.stage_name,
            max_records=self.config.max_log_records,
            in_progress=[ProgressReporter() for _ in range(self.config.workers)],
            batch_configs=[BatchConfig(max_batch_size=1) for _ in range(self.config.workers)],
        )

    @property
    def fns(self) -> list[Callable[..., Awaitable[Any] | Any]]:
        return self._fns

    @classmethod
    def from_factory(
        cls,
        fn_factory: Callable[[int], Callable[..., Awaitable[Any] | Any]],
        device_ids: list[int],
        config: StageConfig | None = None,
    ) -> Stage[T]:
        config = config or StageConfig()
        config.device_ids = device_ids
        config.workers = len(device_ids)
        instance = cls.__new__(cls)
        instance.config = config
        fns = [fn_factory(did) for did in device_ids]
        instance._fns = fns
        instance._in_q = asyncio.Queue(maxsize=config.max_queue_size)
        instance._tracker = SimpleTracker(
            identifier=config.stage_name,
            max_records=config.max_log_records,
            in_progress=[ProgressReporter() for _ in range(config.workers)],
            batch_configs=[BatchConfig(max_batch_size=1) for _ in range(config.workers)],
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
        log = logging.getLogger(f"conveyor.stage/{self.config.stage_name}:{runner_index}")
        log.info("Starting worker")

        while True:
            item = await self._in_q.get()

            if item is _SENTINEL:
                self._in_q.task_done()
                break

            req_id, payload = item

            try:
                t0 = time.perf_counter()
                log.info("Start processing request %s", req_id)

                args = (payload, reporter) if use_progress else (payload,)
                result = await self._run_fn(fn, args, executor=executor)

                log.info("Finished request %s in %.2fs", req_id, time.perf_counter() - t0)

                if next_q is not None:
                    await next_q.put((req_id, result))
                elif req_id in results:
                    results[req_id].set_result(result)

                self._tracker.on_process_done(time.perf_counter() - t0, success=True)

            except Exception as e:
                log.error("Error processing request %s: %s", req_id, e)
                self._tracker.on_process_done(time.perf_counter() - t0, success=False)
                if req_id in results and not results[req_id].done():
                    results[req_id].set_exception(e)

            finally:
                reporter.reset()
                self._in_q.task_done()
