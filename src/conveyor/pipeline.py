from __future__ import annotations

import asyncio
import inspect
import logging
from itertools import count
from typing import Any, Generic, TypeVar

from conveyor.types import _SENTINEL, IStage
from conveyor.metrics import StageMetrics
from conveyor.tracker import StatusReport, StageInfo
from concurrent.futures import ThreadPoolExecutor

T = TypeVar("T")
logger = logging.getLogger(__name__)


class Pipeline(Generic[T]):
    """Streaming inference pipeline with stage-level parallelism.

    Stages are connected by async queues and run concurrently --
    while request A is in the model stage, request B can be preprocessing.
    """

    def __init__(self, stages: list[IStage], *, name: str = "default"):
        self._stages = stages
        self._name = name
        self._futures: dict[int, asyncio.Future] = {}
        self._counter = count()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._pool: ThreadPoolExecutor | None = None

    @property
    def name(self) -> str:
        return self._name

    async def start(self):
        if self._running:
            return

        self._running = True

        pool_size = 0
        for stage in self._stages:
            for fn in stage.fns:
                if not inspect.iscoroutinefunction(fn):
                    pool_size += 1

        if pool_size > 0:
            logger.info("Starting thread pool with %d workers", pool_size)
            self._pool = ThreadPoolExecutor(max_workers=pool_size)

        for i, stage in enumerate(self._stages):
            next_q = self._stages[i + 1].in_q if i + 1 < len(self._stages) else None
            for runner_index in range(len(stage.fns)):
                self._workers.append(
                    asyncio.create_task(
                        stage._worker(
                            next_q, 
                            self._futures, 
                            runner_index, 
                            StageMetrics(self._name, stage.stage_name),
                            self._pool,
                        )
                    )
                )

    async def stop(self):
        if not self._running:
            return

        for stage in self._stages:
            for _ in range(len(stage.fns)):
                await stage.in_q.put(_SENTINEL)

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._running = False

        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def available_slots(self) -> int:
        if not self._stages:
            return 0

        q = self._stages[0].in_q
        return q.maxsize - q.qsize()

    async def submit(self, payload: Any) -> Any:
        """Submit a single request and wait for its result."""
        req_id = next(self._counter)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._futures[req_id] = fut

        await self._stages[0].in_q.put((req_id, payload))

        try:
            return await fut
        finally:
            self._futures.pop(req_id, None)

    async def submit_nowait(self, payload: Any) -> bool:
        """Submit a request without waiting for the result."""
        if not self._running:
            return False
        req_id = next(self._counter)
        await self._stages[0].in_q.put((req_id, payload))
        return True

    async def report(self) -> StatusReport:
        worker_idx = 0
        stage_infos: list[StageInfo] = []

        for stage in self._stages:
            n_fns = len(stage.fns)
            alive = sum(
                1 for w in self._workers[worker_idx : worker_idx + n_fns]
                if not w.done()
            )
            stage_infos.append(
                StageInfo(
                    name=stage.stage_name,
                    workers_alive=alive,
                    workers_total=n_fns,
                    queue_depth=stage.in_q.qsize(),
                    queue_capacity=stage.in_q.maxsize,
                )
            )
            worker_idx += n_fns

        is_available = (
            self._running
            and len(self._stages) > 0
            and all(si.workers_alive > 0 for si in stage_infos)
        )

        return StatusReport(
            running=self._running,
            available=is_available,
            slots=self.available_slots(),
            stages=stage_infos,
        )

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()
