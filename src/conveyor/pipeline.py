from __future__ import annotations

import asyncio
from itertools import count
from typing import Any, Generic, TypeVar

from conveyor._sentinel import _SENTINEL
from conveyor.batch_stage import BatchStage
from conveyor.stage import Stage
from conveyor.tracker import StatusReport

T = TypeVar("T")

AnyStage = Stage | BatchStage


class Pipeline(Generic[T]):
    """Streaming inference pipeline with stage-level parallelism.

    Stages are connected by async queues and run concurrently --
    while request A is in the model stage, request B can be preprocessing.
    """

    def __init__(self, stages: list[AnyStage]):
        self._stages = stages
        self._results: dict[int, asyncio.Future] = {}
        self._counter = count()
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True

        for i, stage in enumerate(self._stages):
            next_q = self._stages[i + 1]._in_q if i + 1 < len(self._stages) else None
            for runner_index in range(stage.config.workers):
                self._workers.append(
                    asyncio.create_task(
                        stage._worker(next_q, self._results, runner_index)
                    )
                )

    async def stop(self):
        if not self._running:
            return

        for stage in self._stages:
            for _ in range(stage.config.workers):
                await stage._in_q.put(_SENTINEL)

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._running = False

    def available_slots(self) -> int:
        if not self._stages:
            return 0
        return self._stages[0].config.max_queue_size - self._stages[0]._in_q.qsize()

    async def submit(self, payload: Any) -> Any:
        """Submit a single request and wait for its result."""
        req_id = next(self._counter)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._results[req_id] = fut

        await self._stages[0]._in_q.put((req_id, payload))

        try:
            return await fut
        finally:
            self._results.pop(req_id, None)

    async def submit_nowait(self, payload: Any) -> bool:
        """Submit a request without waiting for the result."""
        if not self._running:
            return False
        req_id = next(self._counter)
        await self._stages[0]._in_q.put((req_id, payload))
        return True

    async def report(self) -> StatusReport:
        is_available = (
            len(self._stages) > 0
            and self._running
            and all(stage.config.workers > 0 for stage in self._stages)
        )

        cnt_stage_instances: dict[str, int] = {
            stage.config.stage_name: 0 for stage in self._stages
        }

        for worker, stage in zip(self._workers, self._stages):
            if not worker.done():
                cnt_stage_instances[stage.config.stage_name] += 1

        is_available &= all(
            cnt_stage_instances[stage.config.stage_name] > 0
            for stage in self._stages
        )

        return StatusReport(
            stats=[stage.tracker for stage in self._stages],
            running=self._running,
            available=is_available,
            slots=self.available_slots(),
        )

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()
