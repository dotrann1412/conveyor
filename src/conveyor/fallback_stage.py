from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from conveyor.types import _SENTINEL
from conveyor.metrics import StageMetrics
from conveyor.stage import Stage
from concurrent.futures import ThreadPoolExecutor


class FallbackStage(Stage):
    """Terminal stage that handles failures from upstream stages.

    Fn signature: ``fn(payload, exc) -> result``. The return value resolves
    the request future. If the fallback fn itself raises, the caller sees
    that exception.
    """

    async def _worker(
        self,
        next_q: asyncio.Queue | None,
        results: dict[int, asyncio.Future],
        runner_index: int,
        metrics: StageMetrics,
        executor: ThreadPoolExecutor | None = None,
        err_q: asyncio.Queue | None = None,
    ):
        fn = self._fns[runner_index]
        logger = logging.getLogger(f"fallback:{self._stage_name}:{runner_index}")
        logger.info("Starting fallback worker")

        while True:
            item = await self._in_q.get()

            if item is _SENTINEL:
                self._in_q.task_done()
                break

            req_id, payload, exc = item
            t0 = time.perf_counter()

            try:
                result = await self._run_fn(fn, (payload, exc), executor=executor)
                elapsed = time.perf_counter() - t0

                logger.info("Recovered request %s in %.2fs", req_id, elapsed)
                metrics.record_success(1, elapsed)

                if req_id in results and not results[req_id].done():
                    results[req_id].set_result(result)

            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error("Fallback failed for request %s: %s", req_id, e)
                metrics.record_failure(1, elapsed)

                if req_id in results and not results[req_id].done():
                    results[req_id].set_exception(e)

            finally:
                self._in_q.task_done()
