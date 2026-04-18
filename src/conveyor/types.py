"""Shared sentinel object used to signal worker shutdown."""

from __future__ import annotations

_SENTINEL = object()

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Awaitable, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio

if TYPE_CHECKING:
    from conveyor.metrics import StageMetrics

class IStage(ABC):
    _metrics: StageMetrics | None

    @property
    @abstractmethod
    def in_q(self) -> asyncio.Queue:
        ...

    @property
    @abstractmethod
    def stage_name(self) -> str:
        ...

    @property
    @abstractmethod
    def fns(self) -> list[Callable[..., Awaitable[Any] | Any]]:
        ...

    @abstractmethod
    def _run_fn(self, fn: Callable[..., Awaitable[Any] | Any], args: tuple[Any, ...], executor: ThreadPoolExecutor | None = None) -> Awaitable[Any]:
        ...

    @abstractmethod
    async def _worker(
        self, 
        next_q: asyncio.Queue | None, 
        results: dict[int, asyncio.Future], 
        runner_index: int, 
        metrics: StageMetrics,
        executor: ThreadPoolExecutor | None = None,
    ):
        ...
