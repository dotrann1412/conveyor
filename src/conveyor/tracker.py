from __future__ import annotations

from dataclasses import dataclass

@dataclass
class StageInfo:
    """Snapshot of a single stage's health."""

    name: str
    workers_alive: int
    workers_total: int
    queue_depth: int
    queue_capacity: int
    utilization: float


@dataclass
class StatusReport:
    """Pipeline health check returned by ``Pipeline.report()``."""

    running: bool
    available: bool
    slots: int
    stages: list[StageInfo]
    utilization: float
