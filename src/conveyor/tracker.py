from __future__ import annotations

from dataclasses import dataclass, field

from conveyor.config import BatchConfig
from conveyor.progress import ProgressReporter


@dataclass
class SimpleTracker:
    identifier: str = ""
    max_records: int = 128
    processing_time_records: list[float] = field(default_factory=list)
    failure_records: list[bool] = field(default_factory=list)
    in_progress: list[ProgressReporter] = field(default_factory=list)
    batch_configs: list[BatchConfig] = field(default_factory=list)

    def on_process_done(self, processing_time: float, success: bool):
        while len(self.processing_time_records) >= self.max_records:
            self.processing_time_records.pop(0)

        while len(self.failure_records) >= self.max_records:
            self.failure_records.pop(0)

        self.processing_time_records.append(processing_time)
        self.failure_records.append(not success)


@dataclass
class StatusReport:
    stats: list[SimpleTracker]
    running: bool
    available: bool
    slots: int
