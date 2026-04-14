from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageConfig:
    stage_name: str = ""
    workers: int = 1
    max_queue_size: int = 100
    max_log_records: int = 128
    device_ids: list[int] | None = None

    def __post_init__(self):
        if self.device_ids is not None:
            self.workers = len(self.device_ids)


@dataclass
class BatchConfig:
    max_batch_size: int = 32
    timeout_s: float = 1
