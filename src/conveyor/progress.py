from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable


@dataclass
class ProgressReporter:
    """Mutable progress object passed to stage functions via ``progress=`` kwarg.

    Call it like ``progress(current_step, total_steps)`` from inside your
    processing function to report progress on a long-running task.
    """

    current_step: int = 0
    total_steps: int = 0
    active: bool = False

    def __call__(self, current: int, total: int):
        self.current_step = current
        self.total_steps = total
        self.active = True

    def reset(self):
        self.current_step = 0
        self.total_steps = 0
        self.active = False

    @property
    def remaining_ratio(self) -> float:
        if self.total_steps <= 0:
            return 1.0
        return (self.total_steps - self.current_step) / self.total_steps


def accepts_progress(fn: Callable) -> bool:
    """Check whether *fn* has a ``progress`` parameter."""
    try:
        sig = inspect.signature(fn)
        return "progress" in sig.parameters
    except (ValueError, TypeError):
        return False
