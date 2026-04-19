"""Prometheus metrics for Conveyor pipeline observability.

Install via ``pip install conveyor[metrics]``.  When ``prometheus_client``
is not installed every recording method is a silent no-op.

Exposed metrics (all labeled by *pipeline* and *stage*):

- ``conveyor_items_total``  (Counter, +status label)
    Derive throughput:   rate(conveyor_items_total[5m])
    Derive error rate:   rate(conveyor_items_total{status="error"}[5m])
                       / rate(conveyor_items_total[5m])

- ``conveyor_processing_duration_seconds``  (Histogram)
    Latency percentiles via histogram_quantile().
"""

from __future__ import annotations

try:
    from prometheus_client import (
        Counter,
        Histogram,
        CollectorRegistry,
        disable_created_metrics,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

if _HAS_PROMETHEUS:
    disable_created_metrics()

    REGISTRY = CollectorRegistry()

    _items_total = Counter(
        "conveyor_items_total",
        "Total items processed by a pipeline stage",
        ["pipeline", "stage", "status"],
        registry=REGISTRY,
    )

    _processing_seconds = Histogram(
        "conveyor_processing_duration_seconds",
        "Wall-clock processing time per item or batch",
        ["pipeline", "stage"],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf"),
        ),
        registry=REGISTRY,
    )


class StageMetrics:
    """Records Prometheus metrics for one pipeline stage.

    All methods are safe to call when ``prometheus_client`` is absent.
    """

    __slots__ = ("_p", "_s")

    def __init__(self, pipeline_name: str, stage_name: str) -> None:
        self._p = pipeline_name
        self._s = stage_name

    def record_success(self, count: int, duration_s: float) -> None:
        if not _HAS_PROMETHEUS:
            return

        _items_total.labels(self._p, self._s, "success").inc(count)
        _processing_seconds.labels(self._p, self._s).observe(duration_s)

    def record_failure(self, count: int, duration_s: float) -> None:
        if not _HAS_PROMETHEUS:
            return

        _items_total.labels(self._p, self._s, "error").inc(count)
        _processing_seconds.labels(self._p, self._s).observe(duration_s)
