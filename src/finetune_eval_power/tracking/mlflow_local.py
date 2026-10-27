"""A local MLflow run per sweep configuration, and a no-op when MLflow is absent.

Two decisions worth naming.

**SQLite, not the file store.** Current MLflow releases refuse a bare file backend
for anything that needs a run store with a schema, so the tracking URI is
`sqlite:///mlruns/tracking.db`. It needs no server and no network.

**Tracking never fails the sweep.** A tracking backend is observability, and
observability that can take down the thing it observes is a liability. Every call
here is wrapped, and a sweep run on a machine without the training extra installed
records nothing and still produces the identical measurement.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_DIRECTORY = "mlruns"
EXPERIMENT_NAME = "finetune-eval-power"


class SweepTracker:
    """Records one run per (arm, seed, freeze depth). Silent when MLflow is missing."""

    def __init__(self, client: Any | None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def run(self, name: str, params: dict[str, Any]) -> Iterator[SweepTracker]:
        if self._client is None:
            yield self
            return
        try:
            with self._client.start_run(run_name=name):
                self._client.log_params(params)
                yield self
        except Exception:  # pragma: no cover - tracking must never break the sweep
            yield self

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._client is None:
            return
        try:
            self._client.log_metrics(metrics)
        except Exception:  # pragma: no cover - see above
            return


def open_tracker(directory: str | Path = DEFAULT_TRACKING_DIRECTORY) -> SweepTracker:
    """Open a local tracker, or return a disabled one if MLflow is not installed."""
    try:
        import mlflow
    except ImportError:
        return SweepTracker(None)
    try:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{root / 'tracking.db'}")
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception:  # pragma: no cover - a broken backend disables tracking
        return SweepTracker(None)
    return SweepTracker(mlflow)
