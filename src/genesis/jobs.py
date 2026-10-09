"""Process-wide lock that lets only one job run at a time.

All jobs share it, not only GPU work: cutting, captioning, metadata preparation and
training read and write the same dataset files.
"""
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class JobBusyError(RuntimeError):
    """Raised when a job is started while another one is running."""


class JobLock:
    """Non-blocking lock that remembers which job holds it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_job: str | None = None
        self._started_at = 0.0

    @contextmanager
    def acquire(self, job: str) -> Iterator[None]:
        """Hold the lock for the duration of the `with` block.

        Raises:
            JobBusyError: Another job holds the lock.
        """
        if not self._lock.acquire(blocking=False):
            raise JobBusyError(f"Busy with {self._active_job}, wait for it to finish.")
        self._active_job, self._started_at = job, time.monotonic()
        logger.info("Job started: %s", job)
        try:
            yield
        finally:
            logger.info("Job finished: %s (%.1fs)", job, time.monotonic() - self._started_at)
            self._active_job = None
            self._lock.release()

    @property
    def active_job(self) -> str | None:
        """Name of the running job, None when idle."""
        return self._active_job

    @property
    def elapsed(self) -> float:
        """Seconds since the active job started, 0.0 when idle."""
        return time.monotonic() - self._started_at if self._active_job else 0.0


job_lock = JobLock()
