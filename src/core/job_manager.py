"""
Job management system to prevent concurrent GPU operations.
Ensures only one training/generation job runs at a time on the H200.
"""
import threading
import time
from contextlib import contextmanager
from typing import Optional
import logging

logger = logging.getLogger("JobManager")


class JobLock:
    """Thread-safe job lock for GPU operations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._active_job: Optional[str] = None
        self._start_time: Optional[float] = None

    @contextmanager
    def acquire_gpu(self, job_type: str, timeout: float = 5.0):
        """
        Acquire GPU lock for a job.

        Args:
            job_type: Type of job (e.g., "generation", "training", "captioning")
            timeout: Maximum seconds to wait for lock

        Raises:
            RuntimeError: If another job is already running
        """
        acquired = self._lock.acquire(timeout=timeout)
        if not acquired:
            raise RuntimeError(
                f"GPU is busy with {self._active_job}. "
                f"Please wait for the current job to finish."
            )

        if self._active_job is not None:
            self._lock.release()
            raise RuntimeError(
                f"GPU is busy with {self._active_job}. "
                f"Please wait for the current job to finish."
            )

        self._active_job = job_type
        self._start_time = time.time()
        logger.info(f"🔒 GPU locked for: {job_type}")

        try:
            yield self
        finally:
            duration = time.time() - self._start_time if self._start_time else 0
            logger.info(f"🔓 GPU released from: {job_type} (duration: {duration:.1f}s)")
            self._active_job = None
            self._start_time = None
            self._lock.release()

    def get_status(self) -> dict:
        """Get current job status."""
        return {
            "active": self._active_job is not None,
            "job_type": self._active_job,
            "duration": time.time() - self._start_time if self._start_time else 0
        }

    def is_locked(self) -> bool:
        """Check if GPU is currently locked."""
        return self._active_job is not None


# Global job manager instance
job_manager = JobLock()
