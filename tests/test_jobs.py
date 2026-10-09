import pytest

from genesis.jobs import JobBusyError, JobLock


def test_second_job_is_rejected_and_lock_released():
    lock = JobLock()
    with lock.acquire("first"):
        assert lock.active_job == "first"
        with pytest.raises(JobBusyError, match="first"):
            with lock.acquire("second"):
                pass
    assert lock.active_job is None
    with lock.acquire("second"):
        pass


def test_lock_released_on_error():
    lock = JobLock()
    with pytest.raises(ValueError):
        with lock.acquire("failing"):
            raise ValueError
    assert lock.active_job is None and lock.elapsed == 0.0
