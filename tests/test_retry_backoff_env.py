import pytest

from arrival.jobs.limits import LimitError, Limits


def test_retry_backoff_must_be_whole():
    with pytest.raises(LimitError, match="ARRIVAL_JOB_RETRY_BACKOFF.*whole number"):
        Limits.from_env({"ARRIVAL_JOB_RETRY_BACKOFF": "2.5"})
    assert Limits.from_env({"ARRIVAL_JOB_RETRY_BACKOFF": "0"}).job_retry_backoff == 0
    assert Limits.from_env({"ARRIVAL_JOB_RETRY_BACKOFF": "10"}).job_retry_backoff == 10
