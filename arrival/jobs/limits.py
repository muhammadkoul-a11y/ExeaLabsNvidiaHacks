"""Worker and queue limits. One place that names every ceiling, read from the environment for a hosted deployment.

``Limits()`` holds the defaults; ``Limits.from_env()`` overrides them from ``ARRIVAL_*`` variables. A limit of 0
(or unset retention) means "not enforced" and is never the default for upload size, episodes, people or queue length.
``job_retries`` is a count, not a ceiling: 0 means a failed job is never retried.
``job_retry_backoff`` is the first retry wait in seconds; each later attempt doubles it, up to 300.
This module imports nothing heavy so the web server can use it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

ENV = {
    "max_upload_bytes": ("ARRIVAL_MAX_UPLOAD_MB", 1024 * 1024),
    "max_episodes": ("ARRIVAL_MAX_EPISODES", 1),
    "max_people": ("ARRIVAL_MAX_PEOPLE", 1),
    "max_job_seconds": ("ARRIVAL_MAX_JOB_SECONDS", 1),
    "max_queue": ("ARRIVAL_MAX_QUEUE", 1),
    "max_job_bytes": ("ARRIVAL_MAX_JOB_MB", 1024 * 1024),
    "retention_hours": ("ARRIVAL_RETENTION_HOURS", 1),
    "job_retries": ("ARRIVAL_JOB_RETRIES", 1),
    "job_retry_backoff": ("ARRIVAL_JOB_RETRY_BACKOFF", 1),
}


class LimitError(ValueError):
    """A request or setting outside the configured limits."""


class QueueFull(LimitError):
    """The queue already holds ``max_queue`` queued or running jobs."""


@dataclass(frozen=True)
class Limits:
    max_upload_bytes: int = MAX_UPLOAD_BYTES  # one upload; web 413 and worker re-check of upload.bin
    max_episodes: int = 100  # episodes per job (the sim's attempt count)
    max_people: int = 4  # simulated people per job
    max_job_seconds: float = 1800.0  # wall clock per job; 0 runs the job in-process with no timeout
    max_queue: int = 20  # queued + running jobs; a new upload beyond this is refused (0 = unlimited)
    max_job_bytes: int = 0  # resident memory of a job's child process tree; the worker kills it past this; 0 = off
    retention_hours: float | None = None  # finished jobs older than this are deleted by the worker sweep; None keeps them
    stale_seconds: float = 120.0  # a running job whose worker has not heartbeat this long is requeued
    job_retries: int = 2  # a failed attempt goes back to the queue at most this many times; 0 = never retry
    job_retry_backoff: int = 5  # seconds before a requeued job can be claimed; doubles per attempt, capped at 300

    def check_request(self, episodes: int, people: int) -> None:
        """Raise ``LimitError`` if one job's sim options exceed the ceilings."""
        if type(episodes) is not int or episodes < 1:
            raise LimitError("episodes must be a positive whole number")
        if type(people) is not int or people < 0:
            raise LimitError("people must be a nonnegative whole number")
        if self.max_episodes and episodes > self.max_episodes:
            raise LimitError(f"episodes {episodes} exceeds the limit of {self.max_episodes}")
        if self.max_people and people > self.max_people:
            raise LimitError(f"people {people} exceeds the limit of {self.max_people}")

    def replace(self, **kw) -> "Limits":
        known = {f.name for f in fields(self)}
        bad = set(kw) - known
        if bad:
            raise TypeError(f"unknown limits {sorted(bad)}")
        return Limits(**{**{k: getattr(self, k) for k in known}, **{k: v for k, v in kw.items() if v is not None}})

    @classmethod
    def from_env(cls, environ=None) -> "Limits":
        env = os.environ if environ is None else environ
        kw: dict = {}
        for name, (var, scale) in ENV.items():
            raw = env.get(var)
            if raw is None or not str(raw).strip():
                continue
            try:
                num = float(raw)
            except ValueError:
                raise LimitError(f"{var}={raw!r} is not a number") from None
            if not num >= 0 or num == float("inf"):
                raise LimitError(f"{var}={raw!r} must be a finite number >= 0")
            if name in ("max_job_seconds", "retention_hours"):
                kw[name] = num
            else:
                if name in ("max_episodes", "max_people", "max_queue", "job_retries", "job_retry_backoff") and not num.is_integer():
                    raise LimitError(f"{var}={raw!r} must be a whole number")
                scaled = num * scale
                if num > 0 and scaled < 1:
                    raise LimitError(f"{var}={raw!r} is too small for this limit")
                kw[name] = int(scaled)
        return cls(**kw)
