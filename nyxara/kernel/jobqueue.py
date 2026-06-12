"""NYXARA · kernel/jobqueue.py — an async bounded worker pool (⚙).

A small, robust job queue for fanning out work — synchronous callables *or*
coroutine functions — across a bounded pool of asyncio workers. It exists so the
kernel and its faculties can run many independent units of work concurrently
without (a) blocking the event loop on sync code, or (b) exceeding the governor's
concurrency ceiling.

Design constraints, mirroring the rest of the kernel:

* **Bounded.** Parallelism is capped by an :class:`asyncio.Semaphore`, defaulting
  to ``settings.resources.max_concurrent_tasks`` (sanely capped). One slow burst
  can never starve the machine.
* **Priority-ordered.** Higher ``priority`` runs first; ties break by submission
  order (FIFO), so scheduling is deterministic and testable.
* **Sync-safe.** A plain callable is run in a thread executor so it never blocks
  the loop; a coroutine function is awaited directly.
* **Failure-isolated.** A job that raises is recorded FAILED (status + error) and
  never takes the queue — or its siblings — down with it.
* **Caller-friendly.** :meth:`run_sync` spins up its own loop for non-async callers
  and tests, returning completed :class:`Job` records.

Depends only on the standard library + :mod:`nyxara.kernel.config`.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.kernel.config import get_settings

__all__ = [
    "JobStatus",
    "Job",
    "JobQueue",
]

# Hard ceiling so a misconfigured budget can never spawn an absurd worker pool.
_MAX_CONCURRENCY = 32


# --------------------------------------------------------------------------- #
# Job record
# --------------------------------------------------------------------------- #
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A single unit of work and its lifecycle record.

    The callable + args are captured at submission; ``status``/``result``/``error``
    and the timing fields are filled in as the job moves through the pool.
    """

    id: str
    name: str
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    @property
    def done(self) -> bool:
        return self.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "status": self.status.value,
            "error": self.error,
            "duration_s": round(self.duration_s, 4) if self.duration_s is not None else None,
        }


# --------------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------------- #
class JobQueue:
    """An asyncio-based bounded worker pool with priority scheduling.

    Submit work with :meth:`submit` (returns a :class:`Job` immediately), then
    :meth:`drain` to await everything, or use :meth:`run_all` / :meth:`run_sync`
    as one-shot conveniences.
    """

    def __init__(self, concurrency: Optional[int] = None) -> None:
        if concurrency is None:
            try:
                concurrency = get_settings().resources.max_concurrent_tasks
            except Exception:  # noqa: BLE001 - config must never break the pool
                concurrency = 8
        self.concurrency = max(1, min(int(concurrency), _MAX_CONCURRENCY))
        self._jobs: Dict[str, Job] = {}
        self._tasks: Dict[str, "asyncio.Task[Any]"] = {}
        self._sem: Optional[asyncio.Semaphore] = None
        self._seq = itertools.count()
        # (-(priority), seq) so higher priority and earlier submission sort first.
        self._order: Dict[str, tuple] = {}

    # ---- introspection ---- #
    @property
    def jobs(self) -> List[Job]:
        return list(self._jobs.values())

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def status(self) -> Dict[str, int]:
        """Counts of jobs by status (always includes every status key)."""
        counts = {s.value: 0 for s in JobStatus}
        for job in self._jobs.values():
            counts[job.status.value] += 1
        return counts

    # ---- submission ---- #
    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: Optional[str] = None,
        priority: int = 0,
        **kwargs: Any,
    ) -> Job:
        """Register a job and schedule it (if a loop is running). Returns immediately.

        ``fn`` may be a sync callable (run in a thread executor so it never blocks
        the loop) or a coroutine function (awaited directly). Priority is honoured
        only among jobs still PENDING when a worker slot frees up.
        """
        job = Job(
            id=uuid.uuid4().hex,
            name=name or getattr(fn, "__name__", "job"),
            fn=fn,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )
        self._jobs[job.id] = job
        self._order[job.id] = (-priority, next(self._seq))
        # If we're already inside the running loop, schedule eagerly.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and self._sem is not None:
            self._tasks[job.id] = loop.create_task(self._run_job(job))
        return job

    async def start(self) -> None:
        """Initialise the semaphore and schedule any already-submitted PENDING jobs."""
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        loop = asyncio.get_running_loop()
        for job in self._pending_in_priority_order():
            if job.id not in self._tasks:
                self._tasks[job.id] = loop.create_task(self._run_job(job))

    def _pending_in_priority_order(self) -> List[Job]:
        pending = [j for j in self._jobs.values() if j.status is JobStatus.PENDING]
        pending.sort(key=lambda j: self._order[j.id])
        return pending

    async def _run_job(self, job: Job) -> None:
        assert self._sem is not None
        # A job cancelled before acquiring a slot stays CANCELLED.
        if job.status is JobStatus.CANCELLED:
            return
        async with self._sem:
            if job.status is JobStatus.CANCELLED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            try:
                if asyncio.iscoroutinefunction(job.fn):
                    job.result = await job.fn(*job.args, **job.kwargs)
                else:
                    loop = asyncio.get_running_loop()
                    job.result = await loop.run_in_executor(
                        None, lambda: job.fn(*job.args, **job.kwargs)
                    )
                job.status = JobStatus.DONE
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.error = "cancelled"
                job.finished_at = time.time()
                raise
            except Exception as exc:  # noqa: BLE001 - isolate every failure
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                if job.finished_at is None:
                    job.finished_at = time.time()

    # ---- draining / running ---- #
    async def drain(self) -> List[Job]:
        """Await every submitted job (scheduling any still-pending first)."""
        await self.start()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        return self.jobs

    async def run_all(self, jobs: Optional[Sequence[Job]] = None) -> List[Job]:
        """Start the pool and drain it. ``jobs`` is accepted for symmetry/ergonomics.

        Any jobs passed that are not yet registered are adopted before draining.
        """
        if jobs:
            for job in jobs:
                self._jobs.setdefault(job.id, job)
                self._order.setdefault(job.id, (-job.priority, next(self._seq)))
        return await self.drain()

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Pending jobs flip to CANCELLED; running tasks are cancelled.

        Returns True if the job existed and was cancellable (not already finished).
        """
        job = self._jobs.get(job_id)
        if job is None or job.done:
            return False
        if job.status is JobStatus.RUNNING:
            task = self._tasks.get(job_id)
            if task is not None:
                task.cancel()
            return True
        # PENDING: mark cancelled so the worker skips it when its turn arrives.
        job.status = JobStatus.CANCELLED
        job.finished_at = time.time()
        return True

    # ---- synchronous convenience ---- #
    def run_sync(self, callables: Sequence[Callable[..., Any]]) -> List[Job]:
        """Run a batch of zero-arg callables (sync or coroutine fns) to completion.

        Spins up a private event loop, submits each callable, drains the pool
        bounded-concurrently, and returns the completed :class:`Job` records — handy
        for non-async callers and tests.
        """
        for fn in callables:
            self.submit(fn, name=getattr(fn, "__name__", "job"))

        async def _go() -> List[Job]:
            return await self.drain()

        return asyncio.run(_go())


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA jobqueue self-test")
    print("=" * 70)

    # ---- run_sync with mixed sync + async callables ---- #
    def sync_job(x: int = 2) -> int:
        return x * x

    async def async_job() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    def boom() -> None:
        raise ValueError("intentional")

    q = JobQueue(concurrency=4)
    jobs = q.run_sync([sync_job, async_job, boom, sync_job])
    statuses = {j.name: j.status for j in jobs}
    print(f"run_sync statuses    : { {k: v.value for k, v in statuses.items()} }")
    assert statuses["sync_job"] is JobStatus.DONE
    assert statuses["async_job"] is JobStatus.DONE
    assert statuses["boom"] is JobStatus.FAILED
    # a failing job is isolated, others still complete
    assert q.status()["done"] == 3
    assert q.status()["failed"] == 1
    print("failure isolation    : a failing job stays FAILED, siblings DONE ✓")

    # ---- bounded concurrency is respected ---- #
    counter = {"now": 0, "max": 0}

    async def tracked() -> None:
        counter["now"] += 1
        counter["max"] = max(counter["max"], counter["now"])
        await asyncio.sleep(0.02)
        counter["now"] -= 1

    async def _bounded() -> None:
        q2 = JobQueue(concurrency=3)
        for _ in range(12):
            q2.submit(tracked, name="tracked")
        await q2.drain()
        assert q2.status()["done"] == 12

    asyncio.run(_bounded())
    print(f"max concurrent seen  : {counter['max']} (cap 3)")
    assert counter["max"] <= 3

    # ---- priority ordering ---- #
    order: List[str] = []

    async def _priority() -> None:
        q3 = JobQueue(concurrency=1)

        def record(tag: str) -> str:
            order.append(tag)
            return tag

        q3.submit(record, "low", name="low", priority=0)
        q3.submit(record, "high", name="high", priority=10)
        q3.submit(record, "mid", name="mid", priority=5)
        await q3.drain()

    asyncio.run(_priority())
    print(f"priority order       : {order}")
    assert order[0] == "high"

    # ---- cancel a pending job ---- #
    async def _cancel() -> None:
        q4 = JobQueue(concurrency=1)

        async def slow() -> None:
            await asyncio.sleep(0.05)

        first = q4.submit(slow, name="first")  # occupies the only slot
        pending = q4.submit(slow, name="pending")
        assert q4.cancel(pending.id) is True
        await q4.drain()
        assert pending.status is JobStatus.CANCELLED
        assert first.status is JobStatus.DONE

    asyncio.run(_cancel())
    print("cancel pending job   : pending -> CANCELLED, first -> DONE ✓")

    print("\nALL SELF-TESTS PASSED ✓")
