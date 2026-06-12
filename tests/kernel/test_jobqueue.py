"""Tests for nyxara.kernel.jobqueue — the bounded async job queue.

Async paths are driven via ``asyncio.run`` inside plain sync tests, so they don't
depend on pytest-asyncio's mode configuration.
"""

from __future__ import annotations

import asyncio

from nyxara.kernel.jobqueue import Job, JobQueue, JobStatus


def test_run_sync_completes_all():
    q = JobQueue(concurrency=2)
    jobs = q.run_sync([lambda: 1 + 1, lambda: 2 * 3, lambda: "ok"])
    assert all(isinstance(j, Job) for j in jobs)
    assert {j.status for j in jobs} == {JobStatus.DONE}
    results = [j.result for j in jobs]
    assert 2 in results and 6 in results and "ok" in results


def test_failing_job_isolated():
    def boom():
        raise ValueError("nope")

    q = JobQueue(concurrency=2)
    jobs = q.run_sync([lambda: 42, boom, lambda: 7])
    by_status = {}
    for j in jobs:
        by_status.setdefault(j.status, []).append(j)
    assert JobStatus.DONE in by_status and JobStatus.FAILED in by_status
    failed = by_status[JobStatus.FAILED][0]
    assert failed.error and "nope" in failed.error
    # the other jobs still succeeded
    assert any(j.result == 42 for j in by_status[JobStatus.DONE])


def test_mixed_sync_and_async_jobs():
    async def go():
        q = JobQueue(concurrency=3)
        await q.start()

        async def acoro():
            await asyncio.sleep(0.01)
            return "async"

        q.submit(lambda: "sync")
        q.submit(acoro)
        await q.drain()
        return q

    q = asyncio.run(go())
    results = {j.result for j in q.jobs}
    assert "sync" in results and "async" in results
    assert q.status()[JobStatus.DONE.value] == 2


def test_bounded_concurrency():
    async def go():
        q = JobQueue(concurrency=2)
        await q.start()
        state = {"cur": 0, "max": 0}

        async def worker():
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.02)
            state["cur"] -= 1

        for _ in range(6):
            q.submit(worker)
        await q.drain()
        return state

    state = asyncio.run(go())
    assert state["max"] <= 2


def test_cancel_pending_job():
    async def go():
        q = JobQueue(concurrency=1)
        await q.start()

        async def slow():
            await asyncio.sleep(0.05)
            return "first"

        first = q.submit(slow)
        second = q.submit(lambda: "second")
        cancelled = q.cancel(second.id)
        await q.drain()
        return first, second, cancelled

    first, second, cancelled = asyncio.run(go())
    assert cancelled is True
    assert second.status is JobStatus.CANCELLED
    assert first.status is JobStatus.DONE


def test_status_counts():
    q = JobQueue(concurrency=2)
    q.run_sync([lambda: 1, lambda: 2])
    counts = q.status()
    assert counts.get(JobStatus.DONE.value, 0) == 2
