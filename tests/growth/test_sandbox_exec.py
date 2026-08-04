"""Tests for growth/sandbox_exec.py — running generated code without losing the process.

The property under test is not "does it compute x*x". It is: **a program that never terminates,
recurses forever, or allocates the box to death must become a reported failure rather than a
hung worker.** The existing `synthesis._sandbox_compile` has no timeout at all, which is safe
only while the "candidate" is always the trusted reference; the moment mutated programs are fed
to it — which is what corpus-scale code generation does — a single non-terminating mutant costs
the whole build.
"""

from __future__ import annotations

import time

import pytest

from nyxara.growth.sandbox_exec import (
    GuardedExecutor,
    SandboxLimits,
    run_guarded,
)

pytestmark = pytest.mark.skipif(
    not hasattr(__import__("resource"), "setrlimit"),
    reason="the guarded sandbox needs POSIX resource limits")

_FAST = SandboxLimits(cpu_seconds=1, wall_seconds=4.0)


# --------------------------------------------------------------------------- #
# It has to actually run code
# --------------------------------------------------------------------------- #
def test_a_normal_function_runs_and_returns_every_result() -> None:
    r = run_guarded("def f(x):\n    return x * x\n", "f", [[3], [4], [5]], limits=_FAST)
    assert r.ok and r.values() == [9, 16, 25]


def test_a_missing_callable_is_reported_not_raised() -> None:
    r = run_guarded("g = 1\n", "f", [[1]], limits=_FAST)
    assert not r.ok and "f" in r.error


def test_a_syntax_error_is_reported_not_raised() -> None:
    r = run_guarded("def f(x:\n", "f", [[1]], limits=_FAST)
    assert not r.ok and "Error" in r.error


def test_a_raising_function_fails_only_that_call() -> None:
    r = run_guarded("def f(x):\n    return 1 // x\n", "f", [[1], [0], [2]], limits=_FAST)
    assert r.ok, "the program itself loaded fine"
    assert r.results[0] == (True, 1)
    assert r.results[1][0] is False and "ZeroDivisionError" in r.results[1][1]
    assert r.results[2] == (True, 0)


# --------------------------------------------------------------------------- #
# The failures that cost a build
# --------------------------------------------------------------------------- #
def test_an_infinite_loop_times_out_instead_of_hanging() -> None:
    """The regression this module exists for: a mutant whose base case never fires."""
    started = time.monotonic()
    r = run_guarded("def f(n):\n    while n != 1:\n        n = n - 2\n    return n\n",
                    "f", [[4]], limits=SandboxLimits(cpu_seconds=1, wall_seconds=5.0))
    elapsed = time.monotonic() - started
    assert not r.ok and r.timed_out
    assert elapsed < 5.0, f"took {elapsed:.1f}s — the deadline did not fire"


def test_runaway_recursion_is_capped() -> None:
    r = run_guarded("def f(n):\n    return f(n + 1)\n", "f", [[0]], limits=_FAST)
    assert r.ok and r.results[0][0] is False
    assert "RecursionError" in r.results[0][1]


def test_a_memory_bomb_is_capped() -> None:
    r = run_guarded("def f(n):\n    return [0] * (10 ** 9)\n", "f", [[1]],
                    limits=SandboxLimits(cpu_seconds=2, address_space=64 * 1024 * 1024,
                                         wall_seconds=6.0))
    assert not r.ok or r.results[0][0] is False


# --------------------------------------------------------------------------- #
# Containment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src, needle", [
    ("def f(x):\n    import os\n    return os.listdir('/')\n", "not permitted"),
    ("def f(x):\n    import socket\n    return 1\n", "not permitted"),
    ("def f(x):\n    import subprocess\n    return 1\n", "not permitted"),
    ("def f(x):\n    return open('/tmp/nyx-sbx', 'w')\n", "open"),
    ("def f(x):\n    return eval('1+1')\n", "eval"),
])
def test_the_sandbox_cannot_reach_the_machine(src: str, needle: str) -> None:
    r = run_guarded(src, "f", [[1]], limits=_FAST)
    detail = r.results[0][1] if (r.ok and r.results) else r.error
    assert (not r.ok) or r.results[0][0] is False, f"{src!r} was allowed to run"
    assert needle in str(detail)


def test_the_allowlisted_modules_still_work() -> None:
    """Barring `math` from every generated program would distort the data more than it protects."""
    r = run_guarded("def f(x):\n    import math\n    return math.gcd(x, 12)\n", "f", [[18]],
                    limits=_FAST)
    assert r.ok and r.values() == [6]


# --------------------------------------------------------------------------- #
# The executor is reusable, which is the whole performance argument
# --------------------------------------------------------------------------- #
def test_the_executor_survives_a_timeout_and_keeps_working() -> None:
    """A hung item must cost one restart, not the worker."""
    with GuardedExecutor(SandboxLimits(cpu_seconds=1, wall_seconds=3.0)) as ex:
        hung = ex.call("def f(n):\n    while True:\n        pass\n", "f", [[1]])
        assert hung.timed_out
        after = ex.call("def f(x):\n    return x + 1\n", "f", [[41]])
        assert after.ok and after.values() == [42]
        assert ex.restarts >= 1


def test_a_dropped_executor_does_not_leak_its_child() -> None:
    """Lazy owners (RivalVerifier) never call close(), so collection has to reap the child.

    Without this, a multi-hour build that constructs verifiers in a loop accumulates one idle
    subprocess per verifier until it exhausts the process table.
    """
    import gc
    import os
    import subprocess as sp

    def children() -> int:
        out = sp.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout
        return len(out.split())

    before = children()
    for i in range(4):
        ex = GuardedExecutor(_FAST)
        assert ex.call("def f(x):\n    return x + 1\n", "f", [[i]]).ok
        del ex
    gc.collect()
    time.sleep(0.5)
    assert children() <= before, "dropped executors leaked their sandbox children"


def test_the_child_is_reused_across_calls() -> None:
    """A fresh subprocess per item costs ~20 ms; at corpus scale that is hours of pure startup."""
    with GuardedExecutor(_FAST) as ex:
        for i in range(20):
            assert ex.call(f"def f(x):\n    return x * {i}\n", "f", [[2]]).ok
        assert ex.restarts == 0, "the executor restarted despite no failure"
