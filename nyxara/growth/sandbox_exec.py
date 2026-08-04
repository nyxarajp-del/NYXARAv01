"""NYXARA · growth/sandbox_exec.py — running generated code without hanging the process (⏱️→🔒).

:func:`~nyxara.growth.synthesis._sandbox_compile` strips ``__import__`` and whitelists a handful
of builtins, which correctly stops generated code reaching the filesystem, the network or the
interpreter. What it does **not** stop is the one failure that costs a multi-hour corpus build:

    def f(n):
        while n != 1:          # a mutated base case that never fires
            n = n - 2
        return n

No timeout, no recursion cap, no memory cap. ``f(3)`` never returns and the worker is gone.

Today that is unreachable, because ``synthesis._code`` sets ``candidate = reference`` — the
sandbox compiles the same trusted string twice and compares a function to itself. The moment a
real proposer feeds it *mutated* programs, which is exactly what corpus-scale code generation
does, every non-terminating mutant becomes a hung process.

**The design, and why it is a subprocess rather than a signal.** ``signal.alarm`` cannot
interrupt a tight loop running in C, and ``RLIMIT_AS`` cannot be lowered and then raised again
inside one process. Both need a separate address space. But a *fresh* subprocess per item costs
~20 ms of interpreter startup, and at 600k generated items that is 3.3 hours of pure overhead —
so the executor is **long-lived**: one child per worker, reused across items, restarted only
when it actually dies or times out. That is ~2 ms per call, and the 20 ms is paid once.

Limits are applied in the child *before* any generated code runs: ``RLIMIT_CPU`` (SIGXCPU on
overrun), ``RLIMIT_AS`` (allocation failure rather than swapping the box to death), a recursion
cap, and a wall-clock deadline in the parent for the case where the child wedges without
burning CPU — a child blocked on nothing still has to be reaped.

Pure standard library. Depends on nothing in ``nyxara``; ``growth/synthesis`` and
``growth/synth_code`` depend on it.
"""

from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SandboxLimits",
    "SandboxResult",
    "GuardedExecutor",
    "run_guarded",
]

# The child program. Kept as a string rather than a module so the executor has no import
# dependency on the package being installed, and so `-c` startup stays minimal.
_CHILD_SOURCE = r'''
import builtins, os, pickle, resource, signal, struct, sys

def _limit(cpu_seconds, address_space, recursion):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    if address_space:
        resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))      # cannot write to disk at all
    sys.setrecursionlimit(recursion)
    # SIGXCPU arrives a moment before SIGKILL on the hard limit; turning it into an exception
    # lets one runaway item be reported as a failure instead of killing the whole executor.
    signal.signal(signal.SIGXCPU, lambda *_a: (_ for _ in ()).throw(TimeoutError("cpu")))

_SAFE = ("abs","all","any","bin","bool","bytes","callable","chr","complex","dict","divmod",
         "enumerate","filter","float","format","frozenset","getattr","hasattr","hash","hex",
         "int","isinstance","issubclass","iter","len","list","map","max","min","next","oct",
         "ord","pow","print","range","repr","reversed","round","set","setattr","slice",
         "sorted","str","sum","tuple","type","zip","True","False","None","Exception",
         "ValueError","TypeError","IndexError","KeyError","ZeroDivisionError","StopIteration",
         "ArithmeticError","OverflowError","RecursionError","AttributeError","NotImplementedError")

def _builtins(allow):
    ns = {n: getattr(builtins, n) for n in _SAFE if hasattr(builtins, n)}
    if allow:
        # A deliberate, narrow allowlist. Barring `math` from every generated program would
        # distort the data far more than it protects a sandbox that already cannot touch the
        # filesystem, the network, or write a single byte (RLIMIT_FSIZE=0).
        import importlib
        real_import = builtins.__import__
        def guarded(name, *a, **k):
            root = name.split(".")[0]
            if root not in allow:
                raise ImportError("module %r is not permitted in the sandbox" % root)
            return real_import(name, *a, **k)
        ns["__import__"] = guarded
    return ns

def _read(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def main():
    cpu = int(os.environ.get("NYX_SBX_CPU", "2"))
    mem = int(os.environ.get("NYX_SBX_AS", str(256 * 1024 * 1024)))
    rec = int(os.environ.get("NYX_SBX_RECURSION", "200"))
    allow = tuple(x for x in os.environ.get("NYX_SBX_IMPORTS", "").split(",") if x)
    _limit(cpu, mem, rec)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = _read(stdin, 4)
        if header is None:
            return
        payload = _read(stdin, struct.unpack("<I", header)[0])
        if payload is None:
            return
        try:
            src, fn_name, calls = pickle.loads(payload)
            glb = {"__builtins__": _builtins(allow), "__name__": "__sandbox__"}
            exec(compile(src, "<sandbox>", "exec"), glb, glb)
            fn = glb.get(fn_name)
            if not callable(fn):
                out = (False, "no callable %r" % fn_name, [])
            else:
                results = []
                for args in calls:
                    try:
                        results.append((True, fn(*args)))
                    except Exception as exc:
                        results.append((False, "%s: %s" % (type(exc).__name__, exc)))
                out = (True, "", results)
        except BaseException as exc:
            out = (False, "%s: %s" % (type(exc).__name__, exc), [])
        try:
            blob = pickle.dumps(out)
        except Exception:
            blob = pickle.dumps((False, "result is not picklable", []))
        stdout.write(struct.pack("<I", len(blob)))
        stdout.write(blob)
        stdout.flush()

main()
'''


def _reap_box(box: Dict[str, Any]) -> None:
    """Kill whatever child ``box`` currently holds. Safe during interpreter shutdown."""
    proc = box.get("proc")
    box["proc"] = None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.kill()
        proc.wait(timeout=1.0)
    except Exception:  # noqa: BLE001 — shutdown ordering can pull the rug; never raise here
        pass


@dataclass(frozen=True)
class SandboxLimits:
    """Resource ceilings applied in the child before any generated code runs."""

    cpu_seconds: int = 2
    address_space: int = 256 * 1024 * 1024
    recursion: int = 200
    #: Wall-clock deadline in the parent. Strictly greater than ``cpu_seconds``: a child that is
    #: blocked rather than spinning burns no CPU and would never trip RLIMIT_CPU.
    wall_seconds: float = 5.0
    #: Modules generated code may import. Empty means none.
    allow_imports: Tuple[str, ...] = ("math", "itertools", "collections", "functools",
                                      "fractions", "heapq", "bisect", "string", "re")


@dataclass
class SandboxResult:
    """What one guarded evaluation produced."""

    ok: bool
    error: str = ""
    #: One ``(succeeded, value_or_message)`` per requested call, in order.
    results: List[Tuple[bool, Any]] = None  # type: ignore[assignment]
    timed_out: bool = False

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = []

    def values(self) -> List[Any]:
        """The successful return values; raises nothing — failures become ``None``."""
        return [v if good else None for good, v in self.results]


class GuardedExecutor:
    """A long-lived sandbox child, reused across evaluations and restarted when it dies.

    Not thread-safe by design: give each worker process its own executor rather than sharing one
    behind a lock, which would serialise the very thing multiprocessing exists to parallelise.
    """

    def __init__(self, limits: Optional[SandboxLimits] = None) -> None:
        self.limits = limits or SandboxLimits()
        self.restarts = 0
        # The live child lives in a plain dict so the finalizer can reach it WITHOUT holding a
        # reference to `self` — `weakref.finalize(self, fn, self)` would keep the executor alive
        # forever and never fire, which is the opposite of the intent.
        #
        # Callers that build an executor lazily (RivalVerifier does) have no natural place to
        # call close(). Without this, every such object leaks an idle child that survives until
        # the parent exits: invisible in a short run, process-table exhaustion in a multi-hour
        # corpus build that constructs verifiers in a loop. `weakref.finalize` also runs at
        # interpreter exit by default, so it covers both collection and shutdown.
        self._box: Dict[str, Any] = {"proc": None}
        self._finalizer = weakref.finalize(self, _reap_box, self._box)

    @property
    def _proc(self) -> Optional[subprocess.Popen]:
        return self._box["proc"]

    @_proc.setter
    def _proc(self, value: Optional[subprocess.Popen]) -> None:
        self._box["proc"] = value

    # ---- lifecycle ---- #
    def _spawn(self) -> subprocess.Popen:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "NYX_SBX_CPU": str(self.limits.cpu_seconds),
            "NYX_SBX_AS": str(self.limits.address_space),
            "NYX_SBX_RECURSION": str(self.limits.recursion),
            "NYX_SBX_IMPORTS": ",".join(self.limits.allow_imports),
            # -I would also drop site-packages; -S is enough and keeps startup fast.
            "PYTHONHASHSEED": "0",
        }
        # bufsize=0 is load-bearing, not a tuning knob. With Python's default buffering, the
        # parent's first `read` pulls the whole frame into the BufferedReader's internal buffer
        # and drains the pipe — so the next `select()` on the file descriptor sees no data and
        # blocks until the deadline, even though the bytes are already in hand. Every call would
        # report a spurious timeout. Unbuffered pipes keep select() and the fd in agreement.
        return subprocess.Popen(
            [sys.executable, "-S", "-c", _CHILD_SOURCE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, close_fds=True, bufsize=0,
        )

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            if self._proc is not None:
                self.restarts += 1
            self._proc = self._spawn()
        return self._proc

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            proc.kill()
            try:
                proc.wait(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.kill()
        try:
            proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass
        self.restarts += 1

    def __enter__(self) -> "GuardedExecutor":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---- the one operation ---- #
    def call(self, src: str, fn_name: str,
             calls: Sequence[Sequence[Any]]) -> SandboxResult:
        """Define ``src``, then invoke ``fn_name(*args)`` once per entry in ``calls``.

        Every call for one program is sent in a single round trip: the dominant cost at corpus
        scale is IPC, not execution, and a mutant is judged against ten inputs at once.
        """
        try:
            payload = pickle.dumps((src, fn_name, [list(c) for c in calls]))
        except Exception as exc:  # noqa: BLE001
            return SandboxResult(False, f"arguments are not picklable: {exc}")

        proc = self._ensure()
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(struct.pack("<I", len(payload)))
            proc.stdin.write(payload)
            proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001 — the child died between _ensure and the write
            self._kill()
            return SandboxResult(False, f"sandbox write failed: {exc}")

        blob = _read_with_deadline(proc, self.limits.wall_seconds)
        if blob is None:
            # Timed out or the child died mid-evaluation. Either way this executor's state is
            # unknown, so it is replaced rather than reused.
            self._kill()
            return SandboxResult(False, "sandbox timed out", timed_out=True)
        try:
            ok, error, results = pickle.loads(blob)
        except Exception as exc:  # noqa: BLE001
            self._kill()
            return SandboxResult(False, f"undecodable sandbox reply: {exc}")
        return SandboxResult(bool(ok), str(error), list(results))


def _read_with_deadline(proc: subprocess.Popen, seconds: float) -> Optional[bytes]:
    """Read one length-prefixed frame, giving up after ``seconds``.

    ``select`` on the child's stdout rather than a thread: the parent is often already inside a
    multiprocessing worker, and spawning a reader thread per evaluation would cost more than the
    evaluation.
    """
    import select
    import time

    stream = proc.stdout
    assert stream is not None
    fd = stream.fileno()
    deadline = time.monotonic() + max(0.05, float(seconds))

    def _read_exact(n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _w, _x = select.select([fd], [], [], remaining)
            if not ready:
                return None
            # os.read on the raw descriptor, never stream.read: any buffering between select()
            # and the read reintroduces the deadlock bufsize=0 exists to prevent.
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    header = _read_exact(4)
    if header is None:
        return None
    return _read_exact(struct.unpack("<I", header)[0])


def run_guarded(src: str, fn_name: str, calls: Sequence[Sequence[Any]], *,
                limits: Optional[SandboxLimits] = None) -> SandboxResult:
    """One-shot convenience wrapper. Prefer :class:`GuardedExecutor` in a loop — it reuses the
    child, which is the difference between ~2 ms and ~20 ms per item."""
    with GuardedExecutor(limits) as ex:
        return ex.call(src, fn_name, calls)
