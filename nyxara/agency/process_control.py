"""NYXARA · agency/process_control.py — live process & job control + system introspection (🕹).

``run_shell`` (in :mod:`~nyxara.agency.code_sandbox`) is a *one-shot* terminal: it runs a
command to completion and hands back the output. That is exactly wrong for the other half of
real machine control — the long-running, interactive world: a dev server, a training run,
``top``, an ``ssh`` session, anything you start and then *keep talking to*. This module is that
half. It gives NYXARA a genuine **job table**: she can launch a process in the background, watch
its output stream in, feed it standard input, signal it, and kill it — all across separate turns
of the sovereign loop, exactly like a human juggling terminal tabs.

Three layers, all pure standard library:

* :class:`ProcessManager` — a process-lifetime registry of live jobs keyed by an integer
  ``handle``. :meth:`~ProcessManager.spawn` starts a background process (its own session/group so
  a signal reaps the whole tree); reader threads drain stdout/stderr into bounded ring buffers so
  output never blocks the child. :meth:`~ProcessManager.read` streams new output,
  :meth:`~ProcessManager.write` feeds stdin, :meth:`~ProcessManager.signal` /
  :meth:`~ProcessManager.kill` terminate, :meth:`~ProcessManager.wait` blocks for exit. With
  ``tty=True`` the child is attached to a real pseudo-terminal (:func:`pty.openpty`), so
  interactive / ncurses programs that demand a TTY work too.
* System introspection: :func:`system_info` (cpu / load / memory / uptime / disk / identity),
  :func:`list_processes` (the process table), :func:`kill_pid` / :func:`kill_by_name`.

Like :mod:`code_sandbox` and :mod:`privilege`, this module does **no permission checking** — it
is the executor. Every caller MUST gate it behind
:class:`~nyxara.agency.permissions.Capability.PROC_EXEC` (owner / full_control). The kernel's
``/scram`` kill-switch, oversight and corrigibility gates all sit above it and remain intact.
Nothing here ever raises across its public surface: failures come back as structured data.
"""

from __future__ import annotations

import atexit
import os
import platform
import signal as _signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.agency.code_sandbox import _default_shell, _shell_env, safe_shell

__all__ = [
    "ProcessManager",
    "get_process_manager",
    "system_info",
    "list_processes",
    "kill_pid",
    "kill_by_name",
]

_IS_POSIX = os.name == "posix"

# Default ceiling on how much of each stream we retain in memory per job (bytes). Output beyond
# this is dropped from the *front* (oldest first) — a ring buffer — so a chatty long-runner can
# stream forever without exhausting memory. `read` reports how many bytes were dropped.
_DEFAULT_BUFFER_CAP = 1_000_000


# --------------------------------------------------------------------------- #
# Bounded, cursor-tracked stream buffer
# --------------------------------------------------------------------------- #
class _Stream:
    """A bounded byte buffer with a read cursor, so ``read_new`` yields only fresh output.

    The buffer keeps at most ``cap`` bytes; older bytes are dropped once that is exceeded. A
    logical cursor tracks absolute position in the *whole* stream (including dropped bytes), so
    successive :meth:`read_new` calls return exactly what arrived since the previous read, and
    ``dropped`` reports any output that scrolled past the cap before it was read.
    """

    __slots__ = ("buf", "cap", "dropped", "cursor")

    def __init__(self, cap: int) -> None:
        self.buf = bytearray()
        self.cap = max(4096, cap)
        self.dropped = 0          # total bytes evicted from the front
        self.cursor = 0           # absolute position already handed out by read_new

    def append(self, data: bytes) -> None:
        self.buf += data
        overflow = len(self.buf) - self.cap
        if overflow > 0:
            del self.buf[:overflow]
            self.dropped += overflow

    @property
    def total(self) -> int:
        return self.dropped + len(self.buf)

    def read_new(self) -> bytes:
        start = self.cursor - self.dropped
        if start < 0:              # the cursor pointed at bytes we have since evicted
            start = 0
        out = bytes(self.buf[start:])
        self.cursor = self.total
        return out

    def snapshot(self) -> bytes:
        return bytes(self.buf)


# --------------------------------------------------------------------------- #
# One live job
# --------------------------------------------------------------------------- #
@dataclass
class _Job:
    handle: int
    command: str
    proc: subprocess.Popen
    tty: bool
    started_at: float
    master_fd: Optional[int] = None          # pty master (tty jobs only)
    out: _Stream = field(default_factory=lambda: _Stream(_DEFAULT_BUFFER_CAP))
    err: _Stream = field(default_factory=lambda: _Stream(_DEFAULT_BUFFER_CAP))
    lock: threading.Lock = field(default_factory=threading.Lock)
    readers: List[threading.Thread] = field(default_factory=list)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def poll(self) -> Optional[int]:
        return self.proc.poll()


def _decode(data: bytes) -> str:
    return data.decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# The manager
# --------------------------------------------------------------------------- #
class ProcessManager:
    """A process-lifetime table of live background jobs, driven across many turns.

    Every method returns plain data and never raises: a bad handle, a dead process, or an OS
    error comes back as ``{"ok": False, "error": ...}``. Does no permission checking — the
    governing :class:`~nyxara.agency.tools.ToolRegistry` gates it behind ``PROC_EXEC``.
    """

    def __init__(self, *, buffer_cap: int = _DEFAULT_BUFFER_CAP) -> None:
        self._jobs: Dict[int, _Job] = {}
        self._next = 1
        self._lock = threading.Lock()
        self._buffer_cap = max(4096, buffer_cap)

    # ---- spawn ---- #
    def spawn(self, command: str, *, tty: bool = False, cwd: Optional[str] = None,
              env: Optional[Dict[str, str]] = None, stdin: Optional[str] = None) -> Dict[str, Any]:
        """Launch ``command`` in a real shell as a background job; return its handle + pid.

        The child runs through ``bash``/``sh`` (so pipes, chaining, redirection and ``$VAR``
        expansion all work) in its **own session/process group**, so a later signal reaps its
        whole subtree. Output is captured asynchronously into ring buffers — the child never
        blocks on a full pipe. ``tty=True`` attaches it to a pseudo-terminal for programs that
        require a TTY. Optional ``stdin`` is written once at launch (a convenience for a
        non-interactive feed; use :meth:`write` for ongoing input).
        """
        if not command or not command.strip():
            return {"ok": False, "error": "empty command"}
        try:
            if tty:
                return self._spawn_tty(command, cwd=cwd, env=env, stdin=stdin)
            return self._spawn_pipes(command, cwd=cwd, env=env, stdin=stdin)
        except Exception as exc:  # noqa: BLE001 — launch failures are data, not exceptions
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _new_handle(self) -> int:
        with self._lock:
            h = self._next
            self._next += 1
            return h

    def _register(self, job: _Job) -> None:
        with self._lock:
            self._jobs[job.handle] = job

    def _popen_kwargs(self, cwd: Optional[str], env: Optional[Dict[str, str]]) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(shell=True, executable=_default_shell(), cwd=cwd,
                                  env=_shell_env(env))
        # A fresh session makes the child a group leader (pgid == pid) so killpg reaps its tree.
        if _IS_POSIX:
            kw["start_new_session"] = True
        return kw

    def _spawn_pipes(self, command: str, *, cwd, env, stdin) -> Dict[str, Any]:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, **self._popen_kwargs(cwd, env))
        job = _Job(handle=self._new_handle(), command=command, proc=proc, tty=False,
                   started_at=time.time(),
                   out=_Stream(self._buffer_cap), err=_Stream(self._buffer_cap))
        job.readers.append(self._start_pipe_reader(job, proc.stdout, job.out))
        job.readers.append(self._start_pipe_reader(job, proc.stderr, job.err))
        self._register(job)
        if stdin:
            self.write(job.handle, stdin)
        return {"ok": True, "handle": job.handle, "pid": job.pid, "tty": False}

    def _spawn_tty(self, command: str, *, cwd, env, stdin) -> Dict[str, Any]:
        if not _IS_POSIX:
            return {"ok": False, "error": "tty jobs require a POSIX pseudo-terminal (pty)"}
        import pty
        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(command, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                                    **self._popen_kwargs(cwd, env))
        finally:
            os.close(slave_fd)     # the child holds its own copy; the parent uses the master
        job = _Job(handle=self._new_handle(), command=command, proc=proc, tty=True,
                   started_at=time.time(), master_fd=master_fd,
                   out=_Stream(self._buffer_cap), err=_Stream(self._buffer_cap))
        job.readers.append(self._start_fd_reader(job, master_fd, job.out))
        self._register(job)
        if stdin:
            self.write(job.handle, stdin)
        return {"ok": True, "handle": job.handle, "pid": job.pid, "tty": True}

    # ---- reader threads ---- #
    def _start_pipe_reader(self, job: _Job, pipe, stream: _Stream) -> threading.Thread:
        fd = pipe.fileno()

        def _drain() -> None:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                with job.lock:
                    stream.append(chunk)
            try:
                pipe.close()
            except Exception:  # noqa: BLE001
                pass

        t = threading.Thread(target=_drain, daemon=True, name=f"nyx-job{job.handle}")
        t.start()
        return t

    def _start_fd_reader(self, job: _Job, fd: int, stream: _Stream) -> threading.Thread:
        def _drain() -> None:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:      # EIO on Linux when the child closes the pty slave — EOF
                    break
                if not chunk:
                    break
                with job.lock:
                    stream.append(chunk)

        t = threading.Thread(target=_drain, daemon=True, name=f"nyx-job{job.handle}-pty")
        t.start()
        return t

    # ---- introspection ---- #
    def _get(self, handle: int) -> Optional[_Job]:
        with self._lock:
            return self._jobs.get(handle)

    def _job_status(self, job: _Job) -> Dict[str, Any]:
        rc = job.poll()
        with job.lock:
            out_total, err_total = job.out.total, job.err.total
            out_dropped, err_dropped = job.out.dropped, job.err.dropped
        return {
            "handle": job.handle, "pid": job.pid, "command": job.command, "tty": job.tty,
            "running": rc is None, "returncode": rc, "started_at": job.started_at,
            "age_s": round(time.time() - job.started_at, 3),
            "stdout_bytes": out_total, "stderr_bytes": err_total,
            "stdout_dropped": out_dropped, "stderr_dropped": err_dropped,
        }

    def list_jobs(self) -> Dict[str, Any]:
        """Every job this manager has spawned, running or exited, newest last."""
        with self._lock:
            jobs = list(self._jobs.values())
        return {"ok": True, "jobs": [self._job_status(j) for j in jobs],
                "count": len(jobs)}

    # ---- stream I/O ---- #
    def read(self, handle: int, *, max_bytes: int = 100_000) -> Dict[str, Any]:
        """Return output that has arrived since the previous read, plus the job's status.

        Non-blocking and incremental: repeated calls stream the child's output over time. For a
        pty job stderr is folded into stdout (a terminal has one output stream).
        """
        job = self._get(handle)
        if job is None:
            return {"ok": False, "error": f"no such job: {handle}"}
        with job.lock:
            out = job.out.read_new()
            err = b"" if job.tty else job.err.read_new()
        status = self._job_status(job)
        status.update({
            "ok": True,
            "stdout": _decode(out[-max_bytes:] if max_bytes > 0 else out),
            "stderr": _decode(err[-max_bytes:] if max_bytes > 0 else err),
        })
        return status

    def write(self, handle: int, data: str, *, newline: bool = False) -> Dict[str, Any]:
        """Feed ``data`` to a running job's standard input (or its pty)."""
        job = self._get(handle)
        if job is None:
            return {"ok": False, "error": f"no such job: {handle}"}
        if job.poll() is not None:
            return {"ok": False, "error": "job has already exited", "returncode": job.poll()}
        payload = data + ("\n" if newline and not data.endswith("\n") else "")
        raw = payload.encode("utf-8", "replace")
        try:
            if job.tty and job.master_fd is not None:
                os.write(job.master_fd, raw)
            elif job.proc.stdin is not None:
                job.proc.stdin.write(raw)
                job.proc.stdin.flush()
            else:
                return {"ok": False, "error": "job has no writable stdin"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "handle": handle, "wrote_bytes": len(raw)}

    # ---- signalling ---- #
    def signal(self, handle: int, sig: int = _signal.SIGTERM) -> Dict[str, Any]:
        """Send ``sig`` to the job's whole process group (POSIX) or the process (else)."""
        job = self._get(handle)
        if job is None:
            return {"ok": False, "error": f"no such job: {handle}"}
        if job.poll() is not None:
            return {"ok": True, "handle": handle, "already_exited": True,
                    "returncode": job.poll()}
        try:
            if _IS_POSIX:
                os.killpg(os.getpgid(job.pid), sig)
            else:
                job.proc.send_signal(sig)
        except ProcessLookupError:
            return {"ok": True, "handle": handle, "already_exited": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "handle": handle, "signalled": int(sig)}

    def kill(self, handle: int, *, timeout_s: float = 5.0) -> Dict[str, Any]:
        """Terminate a job: SIGTERM, wait briefly, then SIGKILL if it clings on."""
        job = self._get(handle)
        if job is None:
            return {"ok": False, "error": f"no such job: {handle}"}
        self.signal(handle, _signal.SIGTERM)
        rc = self._wait_proc(job, timeout_s)
        if rc is None:                       # still alive — escalate
            self.signal(handle, getattr(_signal, "SIGKILL", _signal.SIGTERM))
            rc = self._wait_proc(job, timeout_s)
        if rc is not None:
            self._join_readers(job)
        return {"ok": True, "handle": handle, "returncode": rc,
                "killed": rc is not None}

    def wait(self, handle: int, *, timeout_s: float = 30.0) -> Dict[str, Any]:
        """Block up to ``timeout_s`` for the job to exit; report its return code.

        Once the process has exited, the reader threads are joined so **all** of its output is
        captured before returning — a subsequent :meth:`read` deterministically sees the final
        bytes (important for a pty, whose slave-close can otherwise race the last write).
        """
        job = self._get(handle)
        if job is None:
            return {"ok": False, "error": f"no such job: {handle}"}
        rc = self._wait_proc(job, None if timeout_s <= 0 else timeout_s)
        if rc is not None:
            self._join_readers(job)
        status = self._job_status(job)
        status.update({"ok": True, "timed_out": rc is None})
        return status

    @staticmethod
    def _wait_proc(job: _Job, timeout: Optional[float]) -> Optional[int]:
        try:
            return job.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        except Exception:  # noqa: BLE001
            return job.poll()

    @staticmethod
    def _join_readers(job: _Job, timeout_s: float = 1.0) -> None:
        """Wait (bounded) for the drain threads to finish so buffered output is fully captured."""
        deadline = time.time() + timeout_s
        for t in job.readers:
            remaining = deadline - time.time()
            if remaining > 0:
                t.join(remaining)

    # ---- teardown ---- #
    def shutdown(self) -> None:
        """Best-effort: SIGKILL every still-running job (called at interpreter exit)."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.poll() is None:
                try:
                    if _IS_POSIX:
                        os.killpg(os.getpgid(job.pid),
                                  getattr(_signal, "SIGKILL", _signal.SIGTERM))
                    else:
                        job.proc.kill()
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------- #
# Process-lifetime singleton
# --------------------------------------------------------------------------- #
_MANAGER: Optional[ProcessManager] = None
_MANAGER_LOCK = threading.Lock()


def get_process_manager() -> ProcessManager:
    """The shared, process-lifetime :class:`ProcessManager` (created on first use)."""
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = ProcessManager()
                atexit.register(_MANAGER.shutdown)
    return _MANAGER


# --------------------------------------------------------------------------- #
# System introspection — cpu / memory / uptime / disk / identity
# --------------------------------------------------------------------------- #
def _read_proc_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _meminfo() -> Dict[str, int]:
    """Parse ``/proc/meminfo`` into a KiB→bytes map for the common fields (Linux)."""
    text = _read_proc_file("/proc/meminfo")
    out: Dict[str, int] = {}
    if not text:
        return out
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            # values are in kB
            out[key.strip()] = int(parts[0]) * 1024
    return out


def system_info() -> Dict[str, Any]:
    """Report the machine's live posture — cpu, load, memory, uptime, disk, identity.

    Read-only and best-effort: every field is wrapped so a missing ``/proc`` or an odd platform
    degrades to ``None`` rather than failing. Never raises.
    """
    import shutil
    import socket

    info: Dict[str, Any] = {
        "ok": True,
        "platform": platform.platform(),
        "system": platform.system(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "pid": os.getpid(),
    }
    geteuid = getattr(os, "geteuid", None)
    info["euid"] = geteuid() if geteuid is not None else None
    info["is_root"] = info["euid"] == 0

    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is not None:
        try:
            info["loadavg"] = [round(x, 2) for x in getloadavg()]
        except OSError:
            info["loadavg"] = None
    else:
        info["loadavg"] = None

    mem = _meminfo()
    if mem:
        info["mem_total_bytes"] = mem.get("MemTotal")
        info["mem_available_bytes"] = mem.get("MemAvailable", mem.get("MemFree"))

    uptime_txt = _read_proc_file("/proc/uptime")
    if uptime_txt:
        try:
            up = float(uptime_txt.split()[0])
            info["uptime_s"] = round(up, 1)
            info["boot_time"] = round(time.time() - up, 1)
        except (ValueError, IndexError):
            pass

    try:
        du = shutil.disk_usage("/")
        info["disk_total_bytes"] = du.total
        info["disk_free_bytes"] = du.free
    except OSError:
        pass
    return info


# --------------------------------------------------------------------------- #
# The process table
# --------------------------------------------------------------------------- #
def _uid_to_name(uid: int) -> Optional[str]:
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except Exception:  # noqa: BLE001 — no pwd (Windows) or unknown uid
        return None


def _proc_one(pid: str) -> Optional[Dict[str, Any]]:
    """Read one ``/proc/<pid>`` entry into a record; ``None`` if it vanished mid-scan."""
    base = f"/proc/{pid}"
    cmdline = _read_proc_file(f"{base}/cmdline")
    if cmdline is None:
        return None
    command = cmdline.replace("\x00", " ").strip()
    if not command:
        comm = _read_proc_file(f"{base}/comm") or ""
        command = f"[{comm.strip()}]"
    rec: Dict[str, Any] = {"pid": int(pid), "command": command[:400]}
    status = _read_proc_file(f"{base}/status") or ""
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                rec["ppid"] = int(line.split()[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("Uid:"):
            try:
                uid = int(line.split()[1])
                rec["uid"] = uid
                rec["user"] = _uid_to_name(uid)
            except (ValueError, IndexError):
                pass
        elif line.startswith("VmRSS:"):
            try:
                rec["rss_bytes"] = int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                pass
    return rec


def list_processes(*, limit: int = 500) -> Dict[str, Any]:
    """The system process table: pid, ppid, user, rss and command line for each process.

    Reads ``/proc`` on Linux; where that is unavailable it falls back to ``ps`` via
    :func:`~nyxara.agency.code_sandbox.safe_shell`. Read-only, never raises.
    """
    procs: List[Dict[str, Any]] = []
    if os.path.isdir("/proc"):
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            rec = _proc_one(name)
            if rec is not None:
                procs.append(rec)
            if len(procs) >= limit:
                break
        procs.sort(key=lambda r: r.get("rss_bytes", 0), reverse=True)
        return {"ok": True, "count": len(procs), "processes": procs[:limit], "source": "/proc"}

    # Fallback: parse `ps`.
    res = safe_shell(["ps", "-eo", "pid,ppid,user,rss,args"], timeout_s=10.0)
    if not res.ok:
        return {"ok": False, "error": res.error or "ps failed", "processes": []}
    lines = res.stdout.splitlines()[1:]
    for line in lines[:limit]:
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, user, rss, args = parts
        try:
            procs.append({"pid": int(pid), "ppid": int(ppid), "user": user,
                          "rss_bytes": int(rss) * 1024, "command": args[:400]})
        except ValueError:
            continue
    return {"ok": True, "count": len(procs), "processes": procs, "source": "ps"}


def kill_pid(pid: int, *, sig: int = _signal.SIGTERM) -> Dict[str, Any]:
    """Send signal ``sig`` (default SIGTERM) to a single process id. Never raises."""
    try:
        os.kill(int(pid), int(sig))
    except ProcessLookupError:
        return {"ok": False, "error": f"no such process: {pid}", "pid": int(pid)}
    except PermissionError:
        return {"ok": False, "error": f"not permitted to signal pid {pid} "
                                      "(needs privilege_escalation / root)", "pid": int(pid)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "pid": int(pid)}
    return {"ok": True, "pid": int(pid), "signalled": int(sig)}


def kill_by_name(pattern: str, *, sig: int = _signal.SIGTERM,
                 include_self: bool = False) -> Dict[str, Any]:
    """Signal every process whose command line contains ``pattern`` (substring match).

    Skips this process (and the ``list_processes`` scan is a snapshot) unless ``include_self``.
    Returns each matched pid and the per-pid result. Never raises.
    """
    if not pattern:
        return {"ok": False, "error": "empty pattern", "matched": []}
    table = list_processes()
    me = os.getpid()
    results: List[Dict[str, Any]] = []
    for rec in table.get("processes", []):
        if pattern not in rec.get("command", ""):
            continue
        if rec["pid"] == me and not include_self:
            continue
        results.append(kill_pid(rec["pid"], sig=sig))
    matched = [r["pid"] for r in results]
    return {"ok": True, "pattern": pattern, "signalled": int(sig),
            "matched": matched, "count": len(matched), "results": results}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA process-control self-test")
    print("=" * 70)

    pm = ProcessManager()

    # 1. spawn a background emitter, stream its output, confirm it finishes on its own.
    r = pm.spawn("for i in 0 1 2 3 4; do echo tick $i; sleep 0.1; done")
    print(f"\nspawn                : ok={r['ok']} handle={r.get('handle')} pid={r.get('pid')}")
    assert r["ok"]
    h = r["handle"]
    done = pm.wait(h, timeout_s=10.0)
    seen = pm.read(h)
    print(f"streamed stdout      : {seen['stdout'].split()!r}")
    print(f"exit                 : returncode={done['returncode']} timed_out={done['timed_out']}")
    assert done["returncode"] == 0 and "tick" in seen["stdout"]

    # 2. interactive stdin: `cat` echoes what we write, then EOF-free kill.
    c = pm.spawn("cat")
    ch = c["handle"]
    pm.write(ch, "hello nyxara\n")
    time.sleep(0.2)
    echoed = pm.read(ch)
    print(f"\ninteractive cat      : stdout={echoed['stdout'].strip()!r} running={echoed['running']}")
    assert "hello nyxara" in echoed["stdout"] and echoed["running"]
    killed = pm.kill(ch)
    print(f"kill cat             : returncode={killed['returncode']} killed={killed['killed']}")
    assert pm.read(ch)["running"] is False

    # 3. signal a long sleeper — the whole group dies.
    s = pm.spawn("sleep 300")
    sh = s["handle"]
    assert pm.read(sh)["running"]
    pm.signal(sh, _signal.SIGTERM)
    st = pm.wait(sh, timeout_s=5.0)
    print(f"\nsignal sleep         : running_after={st['running']} rc={st['returncode']}")
    assert st["running"] is False

    # 4. list_jobs sees all three.
    jobs = pm.list_jobs()
    print(f"list_jobs            : count={jobs['count']}")
    assert jobs["count"] == 3

    # 5. pty job: a real terminal echoes typed input back on the master.
    if _IS_POSIX:
        t = pm.spawn("read x; echo \"$x\" | tr a-z A-Z", tty=True)
        assert t["ok"] and t["tty"]
        th = t["handle"]
        pm.write(th, "shout\n")
        pm.wait(th, timeout_s=5.0)
        pty_out = pm.read(th)
        print(f"\npty echo             : {pty_out['stdout'].strip()!r}")
        assert "SHOUT" in pty_out["stdout"]

    # 6. system_info + process table + our own pid is present.
    info = system_info()
    print(f"\nsystem_info          : cpu={info['cpu_count']} load={info.get('loadavg')} "
          f"mem_total={info.get('mem_total_bytes')} root={info['is_root']}")
    assert info["ok"] and info["cpu_count"]
    table = list_processes()
    pids = {p["pid"] for p in table["processes"]}
    print(f"list_processes       : count={table['count']} source={table['source']} "
          f"self_present={os.getpid() in pids}")
    assert table["ok"] and table["count"] > 0

    pm.shutdown()
    print("\nALL SELF-TESTS PASSED ✓")
