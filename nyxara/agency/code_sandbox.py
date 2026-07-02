"""NYXARA · agency/code_sandbox.py — sandboxed code & shell execution facility (⚙, fail-closed).

A mind that can *write* code is only useful if it can also *run* it — but running
arbitrary code is the single most dangerous thing an agent can do. This module is the
disciplined doorway: it executes Python and shell commands **outside the host process**,
under a **wall-clock timeout**, in an **isolated interpreter**, with captured output and
a minimal environment — so a runaway loop, a crashing script, or a misbehaving command
becomes *returned data*, never a hung or corrupted host.

* :func:`run_python` runs a snippet in a **separate subprocess** via
  ``subprocess.run([sys.executable, "-I", "-c", wrapper])`` (``-I`` = isolated mode:
  no site packages from the user, no ``PYTHON*`` env, no implicit ``sys.path[0]``). The
  snippet runs in a fresh temp cwd with a minimal env. When ``allow_network`` is False a
  small bootstrap monkeypatches ``socket.socket`` to raise, so *casual* network use
  fails — **best-effort only, not a true jail**. A conventional ``result`` variable (or
  the value of the last expression) is emitted on a sentinel line and surfaced as
  :attr:`SandboxResult.value`.
* :func:`safe_shell` runs an OS command with ``shell=False`` (a list runs verbatim; a
  string is ``shlex.split``), under a timeout, with captured output. It is the building
  block for an owner-gated shell tool — it just runs and returns data. Because it never
  invokes a shell it also never *interprets* one: pipes, redirection, globbing,
  ``&&``/``;`` chaining and ``$VAR`` expansion are not available, by design.
* :func:`run_shell_command` is the **real terminal**: it runs a command string through an
  actual shell (``bash``/``sh``, ``shell=True``), so every shell feature works — pipes,
  redirection, chaining, globbing, tilde/brace/``$VAR`` expansion, ``$(…)`` substitution,
  heredocs. :class:`ShellSession` wraps it into a *persistent* terminal where the working
  directory (``cd``) and exported environment (``export``) carry across calls, exactly like
  a real terminal tab. Both keep the module's fail-closed contract (never raise; timeouts,
  non-zero exits and launch failures come back as data) and do **no** permission checking —
  the governing :class:`~nyxara.agency.tools.ToolRegistry` still gates them behind
  :class:`~nyxara.agency.permissions.Capability.PROC_EXEC`.

Neither function *raises* on user-code failure: errors, non-zero exits and timeouts are
all returned in a uniform :class:`SandboxResult`. This module performs **no permission
checks** of its own — it is the executor; the governing :class:`~nyxara.agency.tools.ToolRegistry`
wraps it with the :class:`~nyxara.agency.permissions.Capability.CODE_EXEC` /
:class:`~nyxara.agency.permissions.Capability.PROC_EXEC` gate.

Pure standard library.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "SandboxResult",
    "run_python",
    "safe_shell",
    "run_shell_command",
    "ShellSession",
]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class SandboxResult:
    """The uniform outcome of a sandboxed execution — success or failure, always data."""
    ok: bool
    stdout: str = ""
    stderr: str = ""
    value: Any = None
    returncode: int = 0
    timed_out: bool = False
    duration_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "stdout": self.stdout, "stderr": self.stderr,
                "value": self.value, "returncode": self.returncode,
                "timed_out": self.timed_out, "duration_s": round(self.duration_s, 5),
                "error": self.error}


# --------------------------------------------------------------------------- #
# Python execution (separate, isolated subprocess)
# --------------------------------------------------------------------------- #
# A unique sentinel the parent scans for to recover the user's `result` value. Chosen to
# be vanishingly unlikely to occur in ordinary program output.
_VALUE_SENTINEL = "\x00NYXARA_SANDBOX_VALUE\x00"

# Bootstrap that best-effort disables network sockets before the user code runs.
_NO_NETWORK_BOOTSTRAP = (
    "import socket as _s\n"
    "def _blocked(*a, **k):\n"
    "    raise OSError('network disabled in NYXARA sandbox')\n"
    "_s.socket = _blocked\n"
    "_s.create_connection = _blocked\n"
    "try:\n"
    "    _s.SocketType = None\n"
    "except Exception:\n"
    "    pass\n"
)


def _build_wrapper(code: str, *, allow_network: bool) -> str:
    """Compose the program the child interpreter actually runs.

    The user's code executes in its own namespace; afterwards we look for a ``result``
    variable and, if present, emit it on a sentinel line — JSON when serialisable, else
    its ``repr`` — so the parent can recover :attr:`SandboxResult.value` cleanly without
    contaminating the user's own stdout.
    """
    boot = "" if allow_network else _NO_NETWORK_BOOTSTRAP
    user_code = json.dumps(code)  # safely embed arbitrary source as a string literal
    return (
        f"{boot}"
        "import json as _json, sys as _sys\n"
        f"_ns = {{}}\n"
        f"_src = {user_code}\n"
        "exec(compile(_src, '<nyxara-sandbox>', 'exec'), _ns, _ns)\n"
        "if 'result' in _ns:\n"
        "    _v = _ns['result']\n"
        "    try:\n"
        "        _payload = {'json': True, 'value': _json.loads(_json.dumps(_v))}\n"
        "    except Exception:\n"
        "        _payload = {'json': False, 'value': repr(_v)}\n"
        f"    _sys.stdout.write({_VALUE_SENTINEL!r} + _json.dumps(_payload) + '\\n')\n"
    )


def _minimal_env() -> Dict[str, str]:
    """A deliberately sparse environment for the child (no inherited secrets/config)."""
    env: Dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
    }
    # Keep a tempdir hint if present; otherwise the child uses the system default.
    if "TMPDIR" in os.environ:
        env["TMPDIR"] = os.environ["TMPDIR"]
    return env


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def _extract_value(stdout: str) -> Any:
    """Pull the sentinel-encoded ``result`` value (if any) back out of child stdout.

    Returns the decoded value and leaves the caller to strip the sentinel line from the
    visible stdout. When the sentinel is absent, returns ``None``.
    """
    idx = stdout.rfind(_VALUE_SENTINEL)
    if idx == -1:
        return None
    line_end = stdout.find("\n", idx)
    raw = stdout[idx + len(_VALUE_SENTINEL):(line_end if line_end != -1 else len(stdout))]
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload.get("value")


def run_python(code: str, *, timeout_s: float = 5.0, max_output: int = 100_000,
               allow_network: bool = False) -> SandboxResult:
    """Execute a Python snippet in an isolated subprocess and return its outcome as data.

    The snippet runs under ``python -I`` (isolated mode) in a fresh temp cwd with a
    minimal environment and a wall-clock ``timeout_s`` (the process is killed on
    timeout). If the snippet assigns a ``result`` variable it is surfaced as
    :attr:`SandboxResult.value` (JSON-decoded when serialisable, else its ``repr``).

    With ``allow_network=False`` (default) a small bootstrap disables ``socket`` creation
    so *casual* network use fails — this is **best-effort hardening, not a true jail**.

    This function never raises for user-code errors: a crashing snippet returns
    ``ok=False`` with the traceback in :attr:`SandboxResult.stderr`/:attr:`error`; an
    infinite loop returns ``timed_out=True``. Outputs are truncated to ``max_output``.
    """
    wrapper = _build_wrapper(code, allow_network=allow_network)
    start = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="nyxara-sbx-") as workdir:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", wrapper],
                cwd=workdir,
                env=_minimal_env(),
                capture_output=True,
                text=True,
                timeout=max(0.05, timeout_s),
            )
    except subprocess.TimeoutExpired as exc:  # the child was killed at the deadline
        duration = time.monotonic() - start
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return SandboxResult(
            ok=False, stdout=_truncate(out, max_output), stderr=_truncate(err, max_output),
            value=None, returncode=-9, timed_out=True, duration_s=duration,
            error=f"timed out after {timeout_s:g}s")
    except Exception as exc:  # noqa: BLE001 — launch failures are returned as data too
        return SandboxResult(ok=False, returncode=-1, duration_s=time.monotonic() - start,
                             error=f"{type(exc).__name__}: {exc}")

    duration = time.monotonic() - start
    stdout_raw = proc.stdout or ""
    value = _extract_value(stdout_raw)
    # strip the sentinel line from the user-visible stdout
    sidx = stdout_raw.rfind(_VALUE_SENTINEL)
    if sidx != -1:
        line_end = stdout_raw.find("\n", sidx)
        stdout_clean = stdout_raw[:sidx] + (stdout_raw[line_end + 1:] if line_end != -1 else "")
    else:
        stdout_clean = stdout_raw
    stderr = proc.stderr or ""
    ok = proc.returncode == 0
    error = None if ok else (stderr.strip().splitlines()[-1] if stderr.strip()
                             else f"exited with code {proc.returncode}")
    return SandboxResult(
        ok=ok, stdout=_truncate(stdout_clean, max_output), stderr=_truncate(stderr, max_output),
        value=value, returncode=proc.returncode, timed_out=False, duration_s=duration,
        error=error)


# --------------------------------------------------------------------------- #
# Shell execution (the building block for an owner-gated shell tool)
# --------------------------------------------------------------------------- #
def safe_shell(command: Union[List[str], str], *, timeout_s: float = 10.0,
               cwd: Optional[str] = None, stdin: Optional[str] = None) -> SandboxResult:
    """Run an OS command with ``shell=False`` under a timeout, returning the outcome as data.

    A list runs verbatim (no shell, no glob/word-splitting surprises); a string is
    tokenised with :func:`shlex.split`. Output is captured; a non-zero exit yields
    ``ok=False`` with the exit code; exceeding ``timeout_s`` yields ``timed_out=True``.
    When ``stdin`` is given it is fed to the child's standard input (e.g. to drive a
    freshly-compiled program). Never raises: launch failures (e.g. command not found)
    come back as ``error``.

    This is purely the executor — it does **no** permission checking. Callers must gate
    it behind :class:`~nyxara.agency.permissions.Capability.PROC_EXEC` (owner-exclusive
    territory).
    """
    if isinstance(command, str):
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return SandboxResult(ok=False, returncode=-1, error=f"unparseable command: {exc}")
    else:
        argv = list(command)
    if not argv:
        return SandboxResult(ok=False, returncode=-1, error="empty command")

    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv, shell=False, cwd=cwd, input=stdin, capture_output=True, text=True,
            timeout=max(0.05, timeout_s))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return SandboxResult(ok=False, stdout=out, stderr=err, returncode=-9,
                             timed_out=True, duration_s=time.monotonic() - start,
                             error=f"timed out after {timeout_s:g}s")
    except FileNotFoundError as exc:
        return SandboxResult(ok=False, returncode=-1, duration_s=time.monotonic() - start,
                             error=f"command not found: {exc.filename or argv[0]}")
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(ok=False, returncode=-1, duration_s=time.monotonic() - start,
                             error=f"{type(exc).__name__}: {exc}")

    duration = time.monotonic() - start
    ok = proc.returncode == 0
    error = None if ok else f"exited with code {proc.returncode}"
    return SandboxResult(ok=ok, stdout=proc.stdout or "", stderr=proc.stderr or "",
                         returncode=proc.returncode, timed_out=False, duration_s=duration,
                         error=error)


# --------------------------------------------------------------------------- #
# Real shell execution — an actual terminal (shell=True), unrestricted by design
# --------------------------------------------------------------------------- #
# Unlike safe_shell (shell=False, argv only), this runs the command *through a shell*, so
# it behaves like a real terminal: pipes, redirection, &&/;/|| chaining, globbing, brace/
# tilde/$VAR expansion, $(…) substitution and heredocs all work. It performs NO command
# filtering — the governing ToolRegistry gates it behind Capability.PROC_EXEC.


def _default_shell() -> Optional[str]:
    """The shell to run commands through — bash if present, else sh (POSIX) or COMSPEC (Windows).

    Returns the executable path to hand to ``subprocess`` as ``executable=``; ``None`` lets
    the platform pick its default shell (``/bin/sh`` on POSIX, ``%COMSPEC%`` on Windows) —
    which is a safe fallback rather than a failure.
    """
    if os.name == "nt":
        return os.environ.get("COMSPEC") or None
    for candidate in ("/bin/bash", "/usr/bin/bash", "/bin/sh"):
        if os.path.exists(candidate):
            return candidate
    # Fall back to whatever the platform considers the shell.
    return os.environ.get("SHELL") or None


def _shell_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """A real terminal sees the real environment: inherit ``os.environ``, overlay ``env``."""
    merged = dict(os.environ)
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    return merged


def run_shell_command(command: str, *, timeout_s: float = 30.0,
                      cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                      stdin: Optional[str] = None, max_output: int = 100_000
                      ) -> SandboxResult:
    """Run a command string through a **real shell** and return the outcome as data.

    This is the genuine, unrestricted terminal: because it uses ``shell=True`` the command is
    interpreted by an actual shell (``bash``/``sh``), so pipes, redirection, ``&&``/``;``/``||``
    chaining, globbing, brace/tilde/``$VAR`` expansion, ``$(…)`` command substitution and
    heredocs all work exactly as they would at a prompt. The child inherits the full process
    environment (overlaid with ``env``) and runs in ``cwd`` (the current directory when
    ``None``); ``stdin`` is fed to its standard input when given.

    ``timeout_s`` is a wall-clock deadline (the process is killed on expiry); pass
    ``timeout_s <= 0`` for **no timeout** (an unbounded command). Output is truncated to
    ``max_output``. Never raises: a non-zero exit yields ``ok=False`` with the code, a timeout
    yields ``timed_out=True``, and a launch failure (e.g. no shell found) comes back as
    ``error``. This is purely the executor — it does **no** permission checking; callers must
    gate it behind :class:`~nyxara.agency.permissions.Capability.PROC_EXEC`.
    """
    if not command or not command.strip():
        return SandboxResult(ok=False, returncode=-1, error="empty command")

    timeout = None if timeout_s is not None and timeout_s <= 0 else max(0.05, timeout_s)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command, shell=True, executable=_default_shell(), cwd=cwd,
            env=_shell_env(env), input=stdin, capture_output=True, text=True,
            timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return SandboxResult(ok=False, stdout=_truncate(out, max_output),
                             stderr=_truncate(err, max_output), returncode=-9,
                             timed_out=True, duration_s=time.monotonic() - start,
                             error=f"timed out after {timeout_s:g}s")
    except Exception as exc:  # noqa: BLE001 — launch failures are returned as data too
        return SandboxResult(ok=False, returncode=-1, duration_s=time.monotonic() - start,
                             error=f"{type(exc).__name__}: {exc}")

    duration = time.monotonic() - start
    ok = proc.returncode == 0
    error = None if ok else f"exited with code {proc.returncode}"
    return SandboxResult(ok=ok, stdout=_truncate(proc.stdout or "", max_output),
                         stderr=_truncate(proc.stderr or "", max_output),
                         returncode=proc.returncode, timed_out=False,
                         duration_s=duration, error=error)


# --------------------------------------------------------------------------- #
# Persistent terminal session — cd / export carry across calls, like a real tab
# --------------------------------------------------------------------------- #
# Sentinels the parent scans for to recover the child's final cwd + environment. They are
# emitted via ``printf '%s' '<token>'`` inside the shell, so they must be *printable* (a NUL
# byte cannot survive a shell command line); the long unique suffix makes an accidental
# collision with ordinary command output vanishingly unlikely.
_CWD_SENTINEL = "__NYXARA_CWD_9f3a1c7e__"
_ENV_SENTINEL = "__NYXARA_ENV_9f3a1c7e__"
_ENV_END_SENTINEL = "__NYXARA_ENV_END_9f3a1c7e__"


class ShellSession:
    """A persistent real-terminal session: working directory and env survive between calls.

    Each :meth:`run` executes the command through :func:`run_shell_command`, but wraps it so
    the child reports its *final* ``pwd`` and environment on sentinel lines afterward. Those
    lines are parsed out (and stripped from the visible stdout) to update the session's
    ``cwd``/``env``, so ``cd /tmp`` in one call and ``pwd`` in the next behaves exactly like a
    real terminal tab — without holding a fragile long-lived process open. The returned
    :attr:`SandboxResult.returncode` is the real ``$?`` of the *user's* command, captured
    before the reporting runs.

    Does no permission checking — the governing registry gates it behind ``PROC_EXEC``.
    """

    def __init__(self, *, cwd: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None) -> None:
        self.cwd: str = cwd or os.getcwd()
        # Session env starts from the real environment (overlaid with any seed) and is
        # updated from the child's reported env after each command.
        self.env: Dict[str, str] = _shell_env(env)

    def reset(self, *, cwd: Optional[str] = None) -> None:
        """Reset the session to a fresh cwd and a clean inherited environment."""
        self.cwd = cwd or os.getcwd()
        self.env = _shell_env(None)

    def run(self, command: str, *, timeout_s: float = 30.0,
            stdin: Optional[str] = None, max_output: int = 100_000) -> SandboxResult:
        """Run ``command`` in the session, persisting any ``cd``/``export`` it performs."""
        if not command or not command.strip():
            return SandboxResult(ok=False, returncode=-1, error="empty command")

        # Run the user's command, capture its exit status, then emit the final cwd + env on
        # sentinel lines so the session can track them. `env -0` / `\0`-delimited names would
        # be cleaner, but a simple KEY=VALUE dump keyed by sentinels is portable across sh/bash.
        wrapped = (
            f"{command}\n"
            "__nyxara_rc=$?\n"
            f"printf '%s' {_CWD_SENTINEL!r}; pwd; "
            f"printf '%s' {_ENV_SENTINEL!r}; env; printf '%s' {_ENV_END_SENTINEL!r}\n"
            "exit $__nyxara_rc\n"
        )
        res = run_shell_command(wrapped, timeout_s=timeout_s, cwd=self.cwd,
                                env=self.env, stdin=stdin, max_output=max_output + 65_536)

        stdout = res.stdout or ""
        # Parse + strip the trailing cwd/env report; update session state from it.
        cidx = stdout.rfind(_CWD_SENTINEL)
        if cidx != -1:
            tail = stdout[cidx:]
            stdout = stdout[:cidx]
            eidx = tail.find(_ENV_SENTINEL)
            if eidx != -1:
                new_cwd = tail[len(_CWD_SENTINEL):eidx].strip()
                env_end = tail.find(_ENV_END_SENTINEL, eidx)
                env_blob = tail[eidx + len(_ENV_SENTINEL):(env_end if env_end != -1 else len(tail))]
                if new_cwd:
                    self.cwd = new_cwd
                self._absorb_env(env_blob)

        res.stdout = _truncate(stdout, max_output)
        return res

    def _absorb_env(self, env_blob: str) -> None:
        """Replace the session env from a ``env``-style ``KEY=VALUE`` dump (best-effort)."""
        new_env: Dict[str, str] = {}
        cur_key: Optional[str] = None
        for line in env_blob.split("\n"):
            if "=" in line and not line.startswith((" ", "\t")):
                key, _, val = line.partition("=")
                if key and all(c.isalnum() or c == "_" for c in key):
                    cur_key = key
                    new_env[key] = val
                    continue
            # A continuation line of a multi-line value (e.g. a var containing newlines).
            if cur_key is not None:
                new_env[cur_key] += "\n" + line
        if new_env:
            self.env = new_env


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA code-sandbox self-test")
    print("=" * 70)

    # 1. a simple compute, surfaced via the `result` convention
    r = run_python("result = 2 + 2")
    print(f"\ncompute             : ok={r.ok} value={r.value!r} duration={r.duration_s:.3f}s")
    assert r.ok and r.value == 4

    # 1b. stdout is captured; a non-serialisable result falls back to repr
    r = run_python("print('hello world')\nresult = set([1, 2, 3])")
    print(f"stdout+repr value   : stdout={r.stdout.strip()!r} value={r.value!r}")
    assert r.ok and "hello world" in r.stdout and r.value.startswith("{")

    # 2. a deliberate exception is captured — never raised
    r = run_python("raise ValueError('boom')")
    print(f"\ncaught exception    : ok={r.ok} error={r.error!r}")
    assert not r.ok and r.error is not None and "ValueError" in r.stderr

    # 3. an infinite loop times out fast (short timeout keeps the test quick)
    r = run_python("while True:\n    pass", timeout_s=1.0)
    print(f"timeout             : ok={r.ok} timed_out={r.timed_out} error={r.error!r}")
    assert not r.ok and r.timed_out

    # 4. output truncation
    r = run_python("print('x' * 5000)", max_output=200)
    print(f"truncation          : stdout_len={len(r.stdout)} (cap 200 + marker)")
    assert "truncated" in r.stdout and len(r.stdout) < 5000

    # 5. best-effort network block (casual socket use fails when disabled)
    r = run_python("import socket; socket.socket()", allow_network=False)
    print(f"\nnetwork blocked     : ok={r.ok} (socket creation refused)")
    assert not r.ok and "network disabled" in r.stderr

    # 6. safe_shell: echo round-trips stdout
    r = safe_shell(["echo", "hello"])
    print(f"\nshell echo          : ok={r.ok} stdout={r.stdout.strip()!r}")
    assert r.ok and "hello" in r.stdout

    # 6b. a string command is shlex-split
    r = safe_shell("echo from a string")
    assert r.ok and "from a string" in r.stdout
    print(f"shell str command   : stdout={r.stdout.strip()!r}")

    # 7. a non-zero exit is captured as data
    r = safe_shell(["false"])
    print(f"nonzero exit         : ok={r.ok} returncode={r.returncode}")
    assert not r.ok and r.returncode != 0

    # 8. a too-long sleep times out under a small deadline
    r = safe_shell(["sleep", "5"], timeout_s=0.5)
    print(f"shell timeout        : ok={r.ok} timed_out={r.timed_out}")
    assert not r.ok and r.timed_out

    # 9. a missing command comes back as error, not an exception
    r = safe_shell(["nyxara-no-such-binary-xyz"])
    print(f"missing command      : ok={r.ok} error={r.error!r}")
    assert not r.ok and r.error is not None

    # 10. real shell: a pipe (which safe_shell cannot do) works verbatim
    r = run_shell_command("echo hello | tr a-z A-Z")
    print(f"\nreal shell pipe      : ok={r.ok} stdout={r.stdout.strip()!r}")
    assert r.ok and "HELLO" in r.stdout

    # 11. real shell: chaining + redirection + $VAR expansion, all in one line
    r = run_shell_command("X=nyx && echo $X > /dev/stdout && echo done")
    print(f"real shell chain     : ok={r.ok} stdout={r.stdout.split()!r}")
    assert r.ok and "nyx" in r.stdout and "done" in r.stdout

    # 12. persistent session: cd in one call is visible in the next (real terminal)
    s = ShellSession()
    s.run("cd /")
    r = s.run("pwd")
    print(f"session cd persists  : cwd={s.cwd!r} pwd={r.stdout.strip()!r}")
    assert r.stdout.strip() == "/" and s.cwd == "/"

    # 13. persistent session: export in one call is visible in the next
    s2 = ShellSession()
    s2.run("export NYXARA_SELFTEST=42")
    r = s2.run("echo $NYXARA_SELFTEST")
    print(f"session export       : value={r.stdout.strip()!r}")
    assert r.stdout.strip() == "42"

    print("\nALL SELF-TESTS PASSED ✓")
