"""NYXARA · mind/runtime_log.py — read what a native runtime says about its own failures.

``mind/llm.py::LiteRTLMProvider`` sits on a C++ library, and when a turn fails that library raises
the least useful sentence available::

    RuntimeError: litert_lm_conversation_send_message failed

The actual reason — ``INTERNAL: Failed to apply template: unknown method: map has no method named
get`` — is written by the runtime's own logger straight to **file descriptor 2**, bypassing Python
entirely. It never reaches the exception, the logs, or a bug report. Two rounds of this were spent
guessing at a machine that could have said what was wrong all along.

Redirecting ``sys.stderr`` does not help: the logger resolves fd 2 **once, when the shared library
is first loaded**, and writes there forever after. So the capture has to be installed *before* the
library loads and stay installed — which is what :func:`install` does.

It is a **tee, not a hijack**: a pipe takes fd 2, a daemon thread drains it into a bounded ring
buffer *and* writes every line on to the real stderr. Nothing is swallowed, nothing is reordered,
and a caller can ask :func:`recent` what the runtime said in the last moments before it failed.

Best-effort throughout. On a platform where this cannot work (no ``os.pipe``, no ``dup2``) it
installs nothing and :func:`recent` returns an empty list — the caller simply gets the opaque
message it would have got anyway.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import Deque, List

__all__ = ["install", "recent", "interesting"]

_LOCK = threading.Lock()
_LINES: Deque[str] = deque(maxlen=200)
_INSTALLED = False

# Substrings that mark a line as worth surfacing in an error message. The runtime is chatty on
# startup; these are the parts that explain a failure.
_SIGNALS = ("INTERNAL:", "Failed to", "error:", "ERROR:", "FATAL", "Check failed")


def _drain(read_fd: int, passthrough_fd: int) -> None:
    """Copy the runtime's stderr into the ring buffer and on to the real stderr."""
    buf = b""
    try:
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                return
            try:
                os.write(passthrough_fd, chunk)      # never swallow the runtime's own output
            except OSError:
                pass
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                with _LOCK:
                    _LINES.append(line.decode("utf-8", "replace").rstrip())
    except Exception:  # noqa: BLE001 — a reader thread must never take the process down
        return


def install() -> bool:
    """Tee fd 2 into a ring buffer. Call BEFORE the native library loads. Idempotent.

    Returns True when the capture is in place (or already was), False when this platform cannot
    support it — in which case nothing is changed and callers lose only the extra detail.
    """
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return True
    try:
        saved = os.dup(2)                 # the real stderr, kept for passthrough
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)              # everything written to fd 2 now enters the pipe
        os.close(write_fd)
        thread = threading.Thread(target=_drain, args=(read_fd, saved),
                                  name="nyxara-runtime-log", daemon=True)
        thread.start()
    except Exception:  # noqa: BLE001 — Windows, a closed fd 2, a restricted sandbox
        return False
    with _LOCK:
        _INSTALLED = True
    return True


def recent(limit: int = 40) -> List[str]:
    """The last lines the native runtime wrote to fd 2."""
    with _LOCK:
        return list(_LINES)[-limit:]


def interesting(limit: int = 4) -> List[str]:
    """The most recent lines that look like an explanation, newest last.

    Deduplicated and trimmed: the runtime repeats a source-location trace after every failure, and
    an error message carrying it three times is no more informative than one carrying it once.
    """
    out: List[str] = []
    for line in reversed(recent(120)):
        if any(sig in line for sig in _SIGNALS):
            cleaned = line.strip()
            # drop the "E0000 00:00:1785942277.643173 29363 engine.cc:1619] " preamble
            if "] " in cleaned:
                cleaned = cleaned.split("] ", 1)[1]
            if cleaned and cleaned not in out:
                out.append(cleaned)
            if len(out) >= limit:
                break
    out.reverse()
    return out
