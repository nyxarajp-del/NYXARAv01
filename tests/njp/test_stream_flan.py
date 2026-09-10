"""The reader that goes through 94 GB, and the two silences it used to keep.

Neither of the defects pinned here announced itself. One made the reader stop emitting objects
part-way through a file while believing it was still reading; the other made it report a transfer
that had been cut off as a file that had ended. Both produced numbers that were quietly wrong, and
a number that is quietly wrong is worse than an error, so what these tests check is that the reader
now *raises* where it used to return.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import stream_flan  # noqa: E402
from stream_flan import _scan, objects  # noqa: E402


def a_row(n: int) -> str:
    """A row the reader considers worth parsing — `_WORTH` decides that, and `Q:` is on the list."""
    return json.dumps({"inputs": f"Q: question {n}", "targets": "yes", "_task_name": "t"})


class _Response:
    """A urlopen result that hands back the bytes it was given, in chunks, then stops."""

    def __init__(self, body: bytes, length: int, stop_at: Optional[int] = None) -> None:
        self.stream = io.BytesIO(body)
        self.headers = {"Content-Length": str(length)}
        self.stop_at = stop_at
        self.given = 0

    def read(self, size: int) -> bytes:
        # `None` is unlimited; `0` means this connection delivers nothing at all, which is what a
        # server that keeps dying at the same byte does to every resume after the first.
        if self.stop_at is not None:
            size = min(size, max(0, self.stop_at - self.given))
        if not size:
            return b""
        block = self.stream.read(size)
        self.given += len(block)
        return block

    def __enter__(self): return self
    def __exit__(self, *_): return False


# --------------------------------------------------------------------------------------------- #
#  the chunk boundary
# --------------------------------------------------------------------------------------------- #
def test_a_brace_inside_a_prompt_is_not_read_as_structure():
    """The first silence: `{` in somebody's text counted as an object opening, and nothing closed."""
    row = json.dumps({"inputs": "Q: use a dict like {a: 1} here", "targets": "ok"})
    found, carried, depth, start = _scan(row, 0, -1)
    assert len(found) == 1
    assert found[0]["inputs"] == "Q: use a dict like {a: 1} here"
    assert (depth, start) == (0, -1)


def test_a_string_the_chunk_cannot_close_is_carried_not_scanned():
    half = '{"inputs": "a prompt that stops mid-'
    found, carried, _depth, _start = _scan(half, 0, -1)
    assert not found
    assert carried == half


def test_the_carried_half_completes_on_the_next_chunk():
    whole = a_row(1)
    at = len(whole) // 2
    found, carried, depth, start = _scan(whole[:at], 0, -1)
    assert not found
    found, _carried, _depth, _start = _scan(carried + whole[at:], depth, start)
    assert len(found) == 1 and found[0]["inputs"] == "Q: question 1"


# --------------------------------------------------------------------------------------------- #
#  the end of the file
# --------------------------------------------------------------------------------------------- #
def test_a_whole_file_is_read_and_does_not_raise(monkeypatch):
    body = ("\n".join(a_row(i) for i in range(20))).encode()
    monkeypatch.setattr(stream_flan.urllib.request, "urlopen",
                        lambda *_a, **_k: _Response(body, len(body)))
    assert len(list(objects("http://x/f.json"))) == 20


def test_a_transfer_cut_short_is_resumed_rather_than_believed(monkeypatch):
    """The second silence. An empty block is what a finished file returns *and* what a dead one
    returns, so the reader could not tell them apart and reported the short read as complete."""
    body = ("\n".join(a_row(i) for i in range(40))).encode()
    calls = {"n": 0}

    def _open(request, *_a, **_k):
        calls["n"] += 1
        at = int(str(request.headers.get("Range", "bytes=0-")).split("=")[1].split("-")[0])
        # The first connection dies half way; the resumed one delivers the rest.
        stop = (len(body) - at) // 2 if calls["n"] == 1 else None
        return _Response(body[at:], len(body) - at, stop_at=stop)

    monkeypatch.setattr(stream_flan.urllib.request, "urlopen", _open)
    monkeypatch.setattr(stream_flan.time, "sleep", lambda _s: None)
    assert len(list(objects("http://x/f.json"))) == 40
    assert calls["n"] == 2, "it should have reconnected exactly once"


def test_a_read_that_cannot_be_completed_raises_rather_than_returning_what_it_has(monkeypatch):
    """A server that dies at the same byte every time. Four resumes, then it says so."""
    body = ("\n".join(a_row(i) for i in range(40))).encode()
    dies_at = len(body) // 2

    def _open(request, *_a, **_k):
        at = int(str(request.headers.get("Range", "bytes=0-")).split("=")[1].split("-")[0])
        return _Response(body[at:], len(body) - at, stop_at=max(0, dies_at - at))

    monkeypatch.setattr(stream_flan.urllib.request, "urlopen", _open)
    monkeypatch.setattr(stream_flan.time, "sleep", lambda _s: None)
    with pytest.raises(IOError, match="stopped at"):
        list(objects("http://x/f.json"))


def test_a_byte_limit_is_not_a_short_read(monkeypatch):
    """Asking for a prefix on purpose must not look like a transfer that failed."""
    body = ("\n".join(a_row(i) for i in range(400))).encode()
    monkeypatch.setattr(stream_flan, "CHUNK", 256)
    monkeypatch.setattr(stream_flan.urllib.request, "urlopen",
                        lambda *_a, **_k: _Response(body, len(body)))
    got = list(objects("http://x/f.json", limit_bytes=len(body) // 4))
    assert 0 < len(got) < 400
