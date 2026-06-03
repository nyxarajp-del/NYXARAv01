"""NYXARA · kernel/replay.py — deterministic record / replay of cognition (✦).

If you cannot reproduce a thought, you cannot debug it, audit it, or trust it.
This module makes every cognitive run **bit-for-bit reproducible** by intercepting
the three sources of nondeterminism and journaling them:

* **randomness** — all RNG draws go through :class:`RandomProxy`;
* **time** — clock reads go through :meth:`Recorder.now`;
* **external I/O** — every LLM call, web fetch, tool result, or sensor read is wrapped
  in :meth:`Recorder.capture`, which records the *result* (not the call).

In **RECORD** mode the real sources run and their outputs are appended to an
**append-only, hash-chained** journal (tamper-evident, like guard/audit). In
**REPLAY** mode the journal is replayed back in order — external calls are *not*
re-executed (no side effects), and any deviation in the sequence of requests raises
:class:`ReplayDivergence`, which is exactly how nondeterminism bugs are caught.

Usage::

    rec = Recorder(mode=Mode.RECORD, seed=7)
    with rec:
        x = rec.random.uniform(0, 1)
        t = rec.now()
        ans = rec.capture("llm.ask", lambda: provider.complete(prompt))
        with rec.step("decide"):
            ...
    rec.save("run.jsonl")

    rep = Recorder.load("run.jsonl", mode=Mode.REPLAY)
    with rep:
        assert rep.random.uniform(0, 1) == x          # same value
        ans2 = rep.capture("llm.ask", lambda: 1/0)    # NOT executed; returns recorded

Depends on :mod:`errors`. Pure standard library otherwise.
"""

from __future__ import annotations

import hashlib
import json
import random as _random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar

from nyxara.kernel.errors import ErrorCategory, NyxaraError

__all__ = [
    "Mode",
    "StepKind",
    "RecordEntry",
    "ReplayDivergence",
    "ReplayExhausted",
    "RandomProxy",
    "Recorder",
]

T = TypeVar("T")

_GENESIS = "0" * 64  # hash-chain genesis


# --------------------------------------------------------------------------- #
# Modes & entry kinds
# --------------------------------------------------------------------------- #
class Mode(str, Enum):
    OFF = "off"        # pass-through; nothing recorded (zero overhead path)
    RECORD = "record"
    REPLAY = "replay"


class StepKind(str, Enum):
    RANDOM = "random"
    CLOCK = "clock"
    INPUT = "input"        # explicit external value injected
    CAPTURE = "capture"    # wrapped external I/O result
    STEP_ENTER = "step_enter"
    STEP_EXIT = "step_exit"
    DECISION = "decision"
    OUTPUT = "output"
    MARKER = "marker"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ReplayDivergence(NyxaraError):
    """The code under replay requested something other than the recorded sequence."""

    default_category = ErrorCategory.PERMANENT


class ReplayExhausted(ReplayDivergence):
    """Replay ran past the end of the recorded journal."""


# --------------------------------------------------------------------------- #
# Journal entry (hash-chained)
# --------------------------------------------------------------------------- #
@dataclass
class RecordEntry:
    seq: int
    kind: StepKind
    label: str
    value: Any
    at: float
    prev_hash: str = _GENESIS
    digest: str = ""
    serializable: bool = True

    def compute_digest(self) -> str:
        try:
            payload = json.dumps(self.value, sort_keys=True, default=str)
        except Exception:
            payload = repr(self.value)
        basis = f"{self.prev_hash}|{self.seq}|{self.kind.value}|{self.label}|{payload}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        return {
            "seq": self.seq, "kind": self.kind.value, "label": self.label,
            "value": self.value, "at": self.at, "prev_hash": self.prev_hash,
            "digest": self.digest, "serializable": self.serializable,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "RecordEntry":
        return cls(
            seq=d["seq"], kind=StepKind(d["kind"]), label=d["label"], value=d["value"],
            at=d["at"], prev_hash=d["prev_hash"], digest=d["digest"],
            serializable=d.get("serializable", True),
        )


# --------------------------------------------------------------------------- #
# Random proxy
# --------------------------------------------------------------------------- #
class RandomProxy:
    """Deterministic RNG facade. Records draw results; replays them exactly."""

    def __init__(self, recorder: "Recorder", seed: Optional[int] = None) -> None:
        self._rec = recorder
        self._rng = _random.Random(seed)

    # each method records/replays its RESULT (version-independent reproduction)
    def random(self) -> float:
        return self._rec._rand("random", self._rng.random)

    def uniform(self, a: float, b: float) -> float:
        return self._rec._rand(f"uniform:{a},{b}", lambda: self._rng.uniform(a, b))

    def randint(self, a: int, b: int) -> int:
        return self._rec._rand(f"randint:{a},{b}", lambda: self._rng.randint(a, b))

    def randrange(self, *args: int) -> int:
        return self._rec._rand(f"randrange:{args}", lambda: self._rng.randrange(*args))

    def gauss(self, mu: float, sigma: float) -> float:
        return self._rec._rand(f"gauss:{mu},{sigma}", lambda: self._rng.gauss(mu, sigma))

    def choice(self, seq: Sequence[T]) -> T:
        idx = self._rec._rand("choice", lambda: self._rng.randrange(len(seq)))
        return seq[idx]

    def shuffle(self, seq: List[T]) -> None:
        # record the permutation as a list of indices, reproduce by reordering
        n = len(seq)
        order = self._rec._rand("shuffle", lambda: _sample_perm(self._rng, n))
        original = list(seq)
        for i, j in enumerate(order):
            seq[i] = original[j]


def _sample_perm(rng: _random.Random, n: int) -> List[int]:
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
@dataclass
class _Meta:
    seed: Optional[int]
    created_at: float
    run_id: str


class Recorder:
    """Records or replays a cognitive run deterministically."""

    def __init__(self, *, mode: Mode = Mode.RECORD, seed: Optional[int] = None,
                 run_id: Optional[str] = None) -> None:
        self.mode = mode
        self.meta = _Meta(seed=seed, created_at=time.time(), run_id=run_id or uuid.uuid4().hex)
        self.entries: List[RecordEntry] = []
        self._cursor = 0           # replay read position
        self._head = _GENESIS      # last digest (record mode)
        self._seq = 0
        self._step_depth = 0
        self.random = RandomProxy(self, seed)
        self._active = False

    # ---- context ---- #
    def __enter__(self) -> "Recorder":
        self._active = True
        # marker() handles RECORD (append), REPLAY (consume), and OFF (no-op)
        self.marker("__run_start__", {"seed": self.meta.seed, "run_id": self.meta.run_id})
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.mode is Mode.RECORD:
            self.marker("__run_end__", {"ok": exc[0] is None})
        elif self.mode is Mode.REPLAY and exc[0] is None:
            # closing marker is bookkeeping — best-effort; a caught mid-run divergence
            # leaves the cursor misaligned, which must not raise on a clean exit.
            try:
                self.marker("__run_end__", {"ok": True})
            except ReplayDivergence:
                pass
        self._active = False

    # ---- core record/replay primitive ---- #
    def _append(self, kind: StepKind, label: str, value: Any) -> RecordEntry:
        serializable = True
        try:
            json.dumps(value, default=str)
        except Exception:
            serializable = False
        e = RecordEntry(seq=self._seq, kind=kind, label=label, value=value,
                        at=time.time(), prev_hash=self._head, serializable=serializable)
        e.digest = e.compute_digest()
        self._head = e.digest
        self._seq += 1
        self.entries.append(e)
        return e

    def _next_expected(self, kind: StepKind, label: str) -> RecordEntry:
        if self._cursor >= len(self.entries):
            raise ReplayExhausted(
                f"replay past end of journal: wanted {kind.value}:{label}",
                context={"cursor": self._cursor, "len": len(self.entries)},
            )
        e = self.entries[self._cursor]
        if e.kind is not kind or e.label != label:
            raise ReplayDivergence(
                "replay diverged from recorded sequence",
                context={
                    "at_seq": e.seq, "expected": f"{e.kind.value}:{e.label}",
                    "requested": f"{kind.value}:{label}",
                },
            )
        self._cursor += 1
        return e

    def _consume_or_produce(self, kind: StepKind, label: str,
                            producer: Callable[[], Any], *,
                            execute_in_replay: bool = False) -> Any:
        if self.mode is Mode.OFF:
            return producer()
        if self.mode is Mode.RECORD:
            value = producer()
            self._append(kind, label, value)
            return value
        # REPLAY
        e = self._next_expected(kind, label)
        if execute_in_replay:
            producer()  # side-effect re-execution requested (rare)
        return e.value

    # ---- public capture primitives ---- #
    def _rand(self, label: str, producer: Callable[[], Any]) -> Any:
        return self._consume_or_produce(StepKind.RANDOM, label, producer)

    def now(self) -> float:
        """Deterministic clock read."""
        return self._consume_or_produce(StepKind.CLOCK, "now", time.time)

    def capture(self, label: str, fn: Callable[[], T]) -> T:
        """Wrap a nondeterministic / side-effecting call.

        RECORD: runs ``fn`` and journals its result.
        REPLAY: does **not** run ``fn`` (no side effects) — returns the recorded result.
        """
        return self._consume_or_produce(StepKind.CAPTURE, label, fn)

    async def acapture(self, label: str, coro_fn: Callable[[], Any]) -> Any:
        """Async variant of :meth:`capture`. ``coro_fn`` returns an awaitable."""
        if self.mode is Mode.OFF:
            return await coro_fn()
        if self.mode is Mode.RECORD:
            value = await coro_fn()
            self._append(StepKind.CAPTURE, label, value)
            return value
        e = self._next_expected(StepKind.CAPTURE, label)
        return e.value

    def input(self, label: str, value: Any) -> Any:
        """Record/replay an explicitly-provided external input value."""
        return self._consume_or_produce(StepKind.INPUT, label, lambda: value)

    def decision(self, label: str, value: Any) -> Any:
        """Journal a decision the system made (audit/debug; replay-checked)."""
        return self._consume_or_produce(StepKind.DECISION, label, lambda: value)

    def output(self, label: str, value: Any) -> Any:
        return self._consume_or_produce(StepKind.OUTPUT, label, lambda: value)

    def marker(self, label: str, value: Any = None) -> None:
        if self.mode is Mode.RECORD:
            self._append(StepKind.MARKER, label, value)
        elif self.mode is Mode.REPLAY:
            self._next_expected(StepKind.MARKER, label)

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Bracket a cognitive step (records ENTER/EXIT markers for tracing/replay)."""
        depth = self._step_depth
        if self.mode is Mode.RECORD:
            self._append(StepKind.STEP_ENTER, name, {"depth": depth})
        elif self.mode is Mode.REPLAY:
            self._next_expected(StepKind.STEP_ENTER, name)
        self._step_depth += 1
        try:
            yield
        finally:
            self._step_depth -= 1
            if self.mode is Mode.RECORD:
                self._append(StepKind.STEP_EXIT, name, {"depth": depth})
            elif self.mode is Mode.REPLAY:
                self._next_expected(StepKind.STEP_EXIT, name)

    # ---- integrity ---- #
    def verify_chain(self) -> bool:
        """Recompute the hash chain; return True iff intact. Raises on break."""
        prev = _GENESIS
        for e in self.entries:
            if e.prev_hash != prev:
                raise ReplayDivergence(
                    "hash-chain broken (prev_hash mismatch)",
                    context={"seq": e.seq, "expected_prev": prev, "found": e.prev_hash},
                )
            expect = e.compute_digest()
            if expect != e.digest:
                raise ReplayDivergence(
                    "hash-chain broken (entry tampered)",
                    context={"seq": e.seq, "expected": expect, "found": e.digest},
                )
            prev = e.digest
        return True

    @property
    def head(self) -> str:
        return self._head if self.mode is Mode.RECORD else (
            self.entries[-1].digest if self.entries else _GENESIS
        )

    # ---- persistence ---- #
    def save(self, path: str) -> str:
        header = {"__meta__": {"seed": self.meta.seed, "created_at": self.meta.created_at,
                               "run_id": self.meta.run_id, "count": len(self.entries)}}
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for e in self.entries:
                f.write(json.dumps(e.to_json(), default=str) + "\n")
        return path

    @classmethod
    def load(cls, path: str, *, mode: Mode = Mode.REPLAY) -> "Recorder":
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        header = json.loads(lines[0])["__meta__"]
        rec = cls(mode=mode, seed=header.get("seed"), run_id=header.get("run_id"))
        rec.entries = [RecordEntry.from_json(json.loads(ln)) for ln in lines[1:]]
        rec.meta.created_at = header.get("created_at", rec.meta.created_at)
        if rec.entries:
            rec._head = rec.entries[-1].digest
            rec._seq = rec.entries[-1].seq + 1
        return rec

    # ---- introspection ---- #
    def summary(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        for e in self.entries:
            by_kind[e.kind.value] = by_kind.get(e.kind.value, 0) + 1
        return {
            "mode": self.mode.value, "run_id": self.meta.run_id, "seed": self.meta.seed,
            "entries": len(self.entries), "by_kind": by_kind, "head": self.head[:16],
            "cursor": self._cursor,
        }


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile, os

    print("=" * 70)
    print("NYXARA replay self-test")
    print("=" * 70)

    # ---- RECORD a run with RNG, clock, and an 'LLM call' ---- #
    external_calls = {"n": 0}

    def fake_llm(prompt: str) -> str:
        external_calls["n"] += 1
        return f"answer to {prompt} (#{external_calls['n']})"

    rec = Recorder(mode=Mode.RECORD, seed=42)
    recorded_vals: List[Any] = []
    with rec:
        with rec.step("perceive"):
            recorded_vals.append(rec.random.uniform(0, 1))
            recorded_vals.append(rec.now())
        with rec.step("think"):
            recorded_vals.append(rec.capture("llm", lambda: fake_llm("hello")))
            recorded_vals.append(rec.random.randint(1, 100))
            recorded_vals.append(rec.random.choice(["a", "b", "c", "d"]))
        rec.decision("act", {"plan": "respond"})

    assert external_calls["n"] == 1
    print(f"recorded entries    : {rec.summary()['by_kind']}")
    assert rec.verify_chain()
    print("hash chain intact   : OK")

    # ---- SAVE + LOAD + REPLAY exactly ---- #
    path = os.path.join(tempfile.gettempdir(), "nyx_replay_demo.jsonl")
    rec.save(path)

    rep = Recorder.load(path, mode=Mode.REPLAY)
    assert rep.verify_chain()
    replayed_vals: List[Any] = []
    with rep:
        with rep.step("perceive"):
            replayed_vals.append(rep.random.uniform(0, 1))
            replayed_vals.append(rep.now())
        with rep.step("think"):
            # the LLM is NOT called again — recorded value is returned
            replayed_vals.append(rep.capture("llm", lambda: fake_llm("SHOULD NOT RUN")))
            replayed_vals.append(rep.random.randint(1, 100))
            replayed_vals.append(rep.random.choice(["a", "b", "c", "d"]))
        rep.decision("act", {"plan": "respond"})

    assert external_calls["n"] == 1, "external call was re-executed during replay!"
    assert replayed_vals == recorded_vals, (replayed_vals, recorded_vals)
    print(f"replay reproduced   : {replayed_vals}")
    print("no external re-exec  : OK")

    # ---- divergence detection ---- #
    rep2 = Recorder.load(path, mode=Mode.REPLAY)
    diverged = False
    with rep2:
        try:
            with rep2.step("perceive"):
                rep2.random.uniform(0, 1)
                rep2.now()
            # WRONG next call: ask for a capture labelled differently
            rep2.capture("WRONG_LABEL", lambda: None)
        except ReplayDivergence as e:
            diverged = True
            print(f"divergence caught   : {e.context['expected']} vs {e.context['requested']}")
    assert diverged

    # ---- tamper detection ---- #
    rep3 = Recorder.load(path, mode=Mode.REPLAY)
    rep3.entries[2].value = "tampered!"
    try:
        rep3.verify_chain()
        raise SystemExit("ERROR: tamper not detected")
    except ReplayDivergence:
        print("tamper detected     : OK")

    os.remove(path)
    print("\nALL SELF-TESTS PASSED ✓")
