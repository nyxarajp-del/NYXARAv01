"""Deterministic record / replay of every cognitive step (debug + audit).

Every meaningful step — an event consumed, a proposal made, a verdict rendered, an action
taken — is appended to a journal as a self-describing record. Because each record also
captures the random seed and inputs used, a recorded session can be *replayed*
deterministically: feed the same inputs back through the same handlers and you reproduce
the exact cognitive trajectory. Invaluable for debugging "why did it do that?" and for
audits that need to reconstruct a decision.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class StepRecord:
    """One recorded cognitive step."""

    seq: int
    kind: str                       # "event" | "proposal" | "verdict" | "action" | ...
    trace_id: str
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)
    seed: int | None = None         # rng seed in effect, for determinism

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_safe, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "StepRecord":
        return cls(**json.loads(line))


def _safe(obj: Any) -> Any:
    """Best-effort JSON coercion for arbitrary cognitive payloads."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


class Recorder:
    """Append-only recorder. Writes JSONL if a path is given; always keeps memory tail."""

    def __init__(self, path: Path | None = None, *, keep_in_memory: int = 1000) -> None:
        self._path = path
        self._seq = 0
        self._buffer: list[StepRecord] = []
        self._keep = keep_in_memory
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, trace_id: str, *, seed: int | None = None,
               **payload: Any) -> StepRecord:
        rec = StepRecord(seq=self._seq, kind=kind, trace_id=trace_id,
                         payload=payload, seed=seed)
        self._seq += 1
        self._buffer.append(rec)
        if len(self._buffer) > self._keep:
            self._buffer.pop(0)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(rec.to_json() + "\n")
        return rec

    @property
    def tail(self) -> list[StepRecord]:
        return list(self._buffer)

    def by_trace(self, trace_id: str) -> list[StepRecord]:
        return [r for r in self._buffer if r.trace_id == trace_id]


class Player:
    """Reads a recorded JSONL journal back as an ordered stream of steps."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def steps(self) -> Iterator[StepRecord]:
        if not self._path.exists():
            return iter(())
        def _gen() -> Iterator[StepRecord]:
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield StepRecord.from_json(line)
        return _gen()

    def replay(self, handler) -> int:
        """Feed each recorded step to ``handler(StepRecord)``. Returns count replayed."""
        n = 0
        for step in self.steps():
            handler(step)
            n += 1
        return n


__all__ = ["StepRecord", "Recorder", "Player"]
