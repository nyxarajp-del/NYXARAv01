"""NYXARA · njp/ledger.py — the growth record (📒, NJP V.01).

Every claim NJP makes about itself has to be settled here, in numbers, or it is not made. The
ledger is the difference between *"she is getting smarter"* as a slogan and as a reading: it holds
one row per **generation** — how large the fabric was, how fast it answered, how often it was
right, how many self-edits it kept and how many it rolled back — and it survives restarts, so
today's row can actually be compared with yesterday's.

Two things read it back, which is why it is a faculty and not logging:

* :meth:`ErrorMemory.similar` — the **Truth-Seeking Gauntlet** (:mod:`nyxara.njp.truth`) asks the
  ledger whether a claim of this shape has been wrong before, and a claim that resembles a past
  error carries that against it. Learning from your own mistakes only means something if the
  mistakes are still on record when the next one is being considered.
* :meth:`Ledger.regressed` — the **self-editor** (:mod:`nyxara.njp.evolve`) asks whether the last
  edit actually helped. An edit that did not is rolled back, so the ledger is what closes the
  loop between rewriting yourself and being right to have done so.

Honest, as everywhere here: the ledger reports what happened. If growth is flat, it shows flat; if
accuracy fell, it shows the fall and :meth:`regressed` returns True. A record that could only ever
report improvement would be worthless for the one job it has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Generation", "ErrorRecord", "ErrorMemory", "Ledger"]


@dataclass
class Generation:
    """One row: the whole state of NJP's growth at a moment, as numbers."""

    n: int = 0
    t: float = field(default_factory=time.time)
    cells: int = 0
    synapses: int = 0
    mean_degree: float = 0.0
    expansions: int = 0
    grown_total: int = 0
    latency_p50_ms: Optional[float] = None
    accuracy: Optional[float] = None          # held-out replay / precognition accuracy
    error_rate: Optional[float] = None        # 1 - accuracy, kept explicit for readability
    edits_kept: int = 0
    edits_rolled_back: int = 0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "t": round(self.t, 3), "cells": self.cells,
                "synapses": self.synapses, "mean_degree": round(self.mean_degree, 4),
                "expansions": self.expansions, "grown_total": self.grown_total,
                "latency_p50_ms": (round(self.latency_p50_ms, 3)
                                   if self.latency_p50_ms is not None else None),
                "accuracy": round(self.accuracy, 6) if self.accuracy is not None else None,
                "error_rate": (round(self.error_rate, 6)
                               if self.error_rate is not None else None),
                "edits_kept": self.edits_kept, "edits_rolled_back": self.edits_rolled_back,
                "note": self.note}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Generation":
        g = cls()
        for k in ("n", "cells", "synapses", "expansions", "grown_total",
                  "edits_kept", "edits_rolled_back"):
            setattr(g, k, int(d.get(k, 0) or 0))
        g.t = float(d.get("t", time.time()) or time.time())
        g.mean_degree = float(d.get("mean_degree", 0.0) or 0.0)
        for k in ("latency_p50_ms", "accuracy", "error_rate"):
            v = d.get(k)
            setattr(g, k, float(v) if v is not None else None)
        g.note = str(d.get("note", "") or "")
        return g


@dataclass
class ErrorRecord:
    """One thing NJP got wrong, kept so the next similar claim has to answer for it."""

    claim: str = ""
    tokens: List[str] = field(default_factory=list)
    verdict: str = ""              # what it was asserted as
    truth: str = ""                # what it turned out to be
    t: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:400], "tokens": self.tokens[:32],
                "verdict": self.verdict, "truth": self.truth[:400], "t": round(self.t, 3)}


def _tokens(text: str) -> List[str]:
    """Cheap, deterministic shape of a claim. Deliberately lexical — this is a *recall* cue for
    'have I been wrong about something like this', not a semantic judgement."""
    try:
        out, seen = [], set()
        for raw in str(text or "").lower().replace("/", " ").split():
            tok = "".join(ch for ch in raw if ch.isalnum())
            if len(tok) < 3 or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= 32:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


class ErrorMemory:
    """What NJP has been wrong about. Bounded, and consulted before she commits again."""

    def __init__(self, *, capacity: int = 4096) -> None:
        self.capacity = max(16, int(capacity))
        self.records: List[ErrorRecord] = []

    def record(self, claim: str, *, verdict: str = "", truth: str = "") -> ErrorRecord:
        rec = ErrorRecord(claim=str(claim or ""), tokens=_tokens(claim),
                          verdict=str(verdict or ""), truth=str(truth or ""))
        self.records.append(rec)
        if len(self.records) > self.capacity:
            del self.records[: len(self.records) - self.capacity]
        return rec

    def similar(self, claim: str, *, k: int = 5, floor: float = 0.34) -> List[ErrorRecord]:
        """Past errors that look like this claim, strongest first.

        Jaccard over token shape. It will miss a paraphrase — stated rather than papered over —
        which is why the gauntlet treats a hit as *evidence against*, never as the whole verdict.
        """
        try:
            want = set(_tokens(claim))
            if not want:
                return []
            scored = []
            for rec in self.records:
                have = set(rec.tokens)
                if not have:
                    continue
                score = len(want & have) / float(len(want | have))
                if score >= floor:
                    scored.append((score, rec))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [rec for _s, rec in scored[: max(1, int(k))]]
        except Exception:  # noqa: BLE001
            return []

    def to_dict(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "records": [r.to_dict() for r in self.records[-1024:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.capacity = int(d.get("capacity", self.capacity))
            self.records = []
            for row in (d.get("records") or []):
                rec = ErrorRecord(claim=str(row.get("claim", "")),
                                  tokens=list(row.get("tokens") or []),
                                  verdict=str(row.get("verdict", "")),
                                  truth=str(row.get("truth", "")),
                                  t=float(row.get("t", time.time())))
                self.records.append(rec)
        except Exception:  # noqa: BLE001
            pass


class Ledger:
    """Generations of NJP, and the errors she is expected to remember."""

    def __init__(self, *, capacity: int = 8192) -> None:
        self.capacity = max(8, int(capacity))
        self.generations: List[Generation] = []
        self.errors = ErrorMemory()
        self._latencies: List[float] = []

    # ---- writing --------------------------------------------------------- #
    def observe_latency(self, ms: float) -> None:
        """Every answer's cost. The p50 of these becomes the generation's latency column."""
        try:
            self._latencies.append(float(ms))
            if len(self._latencies) > 4096:
                del self._latencies[:-4096]
        except Exception:  # noqa: BLE001
            pass

    def record(self, *, fabric_stats: Optional[Dict[str, Any]] = None,
               accuracy: Optional[float] = None, edits_kept: int = 0,
               edits_rolled_back: int = 0, note: str = "") -> Generation:
        """Close one generation. This is the row tomorrow will be compared against."""
        g = Generation(n=len(self.generations) + 1, note=str(note or ""))
        try:
            fs = fabric_stats or {}
            g.cells = int(fs.get("cells", 0) or 0)
            g.synapses = int(fs.get("synapses", 0) or 0)
            g.mean_degree = float(fs.get("mean_degree", 0.0) or 0.0)
            g.expansions = int(fs.get("expansions", 0) or 0)
            g.grown_total = int(fs.get("grown_total", 0) or 0)
            if accuracy is None:
                accuracy = ((fs.get("manifold") or {}).get("accuracy"))
            if accuracy is not None:
                g.accuracy = float(accuracy)
                g.error_rate = 1.0 - float(accuracy)
            g.latency_p50_ms = self._p50()
            g.edits_kept = int(edits_kept)
            g.edits_rolled_back = int(edits_rolled_back)
        except Exception:  # noqa: BLE001 — a partial row is better than no row
            pass
        self.generations.append(g)
        if len(self.generations) > self.capacity:
            del self.generations[: len(self.generations) - self.capacity]
        self._latencies = []
        return g

    def _p50(self) -> Optional[float]:
        if not self._latencies:
            return None
        s = sorted(self._latencies)
        return s[len(s) // 2]

    # ---- reading back ---------------------------------------------------- #
    @property
    def latest(self) -> Optional[Generation]:
        return self.generations[-1] if self.generations else None

    def regressed(self, *, tolerance: float = 0.01) -> Optional[bool]:
        """Did the most recent generation get **worse**? ``None`` when there is nothing to compare.

        ``None`` rather than ``False`` on a single generation, deliberately: "no evidence yet" and
        "no regression" are different facts, and the self-editor must not read the first as the
        second and keep an edit it never actually checked.
        """
        try:
            if len(self.generations) < 2:
                return None
            now, before = self.generations[-1], self.generations[-2]
            if now.accuracy is None or before.accuracy is None:
                return None
            return bool(now.accuracy < before.accuracy - float(tolerance))
        except Exception:  # noqa: BLE001
            return None

    def report(self, *, k: int = 2) -> Dict[str, Any]:
        """Then vs now — the answer to "is she smarter than she was", in numbers."""
        try:
            if not self.generations:
                return {"generations": 0, "verdict": "no generations recorded yet"}
            now = self.generations[-1]
            if len(self.generations) < 2:
                return {"generations": 1, "now": now.to_dict(),
                        "verdict": "one generation — nothing to compare against yet"}
            before = self.generations[-min(len(self.generations), max(2, int(k)))]
            d_syn = now.synapses - before.synapses
            d_cell = now.cells - before.cells
            d_acc = ((now.accuracy - before.accuracy)
                     if (now.accuracy is not None and before.accuracy is not None) else None)
            d_lat = ((now.latency_p50_ms - before.latency_p50_ms)
                     if (now.latency_p50_ms is not None
                         and before.latency_p50_ms is not None) else None)
            return {
                "generations": len(self.generations),
                "span_s": round(now.t - before.t, 3),
                "before": before.to_dict(), "now": now.to_dict(),
                "delta": {"cells": d_cell, "synapses": d_syn,
                          "accuracy": round(d_acc, 6) if d_acc is not None else None,
                          "latency_p50_ms": round(d_lat, 3) if d_lat is not None else None},
                "grew": bool(d_syn > 0 or d_cell > 0),
                "faster": (bool(d_lat < 0) if d_lat is not None else None),
                "more_accurate": (bool(d_acc > 0) if d_acc is not None else None),
            }
        except Exception:  # noqa: BLE001
            return {"generations": len(self.generations), "verdict": "report failed"}

    def stats(self) -> Dict[str, Any]:
        latest = self.latest
        return {"generations": len(self.generations),
                "errors_recorded": len(self.errors.records),
                "regressed": self.regressed(),
                "latest": latest.to_dict() if latest is not None else None}

    # ---- persistence ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"capacity": self.capacity,
                "generations": [g.to_dict() for g in self.generations[-2048:]],
                "errors": self.errors.to_dict()}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.capacity = int(d.get("capacity", self.capacity))
            self.generations = [Generation.from_dict(row)
                                for row in (d.get("generations") or [])]
            if d.get("errors"):
                self.errors.load_dict(d["errors"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves an empty ledger
            pass
