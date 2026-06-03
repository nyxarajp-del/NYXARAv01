"""Provenance — every belief tagged with source + confidence + acquisition-time.

Knowing *that* something is true is not enough; Nyxara must know *why* it believes it and
*how much* to trust it. This module is the epistemic ledger: it records where each belief
came from, builds derivation chains when beliefs are inferred from others, and computes a
trust-weighted confidence that propagates along the chain (the truth is only as strong as
its weakest cited source).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from nyxara.contracts import Belief, Provenance, new_id

# Default trust priors per source class. Tunable; value_learning may adjust over time.
SOURCE_TRUST: dict[str, float] = {
    "owner": 0.99,
    "perceived": 0.85,
    "tool": 0.8,
    "inferred": 0.7,
    "web": 0.55,
    "rumor": 0.3,
    "unknown": 0.4,
}


@dataclass
class ProvenanceRecord:
    id: str
    source: str
    method: str
    confidence: float
    acquired_at: float
    parents: tuple[str, ...] = ()       # ids of provenance records this derives from


class ProvenanceLedger:
    """Append-only store of provenance records + trust propagation."""

    def __init__(self) -> None:
        self._records: dict[str, ProvenanceRecord] = {}

    def trust_of(self, source: str) -> float:
        key = source.split(":", 1)[0]
        return SOURCE_TRUST.get(key, SOURCE_TRUST["unknown"])

    def tag(self, source: str, method: str = "stated",
            confidence: float | None = None) -> ProvenanceRecord:
        conf = confidence if confidence is not None else self.trust_of(source)
        rec = ProvenanceRecord(id=new_id("prv"), source=source, method=method,
                               confidence=conf, acquired_at=time.time())
        self._records[rec.id] = rec
        return rec

    def derive(self, method: str, by: str, parents: list[ProvenanceRecord],
               combine: str = "min") -> ProvenanceRecord:
        """Create a derived record. Confidence propagates from parents.

        ``min``  — as weak as the weakest cited source (conservative, default).
        ``mult`` — independent-evidence multiplication (stricter still).
        ``mean`` — averaged (optimistic).
        """
        if not parents:
            return self.tag(by, method)
        confs = [p.confidence for p in parents]
        if combine == "mult":
            c = 1.0
            for x in confs:
                c *= x
        elif combine == "mean":
            c = sum(confs) / len(confs)
        else:
            c = min(confs)
        c *= self.trust_of(by)  # the inferring agent's own reliability discounts it
        rec = ProvenanceRecord(id=new_id("prv"), source=by, method=method,
                               confidence=c, acquired_at=time.time(),
                               parents=tuple(p.id for p in parents))
        self._records[rec.id] = rec
        return rec

    def chain(self, record_id: str) -> list[ProvenanceRecord]:
        """Full ancestry of a record, breadth-first."""
        out: list[ProvenanceRecord] = []
        frontier = [record_id]
        seen: set[str] = set()
        while frontier:
            rid = frontier.pop(0)
            rec = self._records.get(rid)
            if rec is None or rid in seen:
                continue
            seen.add(rid)
            out.append(rec)
            frontier.extend(rec.parents)
        return out

    def to_contract(self, record: ProvenanceRecord) -> Provenance:
        """Bridge to the cross-cutting :class:`Provenance` value object."""
        return Provenance(source=record.source, method=record.method,
                          acquired_at=record.acquired_at, confidence=record.confidence,
                          chain=record.parents)

    def attach(self, claim: str, source: str, method: str = "stated") -> Belief:
        """Convenience: mint a provenance-tagged :class:`Belief`."""
        rec = self.tag(source, method)
        return Belief(claim=claim, confidence=rec.confidence,
                      provenance=self.to_contract(rec))

    @property
    def stats(self) -> dict[str, int]:
        return {"records": len(self._records)}


__all__ = ["ProvenanceLedger", "ProvenanceRecord", "SOURCE_TRUST"]
