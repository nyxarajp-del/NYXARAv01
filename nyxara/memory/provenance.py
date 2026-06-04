"""NYXARA · memory/provenance.py — every belief carries its origin (✦).

Nothing NYXARA "knows" is unattributed. Each belief is tagged with **where it came
from**, **how confident** the claim is, **when** it was acquired, and **what it was
derived from**. This is the substrate for Rule 6 (Absolute Transparency — *disclose
the source chain and completeness percentage*) and for honest, calibrated reporting.

Trust model
-----------
Each :class:`SourceType` has a base trustworthiness — the Owner is ground truth;
the open web is untrusted-by-default (Rule 3). A belief's *live* trust is its base
trust, then:

* **decayed** by staleness (old facts are less reliable; some sources never decay),
* **boosted** by independent corroboration (noisy-OR),
* **penalised** by contradiction.

Derived beliefs (inference) combine their parents by the *weakest-link* principle:
a conclusion is no stronger than the flimsiest premise, times the method's
reliability. The full lineage is walkable as a **source chain** for citation.

Pure standard library; depends only on :mod:`nyxara.kernel.errors`.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from nyxara.kernel.errors import MemoryError_

__all__ = [
    "SourceType",
    "Provenance",
    "Belief",
    "ProvenanceStore",
    "combine_confidence_or",
]

T = TypeVar("T")

_DAY = 86400.0


# --------------------------------------------------------------------------- #
# Source taxonomy & base trust
# --------------------------------------------------------------------------- #
class SourceType(str, Enum):
    OWNER = "owner"                      # the Master — ground truth (Rule 1/7)
    DIRECT_OBSERVATION = "observation"   # NYXARA saw it happen
    SENSOR = "sensor"                    # instrument reading
    TOOL = "tool"                        # a registered tool returned it
    MEMORY_DERIVED = "memory_derived"    # recalled / consolidated from memory
    SELF_REFLECTION = "self_reflection"  # introspective conclusion
    LLM_INFERENCE = "llm_inference"      # probabilistic generation (may hallucinate)
    EXTERNAL_AGENT = "external_agent"    # another agent (Level-Omega by default, Rule 3)
    USER_OTHER = "user_other"            # a non-owner human
    WEB = "web"                          # internet content — UNTRUSTED (via shield)
    ASSUMPTION = "assumption"            # a working hypothesis with no source

    @property
    def base_trust(self) -> float:
        return {
            SourceType.OWNER: 1.0,
            SourceType.DIRECT_OBSERVATION: 0.9,
            SourceType.SENSOR: 0.85,
            SourceType.TOOL: 0.8,
            SourceType.MEMORY_DERIVED: 0.7,
            SourceType.SELF_REFLECTION: 0.6,
            SourceType.LLM_INFERENCE: 0.55,
            SourceType.EXTERNAL_AGENT: 0.4,
            SourceType.USER_OTHER: 0.35,
            SourceType.WEB: 0.3,
            SourceType.ASSUMPTION: 0.2,
        }[self]

    @property
    def decays(self) -> bool:
        """Owner statements and direct assumptions are treated as non-decaying."""
        return self not in (SourceType.OWNER, SourceType.ASSUMPTION)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def combine_confidence_or(values: Sequence[float]) -> float:
    """Noisy-OR: independent corroborating evidence raises confidence."""
    prod = 1.0
    for v in values:
        prod *= (1.0 - _clamp(v))
    return _clamp(1.0 - prod)


# --------------------------------------------------------------------------- #
# Provenance record
# --------------------------------------------------------------------------- #
@dataclass
class Provenance:
    """The origin metadata for a single belief."""

    source: SourceType
    confidence: float = 0.8                 # asserted confidence of the claim [0,1]
    detail: str = ""                        # URL / tool name / person id / note
    method: str = "asserted"                # observed | inferred | told | retrieved | assumed
    acquired_at: float = field(default_factory=time.time)
    half_life_days: float = 30.0            # staleness decay constant
    parents: Tuple[str, ...] = ()           # prov_ids this belief was derived from
    tags: frozenset = field(default_factory=frozenset)
    prov_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    corroborations: List[Tuple[str, float, float]] = field(default_factory=list)  # (src, conf, at)
    contradictions: List[Tuple[str, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)
        if not (0.0 < self.half_life_days):
            raise MemoryError_("half_life_days must be positive", context={"value": self.half_life_days})

    # ---- dynamics ---- #
    def age_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, (now or time.time()) - self.acquired_at)

    def staleness_factor(self, now: Optional[float] = None) -> float:
        if not self.source.decays:
            return 1.0
        age_days = self.age_seconds(now) / _DAY
        return math.exp(-age_days / self.half_life_days)

    def corroboration_boost(self) -> float:
        if not self.corroborations:
            return 1.0
        agg = combine_confidence_or([c for _, c, _ in self.corroborations])
        return 1.0 + 0.5 * agg  # up to +50%

    def contradiction_penalty(self) -> float:
        if not self.contradictions:
            return 1.0
        agg = combine_confidence_or([c for _, c, _ in self.contradictions])
        return _clamp(1.0 - 0.7 * agg, 0.05, 1.0)

    def trust(self, now: Optional[float] = None) -> float:
        """Live trustworthiness in [0,1]."""
        t = (self.source.base_trust
             * self.staleness_factor(now)
             * self.corroboration_boost()
             * self.contradiction_penalty())
        return _clamp(t)

    def effective_confidence(self, now: Optional[float] = None) -> float:
        """Calibrated belief strength = asserted confidence × live trust."""
        return _clamp(self.confidence * self.trust(now))

    def is_stale(self, threshold: float = 0.3, now: Optional[float] = None) -> bool:
        return self.trust(now) < threshold

    # ---- evidence ---- #
    def corroborate(self, source: SourceType, confidence: float = 0.7,
                    at: Optional[float] = None) -> None:
        self.corroborations.append((source.value, _clamp(confidence), at or time.time()))

    def contradict(self, source: SourceType, confidence: float = 0.7,
                   at: Optional[float] = None) -> None:
        self.contradictions.append((source.value, _clamp(confidence), at or time.time()))

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {
            "prov_id": self.prov_id,
            "source": self.source.value,
            "detail": self.detail,
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "trust": round(self.trust(now), 4),
            "effective_confidence": round(self.effective_confidence(now), 4),
            "acquired_at": self.acquired_at,
            "age_days": round(self.age_seconds(now) / _DAY, 3),
            "parents": list(self.parents),
            "corroborations": len(self.corroborations),
            "contradictions": len(self.contradictions),
            "tags": sorted(self.tags),
        }

    # ---- derivation ---- #
    @classmethod
    def derive(
        cls,
        parents: Sequence["Provenance"],
        *,
        method: str = "inferred",
        source: SourceType = SourceType.MEMORY_DERIVED,
        reliability: float = 0.9,
        detail: str = "",
        tags: Sequence[str] = (),
        now: Optional[float] = None,
    ) -> "Provenance":
        """Build provenance for a belief inferred from ``parents`` (weakest-link)."""
        if not parents:
            raise MemoryError_("cannot derive a belief from zero parents")
        weakest = min(p.effective_confidence(now) for p in parents)
        conf = _clamp(reliability * weakest)
        return cls(
            source=source,
            confidence=conf,
            detail=detail or f"derived via {method} from {len(parents)} source(s)",
            method=method,
            parents=tuple(p.prov_id for p in parents),
            tags=frozenset(tags),
        )


# --------------------------------------------------------------------------- #
# Belief — a value with provenance
# --------------------------------------------------------------------------- #
@dataclass
class Belief(Generic[T]):
    """A value tagged with where it came from."""

    value: T
    provenance: Provenance

    def trust(self, now: Optional[float] = None) -> float:
        return self.provenance.trust(now)

    def effective_confidence(self, now: Optional[float] = None) -> float:
        return self.provenance.effective_confidence(now)

    def is_stale(self, threshold: float = 0.3, now: Optional[float] = None) -> bool:
        return self.provenance.is_stale(threshold, now)

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {"value": self.value, "provenance": self.provenance.to_dict(now)}


# --------------------------------------------------------------------------- #
# Store — registry + source-chain + completeness (Rule 6)
# --------------------------------------------------------------------------- #
class ProvenanceStore:
    """Registry of provenance records supporting lineage and transparency reports."""

    def __init__(self) -> None:
        self._records: Dict[str, Provenance] = {}

    def register(self, prov: Provenance) -> str:
        self._records[prov.prov_id] = prov
        return prov.prov_id

    def get(self, prov_id: str) -> Provenance:
        try:
            return self._records[prov_id]
        except KeyError:
            raise MemoryError_(f"no provenance with id {prov_id!r}") from None

    def __len__(self) -> int:
        return len(self._records)

    def corroborate(self, prov_id: str, source: SourceType, confidence: float = 0.7) -> None:
        self.get(prov_id).corroborate(source, confidence)

    def contradict(self, prov_id: str, source: SourceType, confidence: float = 0.7) -> None:
        self.get(prov_id).contradict(source, confidence)

    # ---- lineage ---- #
    def source_chain(self, prov_id: str, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Breadth-first walk of the full derivation lineage (citation chain)."""
        chain: List[Dict[str, Any]] = []
        seen = set()
        frontier = [prov_id]
        while frontier:
            pid = frontier.pop(0)
            if pid in seen or pid not in self._records:
                continue
            seen.add(pid)
            p = self._records[pid]
            chain.append(p.to_dict(now))
            frontier.extend(p.parents)
        return chain

    def root_sources(self, prov_id: str) -> List[SourceType]:
        """The original (parentless) sources a belief ultimately rests on."""
        roots: List[SourceType] = []
        seen = set()
        frontier = [prov_id]
        while frontier:
            pid = frontier.pop(0)
            if pid in seen or pid not in self._records:
                continue
            seen.add(pid)
            p = self._records[pid]
            if not p.parents:
                roots.append(p.source)
            else:
                frontier.extend(p.parents)
        return roots

    def completeness(self, prov_id: str, required_sources: Sequence[SourceType]) -> float:
        """Fraction of ``required_sources`` actually present in the lineage (Rule 6)."""
        if not required_sources:
            return 1.0
        present = set(self.root_sources(prov_id)) | {
            self._records[pid].source for pid in self._collect_ids(prov_id)
        }
        hit = sum(1 for r in required_sources if r in present)
        return hit / len(required_sources)

    def _collect_ids(self, prov_id: str) -> List[str]:
        ids, seen, frontier = [], set(), [prov_id]
        while frontier:
            pid = frontier.pop(0)
            if pid in seen or pid not in self._records:
                continue
            seen.add(pid)
            ids.append(pid)
            frontier.extend(self._records[pid].parents)
        return ids

    # ---- queries ---- #
    def by_source(self, source: SourceType) -> List[Provenance]:
        return [p for p in self._records.values() if p.source is source]

    def stale(self, threshold: float = 0.3, now: Optional[float] = None) -> List[Provenance]:
        return [p for p in self._records.values() if p.is_stale(threshold, now)]

    def untrusted(self, now: Optional[float] = None) -> List[Provenance]:
        """Beliefs whose root rests on an untrusted source (web/external/assumption)."""
        risky = {SourceType.WEB, SourceType.EXTERNAL_AGENT, SourceType.ASSUMPTION,
                 SourceType.USER_OTHER}
        return [p for p in self._records.values()
                if set(self.root_sources(p.prov_id)) & risky]

    # ---- Rule 6 transparency report ---- #
    def report(self, prov_id: str, required_sources: Sequence[SourceType] = (),
               now: Optional[float] = None) -> Dict[str, Any]:
        p = self.get(prov_id)
        return {
            "prov_id": prov_id,
            "trust": round(p.trust(now), 4),
            "confidence": round(p.confidence, 4),
            "effective_confidence": round(p.effective_confidence(now), 4),
            "completeness": round(self.completeness(prov_id, required_sources), 4),
            "root_sources": [s.value for s in self.root_sources(prov_id)],
            "contradicted": len(p.contradictions) > 0,
            "source_chain": self.source_chain(prov_id, now),
        }


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA provenance self-test")
    print("=" * 70)

    store = ProvenanceStore()
    now = time.time()

    # Owner statement — ground truth, never decays
    owner = Provenance(SourceType.OWNER, confidence=1.0, detail="JP said so", method="told")
    store.register(owner)
    assert owner.trust(now) == 1.0
    # even far in the future, owner trust holds
    assert owner.trust(now + 365 * _DAY) == 1.0
    print(f"owner trust (now / +1yr): {owner.trust(now):.2f} / {owner.trust(now + 365*_DAY):.2f}")

    # Web claim — untrusted, decays
    web = Provenance(SourceType.WEB, confidence=0.9, detail="https://example.com",
                     method="retrieved", half_life_days=7)
    store.register(web)
    print(f"web trust now / +30d   : {web.trust(now):.3f} / {web.trust(now + 30*_DAY):.3f}")
    assert web.trust(now + 30 * _DAY) < web.trust(now)

    # corroboration raises trust; contradiction lowers it
    before = web.trust(now)
    web.corroborate(SourceType.SENSOR, 0.8)
    assert web.trust(now) > before
    web.contradict(SourceType.OWNER, 0.95)
    print(f"web after +corrob -contra: {web.trust(now):.3f}")

    # derived belief = weakest link
    sensor = Provenance(SourceType.SENSOR, confidence=0.95, detail="thermo#3")
    store.register(sensor)
    derived = Provenance.derive([owner, sensor], method="inferred", reliability=0.9)
    store.register(derived)
    print(f"derived conf           : {derived.confidence:.3f} "
          f"(weakest parent eff-conf × 0.9)")
    assert derived.parents == (owner.prov_id, sensor.prov_id)

    # Rule 6 report
    rep = store.report(derived.prov_id,
                       required_sources=[SourceType.OWNER, SourceType.SENSOR, SourceType.WEB])
    print(f"\nRule-6 report:")
    print(f"  trust={rep['trust']}  eff_conf={rep['effective_confidence']}")
    print(f"  completeness={rep['completeness']} (2 of 3 required sources present)")
    print(f"  root_sources={rep['root_sources']}")
    print(f"  source_chain length={len(rep['source_chain'])}")
    assert rep["completeness"] == round(2 / 3, 4)
    assert set(rep["root_sources"]) == {"owner", "sensor"}

    # untrusted scan
    assert web in store.untrusted()
    print(f"\nuntrusted beliefs      : {len(store.untrusted())}")
    print(f"stale beliefs (+60d)   : {len(store.stale(now=now + 60*_DAY))}")

    print("\nALL SELF-TESTS PASSED ✓")
