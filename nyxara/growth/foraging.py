"""NYXARA · growth/foraging.py — autonomous epistemic foraging (Part O).

A living thing hunts for food; NYXARA hunts for knowledge. When she is idle and compute is cheap, she
forages the web (arXiv / GitHub / tech forums via the existing ``growth/researcher.AutonomousResearcher``),
**verifies** each finding through the Part-M grounded verifier, and **distils** only the verified ones into
her local KnowledgeGraph (Part B) — enriching herself without being asked, while never swallowing a
hallucination as fact.

Honest boundary: strictly **rate-limited** (a per-cycle item cap; egress governed by the environment's
network policy + ``guard/netsec`` budgets), **sandbox-parsed** (fetched content is never executed
in-process — the researcher/knowledge-base ingest path is isolated), and gauntlet/verifier-gated. It is a
clean **no-op** when there is no fetch source (no network) or when the machine is under load — foraging
yields to real work and to the environment's policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

__all__ = ["ForageReport", "EpistemicForager"]

# A fetch source yields (question, answer[, confidence]) candidates for a topic.
FetchFn = Callable[[str], Sequence[Tuple]]


@dataclass
class ForageReport:
    ran: bool
    topic: str = ""
    considered: int = 0
    stored: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"ran": self.ran, "topic": self.topic, "considered": self.considered,
                "stored": self.stored, "reason": self.reason}


class EpistemicForager:
    """Idle-time knowledge hunter: fetch → verify → distil only what checks out (Part O)."""

    def __init__(self, *, fetch: Optional[FetchFn] = None, verifier: Any = None,
                 distiller: Any = None, load_ok: Optional[Callable[[], bool]] = None,
                 max_items_per_cycle: int = 5) -> None:
        self.fetch = fetch
        if verifier is None:
            from nyxara.growth.verified_distill import GroundedVerifier
            verifier = GroundedVerifier()
        self.verifier = verifier
        self.distiller = distiller
        self.load_ok = load_ok                 # callable() -> True when idle+low-load (forage allowed)
        self.max_items = int(max_items_per_cycle)

    @classmethod
    def with_researcher(cls, researcher: Any, **kw) -> "EpistemicForager":
        """Wire the default fetch to an AutonomousResearcher (best-effort extraction of Q/A pairs)."""
        def _fetch(topic: str) -> List[Tuple]:
            try:
                report = researcher.research(topic)
            except Exception:  # noqa: BLE001 — no network / researcher error → forage nothing
                return []
            pairs: List[Tuple] = []
            for attr in ("knowledge_pairs", "pairs"):
                fn = getattr(report, attr, None)
                if callable(fn):
                    try:
                        return list(fn())
                    except Exception:  # noqa: BLE001
                        pass
            for f in (getattr(report, "findings", None) or []):
                q = str(getattr(f, "question", "") or getattr(f, "claim", "") or topic)
                a = str(getattr(f, "answer", "") or getattr(f, "text", "") or f)
                if a:
                    pairs.append((q, a))
            return pairs
        return cls(fetch=_fetch, **kw)

    def should_forage(self) -> bool:
        """Only when idle and lightly loaded — foraging must yield to real work."""
        return self.load_ok is None or bool(self.load_ok())

    def forage(self, topic: str) -> ForageReport:
        if not self.should_forage():
            return ForageReport(ran=False, topic=topic, reason="busy — yielding to real work")
        if self.fetch is None:
            return ForageReport(ran=False, topic=topic, reason="no fetch source (no network)")
        try:
            items = list(self.fetch(topic) or [])
        except Exception:  # noqa: BLE001 — a fetch failure is a no-op, never a crash
            return ForageReport(ran=False, topic=topic, reason="fetch failed")
        considered = stored = 0
        for item in items[:self.max_items]:
            if not item or len(item) < 2:
                continue
            q, a = str(item[0]), str(item[1])
            conf = float(item[2]) if len(item) > 2 else 1.0
            considered += 1
            vr = self.verifier.verify(q, a, prior_confidence=conf)
            if vr.refuted:
                continue                        # never bake a refuted finding
            if self.distiller is not None:
                try:
                    if self.distiller.distill_turn(q, a, source="forage", confidence=conf).stored:
                        stored += 1
                except Exception:  # noqa: BLE001
                    pass
            elif vr.durable:
                stored += 1
        return ForageReport(ran=True, topic=topic, considered=considered, stored=stored)
