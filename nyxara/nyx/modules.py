"""NYXARA · nyx/modules.py — the specialist modules that compete for consciousness (👥).

Global Workspace Theory says the mind is not one process but *many* specialists, each shouting
its own answer, with only the strongest coalition reaching the narrow bottleneck of awareness.
This module is that cast of specialists.

Crucially, **none of these is a new reasoning engine**. Every one is a thin adapter over a
faculty NYXARA already has — memory recall, her concept graph, the first-principles derivation
engine, the creative engine, the three ethical frameworks. What is new is that they now
*compete on the same stage* and are *scored by their measured track record*
(:mod:`nyxara.nyx.metacog`) rather than by a hand-tuned priority.

The one law encoded here is the repo's own: **verifiable beats probabilistic**. A specialist
that can show a derivation reports ``verifiable=True``, and downstream that outranks any
confident guess (see :mod:`nyxara.nyx.hybrid`). A specialist that cannot answer returns
``None`` — silence is a legitimate and common result, and much better than a fabricated one.

Every specialist is fail-soft and reports :meth:`available` honestly when an optional
dependency is missing (e.g. sympy for algebraic proof), rather than faking competence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

__all__ = [
    "Proposal",
    "Situation",
    "SpecialistModule",
    "MemorySpecialist",
    "GraphSpecialist",
    "DerivationSpecialist",
    "CreativeSpecialist",
    "EthicsSpecialist",
    "default_specialists",
]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass
class Proposal:
    """One specialist's bid for consciousness: what it would say, and how much it stands behind it."""

    source: str
    content: str
    confidence: float = 0.5
    verifiable: bool = False         # derived/checked, not guessed — this is the decisive flag
    novelty: float = 0.5
    urgency: float = 0.0
    tags: FrozenSet[str] = field(default_factory=frozenset)
    evidence: List[str] = field(default_factory=list)   # provenance: where this came from
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "content": self.content,
                "confidence": round(self.confidence, 4), "verifiable": self.verifiable,
                "novelty": round(self.novelty, 4), "urgency": round(self.urgency, 4),
                "tags": sorted(self.tags), "evidence": self.evidence,
                "rationale": self.rationale}


@dataclass
class Situation:
    """What every specialist gets to look at: the turn, and what NYX already recalled about it."""

    stimulus: str
    concepts: List[str] = field(default_factory=list)
    context: List[Any] = field(default_factory=list)     # nyx.holomem.Trace
    recall: Any = None                                   # nyx.holomem.Recall
    novelty: float = 0.5
    brain: Any = None                                    # the NyxBrain, for graph/memory access


class SpecialistModule:
    """Base class: a named faculty that may or may not have something to say about a turn."""

    name = "specialist"
    tags: FrozenSet[str] = frozenset()
    priority = 0.5                    # a *prior*; metacog's measured reliability overrides it

    def available(self) -> bool:
        """Honest capability report — False when an optional dependency is missing."""
        return True

    def consider(self, situation: Situation) -> Optional[Proposal]:  # pragma: no cover - base
        raise NotImplementedError

    def safe_consider(self, situation: Situation) -> Optional[Proposal]:
        """Run :meth:`consider` fail-soft. A specialist that raises simply does not bid."""
        try:
            if not self.available():
                return None
            return self.consider(situation)
        except Exception:  # noqa: BLE001 — one broken specialist must not break the workspace
            return None


# --------------------------------------------------------------------------- #
# Memory — what she actually has on record
# --------------------------------------------------------------------------- #
class MemorySpecialist(SpecialistModule):
    """Bids with what content-addressed recall brought back. Evidence, not invention."""

    name = "memory"
    tags = frozenset({"recall", "evidence"})
    priority = 0.7
    floor = 0.05                     # below this the "recall" is crosstalk, not a memory
    # What she *knows* should outrank what merely *happened*: a fact or a settled conclusion is
    # better evidence than an episode, which is only a record of a past exchange.
    kind_weight = {"fact": 1.0, "conclusion": 1.0, "percept": 0.9,
                   "episode": 0.7, "conjecture": 0.5}

    def consider(self, situation: Situation) -> Optional[Proposal]:
        rec = situation.recall
        if rec is None or rec.hit is None:
            return None
        best, score = self._best(situation, rec)
        if best is None:
            return None
        # An undecided recall is a real "I am not sure" — bid, but weakly and honestly.
        decided = score >= getattr(situation.brain, "memory", None).recall_threshold \
            if getattr(situation.brain, "memory", None) is not None else rec.decided
        conf = _clamp01(score) * (1.0 if decided else 0.4) * self.kind_weight.get(best.kind, 0.8)
        if conf < self.floor:
            return None              # silence beats offering noise as if it were evidence
        return Proposal(
            source=self.name, content=best.text, confidence=conf,
            verifiable=False,                      # retrieved, not proven
            novelty=0.1, tags=self.tags,
            evidence=[best.key] + [k for k, _ in rec.associated if k != best.key],
            rationale=(f"recalled {best.key} ({best.kind}, score {score:.2f}"
                       f"{'' if decided else ', low confidence'})"))

    def _best(self, situation: Situation, rec: Any) -> tuple:
        """Pick the most *useful* recalled memory, not merely the most lexically similar one.

        A stray episode often sits closer to the wording of a question than the fact that
        answers it. Ranking similarity × kind lets what she *knows* win over what merely
        *happened*, which is the difference between answering and echoing.
        """
        brain = situation.brain
        memory = getattr(brain, "memory", None)
        if memory is None:
            return rec.hit, float(rec.score)
        ranked = memory.candidates(situation.stimulus, k=5)
        if not ranked:
            return rec.hit, float(rec.score)
        trace, score = max(ranked,
                           key=lambda p: p[1] * self.kind_weight.get(p[0].kind, 0.8))
        return trace, float(score)


# --------------------------------------------------------------------------- #
# Graph — what she has learned relates to what
# --------------------------------------------------------------------------- #
class GraphSpecialist(SpecialistModule):
    """Bids with the associations her own concept graph has grown. Learned, not declared."""

    name = "graph"
    tags = frozenset({"association", "concepts"})
    # Associations are weaker evidence than something she actually recalls or can derive — they
    # say *that* two things go together, never *what* is true of them.
    priority = 0.3
    floor = 0.1                      # below this an edge is noise from a single co-occurrence

    def consider(self, situation: Situation) -> Optional[Proposal]:
        brain = situation.brain
        if brain is None or not situation.concepts:
            return None
        seen: set = set()
        phrases: List[str] = []
        strongest = 0.0
        for concept in situation.concepts[:4]:
            partners: List[str] = []
            for other, weight in brain.related(concept, k=3):
                if weight < self.floor:
                    continue
                # the graph is undirected — a~b and b~a are one association, not two
                edge = (concept, other) if concept <= other else (other, concept)
                if edge in seen:
                    continue
                seen.add(edge)
                partners.append(other)
                strongest = max(strongest, weight)
            if partners:
                phrases.append(f"{concept} goes with {', '.join(partners)}")
        if not phrases:
            return None
        # Rendered as language, not as edge telemetry: whatever wins the broadcast may be
        # remembered and said out loud, so it has to be something a person could read.
        return Proposal(
            source=self.name, content="; ".join(phrases[:4]),
            confidence=_clamp01(strongest), verifiable=False,
            novelty=_clamp01(situation.novelty), tags=self.tags,
            evidence=situation.concepts[:4],
            rationale=f"{len(seen)} learned associations, strongest {strongest:.2f}")


# --------------------------------------------------------------------------- #
# Derivation — the verifiable one
# --------------------------------------------------------------------------- #
class DerivationSpecialist(SpecialistModule):
    """Bids only when it can actually *derive* an answer — logic, algebra, physics, chemistry.

    Wraps :class:`~nyxara.mind.first_principles.FirstPrinciplesEngine`, which routes to the right
    sub-engine itself. When the derivation checks out, this is the one specialist that reports
    ``verifiable=True``, and that is what lets it beat a more confident guess downstream.
    """

    name = "derivation"
    tags = frozenset({"logic", "maths", "physics", "proof"})
    priority = 0.9

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._tried = engine is not None

    def _get(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.mind.first_principles import FirstPrinciplesEngine
                self._engine = FirstPrinciplesEngine()
            except Exception:  # noqa: BLE001 — a missing engine is a silent specialist
                self._engine = None
        return self._engine

    def available(self) -> bool:
        return self._get() is not None

    def consider(self, situation: Situation) -> Optional[Proposal]:
        engine = self._get()
        if engine is None or not situation.stimulus.strip():
            return None
        derivation = engine.derive(situation.stimulus)
        if derivation is None:
            return None                            # nothing derivable here — stay silent
        result = str(getattr(derivation, "result", "") or "").strip()
        if not result:
            return None
        verified = bool(getattr(derivation, "verified", False))
        steps = getattr(derivation, "steps", []) or []
        return Proposal(
            source=self.name, content=result,
            confidence=_clamp01(float(getattr(derivation, "confidence", 0.5))),
            verifiable=verified,                   # only a *checked* derivation claims this
            novelty=0.4, urgency=0.0, tags=self.tags,
            evidence=[str(getattr(derivation, "domain", "?"))]
                     + [str(getattr(s, "claim", s))[:80] for s in steps[:3]],
            rationale=(f"{getattr(derivation, 'domain', '?')} derivation, "
                       f"{len(steps)} steps, {'verified' if verified else 'unverified'}"))


# --------------------------------------------------------------------------- #
# Creative — the divergent one
# --------------------------------------------------------------------------- #
class CreativeSpecialist(SpecialistModule):
    """Bids with a genuinely different angle. Loud on open questions, quiet on factual ones."""

    name = "creative"
    tags = frozenset({"creative", "divergent"})
    priority = 0.4

    # An open/"how might we" shape is where divergence earns its place; a factual lookup is not.
    _OPEN = re.compile(r"\b(how|why|what if|could|might|idea|design|improve|invent|instead)\b",
                       re.I)

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._tried = engine is not None

    def _get(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.mind.creative import CreativeEngine
                self._engine = CreativeEngine()
            except Exception:  # noqa: BLE001
                self._engine = None
        return self._engine

    def available(self) -> bool:
        return self._get() is not None

    def consider(self, situation: Situation) -> Optional[Proposal]:
        engine = self._get()
        text = situation.stimulus.strip()
        if engine is None or not text or not self._OPEN.search(text):
            return None
        idea = engine.best(text)
        if idea is None:
            return None
        content = str(getattr(idea, "content", "") or "").strip()
        if not content:
            return None
        novelty = _clamp01(float(getattr(idea, "novelty", 0.5)))
        useful = _clamp01(float(getattr(idea, "usefulness", 0.5)))
        return Proposal(
            source=self.name, content=content,
            confidence=_clamp01(0.5 * useful + 0.2),   # a good idea is still a proposal, not a fact
            verifiable=False, novelty=novelty, tags=self.tags,
            evidence=[str(getattr(idea, "technique", "creative"))],
            rationale=f"{getattr(idea, 'technique', 'creative')}: "
                      f"novelty {novelty:.2f}, usefulness {useful:.2f}")


# --------------------------------------------------------------------------- #
# Ethics — the one that speaks up when something is at stake
# --------------------------------------------------------------------------- #
class EthicsSpecialist(SpecialistModule):
    """Reads the turn through three frameworks and leaves their disagreement *visible*.

    It bids only when the turn proposes doing something, and — importantly — it never refuses.
    Refusing is the sovereign gate's job alone; this specialist only makes the moral reading
    part of what reaches awareness.
    """

    name = "ethics"
    tags = frozenset({"ethics", "moral"})
    priority = 0.8

    _ACTIONY = re.compile(
        r"\b(delete|remove|wipe|kill|shut|disable|send|share|publish|buy|pay|install|"
        r"overwrite|expose|leak|ignore|hide|force|should i|can you)\b", re.I)

    def __init__(self, frameworks: Any = None) -> None:
        self._frameworks = frameworks
        self._tried = frameworks is not None

    def _get(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.mind.moral import Consequentialist, Deontological, VirtueEthics
                self._frameworks = [Consequentialist(), Deontological(), VirtueEthics()]
            except Exception:  # noqa: BLE001
                self._frameworks = None
        return self._frameworks

    def available(self) -> bool:
        return bool(self._get())

    def consider(self, situation: Situation) -> Optional[Proposal]:
        frameworks = self._get()
        text = situation.stimulus.strip()
        if not frameworks or not text or not self._ACTIONY.search(text):
            return None
        from nyxara.mind.moral import MoralAction
        action = MoralAction(description=text)
        readings, verdicts = [], []
        for framework in frameworks:
            judgment = framework.evaluate(action)
            verdicts.append(str(getattr(judgment.verdict, "value", judgment.verdict)))
            readings.append(f"{judgment.framework}: {verdicts[-1]}")
        disagreement = len(set(verdicts)) > 1
        return Proposal(
            source=self.name, content="; ".join(readings),
            # Disagreement between frameworks is *information*, not noise — report it as
            # lower confidence rather than laundering it into one confident verdict.
            confidence=0.4 if disagreement else 0.75,
            verifiable=False, novelty=0.3,
            urgency=0.6 if disagreement else 0.3, tags=self.tags,
            evidence=verdicts,
            rationale=("frameworks disagree" if disagreement else "frameworks agree"))


def default_specialists(brain: Any = None) -> List[SpecialistModule]:
    """The standard cast. ``brain`` is unused here but kept so callers read symmetrically."""
    return [
        MemorySpecialist(),
        GraphSpecialist(),
        DerivationSpecialist(),
        CreativeSpecialist(),
        EthicsSpecialist(),
    ]
