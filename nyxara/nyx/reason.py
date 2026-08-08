"""NYXARA · nyx/reason.py — a seat for every domain, not just four (🧭, NYX V.02, gap 3).

NYX V.01 could derive an answer from first principles in exactly four domains — physics,
chemistry, maths, logic. Outside them,
:meth:`~nyxara.mind.first_principles.FirstPrinciplesEngine.derive` returned ``None`` and the
turn produced **silence**. Not "I am not sure"; nothing.

:class:`OpenDomainReasoner` is not a new reasoning engine. It is a **seat**: a cascade over
engines this repo already has, arranged strongest-evidence-first, where every answer carries
the tier it came from.

===================== ================================================ ==============
tier                   engine                                            ``verifiable``
===================== ================================================ ==============
``exact``              :func:`nyxara.mind.reasoning_chain.solve_chain`   yes
``derived``            :class:`~nyxara.mind.first_principles.FirstPrinciplesEngine`
                       — the four checkable domains                      yes
``induced``            :class:`~nyxara.cognition.skill_induction.SkillInductionEngine`
                       via demonstrations in the prompt                  only on held-out transfer
``mapped``             :class:`~nyxara.mind.analogy.StructureMapper` —
                       structure carried onto a domain she knows         no
``modelled``           :mod:`nyxara.mind.domain_genesis` — a theory of
                       an alien field from its own regularities          no
``composed``           :mod:`nyxara.cognition.composition`               no
``associative``        her own concept graph — the strongest path
                       between the question's own concepts               no
===================== ================================================ ==============

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* ``verifiable=True`` is set **only** when a step was actually checked — an exact chain, a
  dimensional/stoichiometric/algebraic derivation, or an induced rule that predicted a
  demonstration held out of its own induction. Everything else is a **labelled plausible
  chain**: the steps are exposed so ``/nyx why`` shows the real reasoning rather than a story
  about it, and the label is never quietly dropped.
* Outside the four checkable domains she is **not** verifiable, and says so on every answer.
  Replacing silence with a confident-sounding paragraph would be a worse failure than the
  silence was.
* When every tier declines, that is reported as a reasoned abstention with the tiers tried —
  still not a fabricated answer.

Pure standard library. Every public method is fail-soft: on any error it degrades to a null
result rather than breaking a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Step", "Chain", "OpenDomainReasoner", "TIERS"]

#: Strongest evidence first. The first tier that answers, wins.
TIERS = ("exact", "derived", "induced", "mapped", "modelled", "composed", "associative")

#: Tiers that *may* be verifiable. Even here it is never assumed: the exact chain is checked by
#: construction, and the derived tier reads the derivation's own ``verified`` certificate.
_VERIFIABLE_TIERS = frozenset({"exact"})

#: How a :class:`~nyxara.mind.generalization.GeneralizationResult` source maps onto a tier.
_SOURCE_TIER = {
    "faculty": "derived",
    "skill_induction": "induced",
    "relational_transfer": "mapped",
    "domain_genesis": "modelled",
    "open_world": "modelled",
    "composition": "composed",
}


@dataclass
class Step:
    """One move in a chain, and whether anything actually checked it."""

    text: str = ""
    engine: str = ""
    checked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "engine": self.engine, "checked": self.checked}


@dataclass
class Chain:
    """One attempt at a question: the answer, the steps, and the tier that produced them."""

    question: str = ""
    answer: str = ""
    tier: str = ""
    engine: str = ""
    steps: List[Step] = field(default_factory=list)
    verifiable: bool = False
    confidence: float = 0.0
    tried: List[str] = field(default_factory=list)
    why: str = ""

    @property
    def answered(self) -> bool:
        return bool(self.answer.strip())

    def label(self) -> str:
        """The sentence that must travel with the answer. Never optional, never dropped."""
        if not self.answered:
            return ("I could not derive this, and I am not going to produce a paragraph that "
                    "sounds like I did.")
        if self.verifiable:
            return f"Derived and checked ({self.tier}, via {self.engine})."
        return (f"Plausible, not verified ({self.tier}, via {self.engine}) — "
                f"the steps are shown; nothing checked them.")

    def render(self) -> str:
        lines = [self.answer.strip()] if self.answered else []
        for i, step in enumerate(self.steps, 1):
            mark = "✓" if step.checked else "·"
            lines.append(f"  {mark} {i}. {step.text}")
        lines.append(f"  [{self.label()}]")
        if not self.answered and self.tried:
            lines.append(f"  tried: {', '.join(self.tried)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "tier": self.tier,
                "engine": self.engine, "verifiable": self.verifiable,
                "confidence": round(self.confidence, 4), "answered": self.answered,
                "steps": [s.to_dict() for s in self.steps], "tried": list(self.tried),
                "label": self.label(), "why": self.why}


class OpenDomainReasoner:
    """Give every domain a seat. Answer where she can; label honestly where she cannot."""

    def __init__(self, brain: Any = None, *, min_confidence: float = 0.3,
                 use_generalization: bool = True, use_associative: bool = True) -> None:
        self.brain = brain
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.use_generalization = bool(use_generalization)
        self.use_associative = bool(use_associative)
        self._fp: Any = None
        self._gen: Any = None
        self._tried_fp = False
        self._tried_gen = False

        self.asked = 0
        self.answered = 0
        self.verified = 0
        self.abstained = 0
        self.by_tier: Dict[str, int] = {}

    # ---- the engines, built lazily ---------------------------------------- #
    def _first_principles(self) -> Any:
        if not self._tried_fp:
            self._tried_fp = True
            try:
                from nyxara.mind.first_principles import FirstPrinciplesEngine
                self._fp = FirstPrinciplesEngine()
            except Exception:  # noqa: BLE001 — without it the derived tier is simply absent
                self._fp = None
        return self._fp

    def _generalization(self) -> Any:
        if not self._tried_gen:
            self._tried_gen = True
            if not self.use_generalization:
                return None
            try:
                from nyxara.mind.generalization import GeneralizationEngine
                self._gen = GeneralizationEngine()
            except Exception:  # noqa: BLE001
                self._gen = None
        return self._gen

    # ---- the one call ------------------------------------------------------ #
    def solve(self, question: str) -> Chain:
        """Work down the tiers. The first that genuinely answers wins; silence is a result."""
        out = Chain(question=str(question or "").strip())
        try:
            if not out.question:
                return out
            self.asked += 1
            for tier, attempt in (("exact", self._try_exact),
                                  ("derived", self._try_derived),
                                  ("general", self._try_generalization),
                                  ("associative", self._try_associative)):
                out.tried.append(tier)
                try:
                    got = attempt(out.question)
                except Exception:  # noqa: BLE001 — one tier failing never stops the cascade
                    got = None
                if got is None or got.confidence < self.min_confidence:
                    continue
                got.question = out.question
                got.tried = list(out.tried)
                self._count(got)
                return got
            out.why = ("no tier could answer this: not an exact chain, not one of the four "
                       "checkable domains, no demonstrations to induce from, no structure to "
                       "map or model, and nothing in her graph connects its concepts yet")
            self.abstained += 1
            return out
        except Exception:  # noqa: BLE001 — reasoning never breaks a turn
            return out

    def _count(self, chain: Chain) -> None:
        self.answered += 1
        if chain.verifiable:
            self.verified += 1
        self.by_tier[chain.tier] = self.by_tier.get(chain.tier, 0) + 1

    # ---- tier 1: an exact structured chain --------------------------------- #
    @staticmethod
    def _try_exact(question: str) -> Optional[Chain]:
        try:
            from nyxara.mind.reasoning_chain import solve_chain
            res = solve_chain(question)
        except Exception:  # noqa: BLE001
            return None
        if res is None:
            return None
        steps = [Step(text=str(s), engine="reasoning_chain", checked=True)
                 for s in (getattr(res, "steps", None) or [])]
        answer = str(getattr(res, "answer", "") or getattr(res, "value", "") or "").strip()
        if not answer:
            return None
        return Chain(answer=answer, tier="exact", engine="mind.reasoning_chain",
                     steps=steps, verifiable=True,
                     confidence=float(getattr(res, "confidence", 0.9) or 0.9),
                     why="a fully determined two-step chain, solved exactly")

    # ---- tier 2: derivation in one of the four checkable domains ----------- #
    def _try_derived(self, question: str) -> Optional[Chain]:
        engine = self._first_principles()
        if engine is None:
            return None
        try:
            der = engine.derive(question)
        except Exception:  # noqa: BLE001
            return None
        if der is None:
            return None
        steps: List[Step] = []
        for line in (getattr(der, "steps", None) or []):
            text = getattr(line, "expression", None) or str(line)
            rule = getattr(line, "rule", "") or ""
            steps.append(Step(text=f"{text}" + (f"   [{rule}]" if rule else ""),
                              engine="first_principles", checked=True))
        answer = str(getattr(der, "result", "") or getattr(der, "answer", "") or "").strip()
        if not answer:
            try:
                answer = str(der.render()).strip()
            except Exception:  # noqa: BLE001
                answer = ""
        if not answer:
            return None
        domain = str(getattr(der, "domain", "") or "first principles")
        # The derivation says for itself whether it was symbolically certified. A dimensional
        # argument that reached a plausible form is a derivation and is *not* a certificate,
        # and this reads that flag rather than assuming the tier implies it.
        certified = bool(getattr(der, "verified", False))
        for step in steps:
            step.checked = certified
        return Chain(answer=answer, tier="derived",
                     engine=f"mind.first_principles ({domain})", steps=steps,
                     verifiable=certified,
                     confidence=float(getattr(der, "confidence", 0.85) or 0.85),
                     why=(f"derived and symbolically certified in {domain}" if certified else
                          f"derived in {domain}, but not symbolically certified — the steps "
                          f"follow, nothing proved the result"))

    # ---- tiers 3–6: the open-domain cascade she already owns ---------------- #
    def _try_generalization(self, question: str) -> Optional[Chain]:
        engine = self._generalization()
        if engine is None:
            return None
        try:
            res = engine.generalize(question)
        except Exception:  # noqa: BLE001
            return None
        if res is None:
            return None
        source = str(getattr(res, "source", "") or "")
        tier = _SOURCE_TIER.get(source, "modelled")
        answer = str(getattr(res, "answer", "") or "").strip()
        if not answer:
            return None

        # An induced rule counts as checked only when it predicted a demonstration held out of
        # its own induction. Fit on the examples it was given is not evidence about the probe.
        detail = dict(getattr(res, "detail", {}) or {})
        verifiable = tier in _VERIFIABLE_TIERS or (
            tier == "induced" and bool(detail.get("generalized")))

        steps = [Step(text=str(line), engine=source, checked=verifiable)
                 for line in self._steps_of(detail)]
        return Chain(answer=answer, tier=tier, engine=f"mind.generalization ({source})",
                     steps=steps, verifiable=verifiable,
                     confidence=float(getattr(res, "confidence", 0.5) or 0.5),
                     why=self._why(tier, source, detail))

    # ---- last tier: her own structure, offered as association and nothing more ------- #
    def _try_associative(self, question: str) -> Optional[Chain]:
        """The strongest path her concept graph holds between the question's own concepts.

        This is the floor that replaces silence, and it is the weakest thing in the file. It is
        **not** a derivation and is never labelled as one: it says "these are connected in what
        I have read, by this much", which is a fact about her graph rather than about the
        world. On a cold graph it declines, which is the correct answer on day one.
        """
        if not self.use_associative:
            return None
        graph = getattr(self.brain, "graph", None)
        if graph is None:
            return None
        try:
            from nyxara.nyx.graph import concepts_in
            concepts = [c for c in concepts_in(question, cap=8) if c in graph]
            if len(concepts) < 2:
                return None
            steps: List[Step] = []
            total = 0.0
            for a, b in zip(concepts, concepts[1:]):
                weight = float(graph.weight(a, b))
                bridge = ""
                if weight <= 0.0:
                    # not adjacent — is there a concept that links them?
                    shared = {n for n, _w in graph.neighbours(a, k=24)} & \
                        {n for n, _w in graph.neighbours(b, k=24)}
                    if shared:
                        bridge = max(shared, key=lambda n: graph.weight(a, n)
                                     + graph.weight(b, n))
                        weight = min(graph.weight(a, bridge), graph.weight(b, bridge))
                if weight <= 0.0:
                    continue
                via = f" via {bridge}" if bridge else ""
                steps.append(Step(text=f"{a} — {b}{via}  (association {weight:.2f})",
                                  engine="nyx.graph", checked=False))
                total += weight
            if not steps:
                return None
            strength = min(0.55, total / float(len(steps)))
            answer = ("I cannot derive this, but in what I have actually read these are "
                      "connected: " + " → ".join(concepts))
            return Chain(answer=answer, tier="associative", engine="nyx.graph", steps=steps,
                         verifiable=False, confidence=max(self.min_confidence, strength),
                         why=("association measured in her own concept graph — a fact about "
                              "what she has read, not about the world"))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _steps_of(detail: Dict[str, Any]) -> List[str]:
        """Whatever the winning engine can show of its own working. Never invented."""
        out: List[str] = []
        try:
            if detail.get("skill"):
                out.append(str(detail["skill"]))
            for key in ("transfer", "genesis", "composition"):
                blob = detail.get(key)
                if isinstance(blob, dict):
                    for field_ in ("laws", "mappings", "steps", "correspondences"):
                        for item in (blob.get(field_) or [])[:6]:
                            out.append(f"{field_[:-1]}: {item}")
            return out[:12]
        except Exception:  # noqa: BLE001
            return out

    @staticmethod
    def _why(tier: str, source: str, detail: Dict[str, Any]) -> str:
        if tier == "induced":
            return ("a rule induced from demonstrations in the question"
                    + (", confirmed on a held-out example" if detail.get("generalized")
                       else " — fitted, but not transfer-tested"))
        if tier == "mapped":
            return ("structure carried across from a domain she does know; the mapping is "
                    "consistent, which is not the same as being true here")
        if tier == "modelled":
            return ("a theory of this field built from its own regularities, with no prior "
                    "domain to borrow from — coherent, and unchecked against the world")
        if tier == "composed":
            return "a new combination of primitives she already had"
        return f"answered by {source}"

    # ---- introspection ---------------------------------------------------- #
    def available(self) -> Dict[str, bool]:
        return {"exact": True,
                "derived": self._first_principles() is not None,
                "general": self._generalization() is not None,
                "associative": self.use_associative
                and getattr(self.brain, "graph", None) is not None}

    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "asked": self.asked, "answered": self.answered,
                "verified": self.verified, "abstained": self.abstained,
                "by_tier": dict(self.by_tier),
                "available": self.available(),
                "tiers": list(TIERS),
                "note": ("verifiable=True only for an exact chain, a derivation in the four "
                         "checkable domains, or an induced rule that predicted a held-out "
                         "example; everything else is a labelled plausible chain with its "
                         "steps exposed"),
            }
        except Exception:  # noqa: BLE001
            return {}
