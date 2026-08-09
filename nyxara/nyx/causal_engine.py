"""NYXARA · nyx/causal_engine.py — "because", not "alongside" (⇄, NYX V.02).

Her concept graph is built on Hebbian co-activation, and co-activation is symmetric. Every edge
in it says *these two turn up together* and none of them says *this one brings the other about*.
That is not a small gap: it is the whole of the difference between correlation and causation,
sitting inside the structure she reasons over.

The machinery to do better is already here and is genuinely Pearl-shaped —
:class:`~nyxara.mind.causal_world_model.CausalWorldModel` has ``do()``, back-door and front-door
adjustment, confounder search, counterfactuals and necessity/sufficiency. What was missing is
three things on the NYX side, and this module is those three:

1. **She never asks.** Nothing in :meth:`~nyxara.nyx.brain.NyxBrain.think` has ever called
   ``do()``. :class:`CausalReasoner` answers a *why* or an *if* turn by intervening rather than
   by looking at what co-occurs.
2. **Her edges cannot say "because".** :meth:`CausalReasoner.wire` writes **directed edges with
   a mechanism** into the graph, kept in a *separate store* from the Hebbian weights — so
   "turned up together" and "brought about" can never be read as the same thing.
3. **She never asks why she was wrong.** :meth:`CausalReasoner.root_cause` runs the same
   machinery over her *own* failures and hands the result to :mod:`nyxara.nyx.omega`.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **Identification rests on the graph and on assumptions.** Given the wrong graph, do-calculus
  returns a confident wrong answer — that is a property of the method, not a bug in this file.
  So every verdict carries its **strategy** (back-door, front-door, or none), the **confounder**
  it found, and its **confidence**, and none of those is optional.
* This is the real answer to *"correlation is not causation"*. It is **not** magic: it stands on
  the observations she has and the structure she believes, and it says so on every answer.
* A causal edge is **never** created from co-occurrence alone. It requires a verdict from the
  world model, and it is stored apart from the Hebbian graph so nothing can quietly conflate
  the two.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["CausalEdge", "Answer", "CausalReasoner"]

#: A turn that is asking *why*, or *what if* — the only kind an intervention answers.
_WHY = re.compile(r"\b(why|because|cause[sd]?|causing|reason|due\s+to|leads?\s+to|"
                  r"kyun|kyu|kyon|kyunki|wajah|क्यों|क्योंकि|वजह)\b", re.I)
_IF = re.compile(r"\b(if|what\s+if|suppose|were\s+to|would\s+happen|agar|अगर)\b", re.I)


@dataclass
class CausalEdge:
    """A directed edge that says *brings about* — and what it stands on."""

    cause: str = ""
    effect: str = ""
    mechanism: str = ""            # in her words, from the model's own explanation
    strength: float = 0.0
    confidence: float = 0.0
    strategy: str = ""             # back-door | front-door | direct | none
    confounder: str = ""
    at: float = field(default_factory=time.time)

    def render(self) -> str:
        tail = f" (confounded by {self.confounder})" if self.confounder else ""
        return (f"{self.cause} ⇒ {self.effect}  [{self.strategy or 'unidentified'}, "
                f"conf {self.confidence:.2f}]{tail}")

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "mechanism": self.mechanism,
                "strength": round(self.strength, 6),
                "confidence": round(self.confidence, 4), "strategy": self.strategy,
                "confounder": self.confounder, "at": round(self.at, 3)}


@dataclass
class Answer:
    """What an intervention actually said — with everything needed to distrust it."""

    question: str = ""
    cause: str = ""
    effect: str = ""
    causal: bool = False
    effect_size: float = 0.0       # the do(X) effect, not the correlation
    correlation: float = 0.0       # what the Hebbian graph would have said instead
    confidence: float = 0.0
    strategy: str = ""
    confounder: str = ""
    mechanism: str = ""
    counterfactual: str = ""
    note: str = ""

    def render(self) -> str:
        lines = [f"{self.cause} → {self.effect}: "
                 f"{'CAUSAL' if self.causal else 'not established as causal'}"]
        lines.append(f"  do({self.cause})   : {self.effect_size:+.4f}")
        lines.append(f"  co-occurrence  : {self.correlation:+.4f}   "
                     f"(what association alone would have said)")
        identified = self.strategy and self.strategy != "none"
        lines.append(f"  identification : "
                     f"{self.strategy if identified else 'none — this is not identified'}")
        if self.confounder:
            lines.append(f"  confounder     : {self.confounder}")
        lines.append(f"  confidence     : {self.confidence:.2f}")
        if self.mechanism:
            lines.append(f"  mechanism      : {self.mechanism}")
        if self.counterfactual:
            lines.append(f"  counterfactual : {self.counterfactual}")
        lines.append(f"  {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "cause": self.cause, "effect": self.effect,
                "causal": self.causal, "effect_size": round(self.effect_size, 6),
                "correlation": round(self.correlation, 6),
                "confidence": round(self.confidence, 4), "strategy": self.strategy,
                "confounder": self.confounder, "mechanism": self.mechanism,
                "counterfactual": self.counterfactual, "note": self.note}


class CausalReasoner:
    """Intervene instead of correlating — and keep "because" apart from "alongside"."""

    def __init__(self, brain: Any = None, *, min_confidence: float = 0.3,
                 max_edges: int = 2048, wire_to_graph: bool = True) -> None:
        self.brain = brain
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_edges = max(8, int(max_edges))
        self.wire_to_graph = bool(wire_to_graph)

        #: Directed causal edges, kept **apart** from the graph's symmetric Hebbian weights.
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.asked = 0
        self.identified = 0
        self.refused = 0
        self._model: Any = None

    # ---- the world model underneath ------------------------------------------ #
    def model(self) -> Any:
        if self._model is None:
            try:
                from nyxara.mind.causal_world_model import CausalWorldModel
                self._model = CausalWorldModel()
            except Exception:  # noqa: BLE001 — without it she has only association, and says so
                self._model = None
        return self._model

    def available(self) -> bool:
        return self.model() is not None

    def observe(self, concepts: Any) -> int:
        """Feed a turn's concepts in as a co-occurrence snapshot. Structure comes later."""
        try:
            model = self.model()
            if model is None:
                return 0
            labels = [str(c).strip().lower() for c in (concepts or []) if str(c).strip()]
            if not labels:
                return 0
            model.observe_snapshot(labels)
            return len(labels)
        except Exception:  # noqa: BLE001
            return 0

    # ---- does this turn even want an intervention? ---------------------------- #
    @staticmethod
    def applies(stimulus: str) -> bool:
        """A *why* or a *what-if*. A question of fact has no intervention to run."""
        try:
            text = str(stimulus or "")
            return bool(_WHY.search(text) or _IF.search(text))
        except Exception:  # noqa: BLE001
            return False

    # ---- the question --------------------------------------------------------- #
    def ask(self, cause: str, effect: str, *, question: str = "") -> Answer:
        """``do(cause)`` and what it does to ``effect`` — with how it was identified."""
        out = Answer(question=question or f"does {cause} cause {effect}?",
                     cause=str(cause or "").strip().lower(),
                     effect=str(effect or "").strip().lower())
        try:
            if not out.cause or not out.effect:
                out.note = "a causal question needs two things"
                return out
            model = self.model()
            if model is None:
                out.note = ("no causal world model is available, so all I have is "
                            "co-occurrence — and co-occurrence is not causation")
                return out
            self.asked += 1
            out.correlation = self._association(out.cause, out.effect)

            verdict = model.is_causal(out.cause, out.effect)
            out.causal = bool(getattr(verdict, "is_causal", False))
            out.confidence = float(getattr(verdict, "confidence", 0.0))
            evidence = dict(getattr(verdict, "evidence", {}) or {})
            out.strategy = str(evidence.get("strategy")
                               or evidence.get("adjustment")
                               or ("direct" if out.causal else "none"))
            out.confounder = str(evidence.get("confounder") or "")
            out.mechanism = str(getattr(verdict, "reason", "") or "")

            try:
                # ``do()`` propagates over the *structural* links, and a verdict alone does not
                # write one. Evaluating this pair first is what makes the effect size a real
                # number rather than a silent zero.
                model.update_links_for([out.cause, out.effect])
                out.effect_size = float(model.do({out.cause: 1.0}, out.effect) or 0.0)
            except Exception:  # noqa: BLE001 — an unidentified effect is simply zero, honestly
                out.effect_size = 0.0

            try:
                counter = model.counterfactual(out.cause, out.effect)
                out.counterfactual = str(getattr(counter, "narrative", "")
                                         or getattr(counter, "summary", "") or "")
            except Exception:  # noqa: BLE001
                out.counterfactual = ""

            if out.causal and out.confidence >= self.min_confidence:
                self.identified += 1
                self.wire(out)
                out.note = ("identified by intervention, not by co-occurrence — but "
                            "identification rests on the graph and the assumptions behind it, "
                            "so read the strategy and the confounder before trusting it")
            else:
                self.refused += 1
                out.note = ("not established as causal. They may still co-occur — "
                            f"association is {out.correlation:+.3f} — and that is exactly the "
                            f"thing this is here to keep separate")
            return out
        except Exception as exc:  # noqa: BLE001
            out.note = f"{type(exc).__name__}: {exc}"
            return out

    def _association(self, a: str, b: str) -> float:
        """What the Hebbian graph would have said — reported *beside* the causal answer."""
        try:
            graph = getattr(self.brain, "graph", None)
            return float(graph.weight(a, b)) if graph is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- "because" stored apart from "alongside" -------------------------------- #
    def wire(self, answer: Answer) -> Optional[CausalEdge]:
        """Record a directed causal edge. Never derived from co-occurrence alone."""
        try:
            if not answer.causal or answer.confidence < self.min_confidence:
                return None
            edge = CausalEdge(cause=answer.cause, effect=answer.effect,
                              mechanism=answer.mechanism, strength=answer.effect_size,
                              confidence=answer.confidence, strategy=answer.strategy,
                              confounder=answer.confounder)
            self.edges[(edge.cause, edge.effect)] = edge
            if len(self.edges) > self.max_edges:
                weakest = min(self.edges.values(), key=lambda e: e.confidence)
                self.edges.pop((weakest.cause, weakest.effect), None)
            if self.wire_to_graph:
                # The concepts are made to exist so a causal claim about them is reachable —
                # but the *weight* stays where it was. A causal edge does not strengthen an
                # association, because they are not the same relation.
                graph = getattr(self.brain, "graph", None)
                if graph is not None:
                    graph.activate([edge.cause, edge.effect], learn=False)
            return edge
        except Exception:  # noqa: BLE001
            return None

    def causes_of(self, effect: str) -> List[CausalEdge]:
        """Directed in-edges — what brings this about, not what turns up with it."""
        try:
            key = str(effect or "").strip().lower()
            out = [e for e in self.edges.values() if e.effect == key]
            out.sort(key=lambda e: -e.confidence)
            return out
        except Exception:  # noqa: BLE001
            return []

    def effects_of(self, cause: str) -> List[CausalEdge]:
        try:
            key = str(cause or "").strip().lower()
            out = [e for e in self.edges.values() if e.cause == key]
            out.sort(key=lambda e: -e.confidence)
            return out
        except Exception:  # noqa: BLE001
            return []

    # ---- her own failures --------------------------------------------------------- #
    def root_cause(self, *, k: int = 3) -> List[Answer]:
        """Run the same machinery over her *own* mistakes, for omega to act on.

        Meta-cognition already knows which faculty is weakest. This asks what actually brings
        that about, rather than noting that it correlates with everything.
        """
        out: List[Answer] = []
        try:
            metacog = getattr(self.brain, "metacog", None)
            if metacog is None:
                return out
            stats = metacog.stats() or {}
            weakest = str(stats.get("weakest") or "")
            if not weakest:
                return out
            for source in list((stats.get("specialists") or {}))[: max(1, int(k))]:
                if source == weakest:
                    continue
                out.append(self.ask(source, weakest,
                                    question=f"does {source} bring about {weakest} failing?"))
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- introspection ------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "available": self.available(),
                "asked": self.asked, "identified": self.identified, "refused": self.refused,
                "causal_edges": len(self.edges),
                "min_confidence": self.min_confidence,
                "note": ("do-calculus identifies an effect only when the graph is right — "
                         "given the wrong graph it returns a CONFIDENT WRONG ANSWER, which is "
                         "a property of the method and not a bug here. So every verdict "
                         "carries its identification strategy, its confounder and its "
                         "confidence, and the co-occurrence it is being distinguished from is "
                         "printed beside it. Causal edges are stored APART from the Hebbian "
                         "weights so 'turned up together' and 'brought about' can never be "
                         "read as the same thing"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        try:
            if not self.edges:
                return ("I hold no causal edges yet. They are never derived from "
                        "co-occurrence — each one needs a verdict from the world model.")
            lines = [f"{len(self.edges)} directed causal edge(s), kept apart from association:"]
            for edge in sorted(self.edges.values(), key=lambda e: -e.confidence)[:16]:
                lines.append("  " + edge.render())
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence ----------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"asked": self.asked, "identified": self.identified, "refused": self.refused,
                "edges": [e.to_dict() for e in self.edges.values()]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.asked = int(data.get("asked", 0))
            self.identified = int(data.get("identified", 0))
            self.refused = int(data.get("refused", 0))
            self.edges = {}
            for raw in data.get("edges", []) or []:
                if not isinstance(raw, dict):
                    continue
                edge = CausalEdge(cause=str(raw.get("cause", "")),
                                  effect=str(raw.get("effect", "")),
                                  mechanism=str(raw.get("mechanism", "")),
                                  strength=float(raw.get("strength", 0.0)),
                                  confidence=float(raw.get("confidence", 0.0)),
                                  strategy=str(raw.get("strategy", "")),
                                  confounder=str(raw.get("confounder", "")),
                                  at=float(raw.get("at", time.time())))
                if edge.cause and edge.effect:
                    self.edges[(edge.cause, edge.effect)] = edge
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
