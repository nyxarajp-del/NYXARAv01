"""NYXARA · nyx/ground.py — what her words are actually *about* (🍎, NYX V.01).

The gap between a language model and a mind is the symbol grounding problem (Harnad). To a
statistical model, "apple" is a slot in a table of what word tends to follow what. Nothing in
the world stands behind it — no colour, no weight, no taste, nothing that falls when dropped.
It can talk about apples forever without ever meaning one.

This module is where NYX's symbols get their referents. When she meets "apple", it fires across
**taste** (sweet, tart), **vision** (red, round), **touch** (smooth), **physics** (it has weight,
and it *falls*), **affordance** (you can eat it). Meaning is that multimodal pattern, not
co-occurrence statistics — and because it is a real vector, "how similar are apple and orange"
has a real answer, and so does "do I actually understand this word, or am I just repeating it?"

Two things it adds over the faculties it composes:

* **Grounding binds into the graph.** A grounded concept's features become nodes linked to it in
  :mod:`nyxara.nyx.graph`, so understanding accretes in the same structure everything else does.
* **A measured answer to "do I understand this?"** — :meth:`WorldGrounding.understanding` returns
  how many senses a word fires across and how confident that is. An ungrounded word reports
  itself ungrounded instead of being quietly treated as known.

She can also **learn a word by reading a sentence about it** — no model, no network, pure
stdlib (``cognition/grounded_understanding``'s deterministic path). That is the honest floor.

Honest framing:

* This is a **perceptual-feature model**, not perception. She has no eyes; "red" is a symbol in
  a descriptor ontology, grounded in a seed lexicon and in text she has read. It is a real and
  useful referent structure, and it is *not* sensory experience.
* Coverage is bounded by the seed lexicon plus whatever she has read. Outside that she reports
  ``grounded=False`` — the point is that she can *tell*, not that she knows everything.

Composes :class:`~nyxara.cognition.grounded_understanding.GroundedLexicon` and, for verbs,
:mod:`nyxara.cognition.language_grounding`. Fail-soft throughout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["Understanding", "Grounded", "WorldGrounding"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass
class Understanding:
    """How well she actually understands one word — measured, not asserted."""

    word: str
    grounded: bool = False
    senses: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence: float = 0.0
    neighbours: List[Tuple[str, float]] = field(default_factory=list)
    explanation: str = ""

    @property
    def depth(self) -> int:
        """Across how many senses the word fires. One sense is a label; five is a referent."""
        return len(self.senses)

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "grounded": self.grounded, "depth": self.depth,
                "senses": {k: dict(v) for k, v in self.senses.items()},
                "confidence": round(self.confidence, 4),
                "neighbours": [[n, round(s, 4)] for n, s in self.neighbours],
                "explanation": self.explanation}


@dataclass
class Grounded:
    """The result of grounding a whole stimulus."""

    understood: List[str] = field(default_factory=list)
    ungrounded: List[str] = field(default_factory=list)
    learned: List[str] = field(default_factory=list)
    coverage: float = 0.0            # fraction of the turn's concepts she actually has referents for

    def to_dict(self) -> Dict[str, Any]:
        return {"understood": self.understood, "ungrounded": self.ungrounded,
                "learned": self.learned, "coverage": round(self.coverage, 4)}


class WorldGrounding:
    """Gives NYX's symbols referents, binds them into her graph, and says when they have none."""

    def __init__(self, *, lexicon: Any = None, read_unknown: bool = True,
                 max_new_per_turn: int = 4, bind_to_graph: bool = True) -> None:
        self._lexicon = lexicon
        self._tried = lexicon is not None
        self.read_unknown = bool(read_unknown)
        self.max_new_per_turn = max(0, int(max_new_per_turn))
        self.bind_to_graph = bool(bind_to_graph)
        self.learned: List[str] = []

    def _get(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.cognition.grounded_understanding import GroundedLexicon
                self._lexicon = GroundedLexicon()
            except Exception:  # noqa: BLE001 — without it every word is honestly ungrounded
                self._lexicon = None
        return self._lexicon

    def available(self) -> bool:
        """Honest capability report — False when no lexicon could be built."""
        return self._get() is not None

    # ---- one word --------------------------------------------------------- #
    def understanding(self, word: str) -> Understanding:
        """What she actually has behind this word. ``grounded=False`` is a real answer."""
        out = Understanding(word=str(word or ""))
        lexicon = self._get()
        if lexicon is None or not out.word:
            return out
        try:
            activation = lexicon.activate(out.word)
            out.grounded = bool(getattr(activation, "grounded", False))
            if not out.grounded:
                return out
            out.senses = {getattr(sense, "value", str(sense)): dict(features)
                          for sense, features in (getattr(activation, "senses", {}) or {}).items()}
            out.confidence = _clamp01(float(getattr(activation, "confidence", 0.0)))
            out.neighbours = [(str(n), float(s)) for n, s in lexicon.neighbors(out.word, top=4)]
            out.explanation = str(lexicon.explain(out.word) or "")
            return out
        except Exception:  # noqa: BLE001 — a grounding failure is "ungrounded", never a crash
            return out

    def understands(self, word: str) -> bool:
        """Does she have a referent for this word at all?"""
        return self.understanding(word).grounded

    def similarity(self, a: str, b: str) -> float:
        """How alike two words are *in what they refer to* — not in how they are spelled."""
        lexicon = self._get()
        if lexicon is None:
            return 0.0
        try:
            return float(lexicon.similarity(str(a), str(b)))
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def defines(word: str, text: str) -> bool:
        """Does ``text`` actually *define* ``word``, or merely mention it?

        This distinction is the whole honesty of learning-by-reading. "an apple falls because of
        gravity" mentions "because"; it does not say what "because" means. Without this check
        every word in every sentence gets grounded in whatever else that sentence contained,
        and the lexicon fills up with confident nonsense.
        """
        try:
            word, text = str(word or "").strip().lower(), str(text or "").lower()
            if not word or not text:
                return False
            pattern = (rf"\b(?:an?|the)?\s*{re.escape(word)}\s+"
                       r"(?:is|are|was|were|means?|refers?\s+to|denotes?)\b")
            return re.search(pattern, text) is not None
        except Exception:  # noqa: BLE001
            return False

    def learn(self, word: str, description: str, *, require_definition: bool = False) -> bool:
        """Ground a new word by reading a sentence about it. No model, no network.

        Called directly, this trusts the caller that ``description`` describes ``word``. The
        automatic path (:meth:`understand`) sets ``require_definition`` so a passing mention is
        never mistaken for a definition.
        """
        lexicon = self._get()
        if lexicon is None or not str(word).strip() or not str(description).strip():
            return False
        if require_definition and not self.defines(word, description):
            return False
        try:
            report = lexicon.learn_from_text(str(description), str(word))
            got = bool(report and str(word) in (report.get("grounded") or []))
            if got:
                self.learned.append(str(word))
                del self.learned[:-256]
            return got
        except Exception:  # noqa: BLE001
            return False

    # ---- a whole turn ----------------------------------------------------- #
    def understand(self, concepts: Any, *, text: str = "", graph: Any = None) -> Grounded:
        """Ground every concept in a turn, learn what it can, and bind the rest into the graph."""
        out = Grounded()
        lexicon = self._get()
        if lexicon is None:
            return out
        try:
            from nyxara.nyx.graph import concepts_in
            words = concepts_in(concepts) if isinstance(concepts, str) else \
                [str(c).strip().lower() for c in (concepts or []) if str(c).strip()]
            if not words:
                return out

            for word in words:
                if self.understands(word):
                    out.understood.append(word)
                    continue
                # A word she has just met, in a sentence that *defines* it, is learnable now.
                if self.read_unknown and text and len(out.learned) < self.max_new_per_turn \
                        and self.learn(word, text, require_definition=True):
                    out.understood.append(word)
                    out.learned.append(word)
                else:
                    out.ungrounded.append(word)

            out.coverage = len(out.understood) / float(len(words))
            if self.bind_to_graph and graph is not None:
                self._bind(out.understood, graph)
            return out
        except Exception:  # noqa: BLE001
            return out

    def _bind(self, words: List[str], graph: Any) -> None:
        """Let what a word *means* become structure in the graph, not a separate island.

        Grounding a word links it to its perceptual features, so "apple" and "cherry" end up
        connected through *red* and *sweet* — a route the co-occurrence edges alone never build.
        """
        try:
            for word in words[:8]:
                understanding = self.understanding(word)
                features = [feature
                            for sense_features in understanding.senses.values()
                            for feature in sense_features]
                if features:
                    # decay=False: this is the same moment as the turn that triggered it, so
                    # binding several words must not age the graph once per word.
                    graph.activate([word] + features[:6], decay=False)
        except Exception:  # noqa: BLE001 — binding is best-effort
            pass

    # ---- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        lexicon = self._get()
        try:
            return {"available": lexicon is not None,
                    "learned": len(self.learned),
                    "recently_learned": self.learned[-5:],
                    "read_unknown": self.read_unknown}
        except Exception:  # noqa: BLE001
            return {"available": False}

    def to_dict(self) -> Dict[str, Any]:
        """Persist what she has *learned*, not the seed lexicon (which rebuilds itself)."""
        lexicon = self._get()
        out: Dict[str, Any] = {"learned": list(self.learned)}
        try:
            if lexicon is not None and hasattr(lexicon, "to_dict"):
                out["lexicon"] = lexicon.to_dict()
        except Exception:  # noqa: BLE001
            pass
        return out

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.learned = [str(w) for w in (data.get("learned") or []) if str(w).strip()]
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
