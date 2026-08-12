"""NYXARA · nyx001/lingua/grounding.py — language last, not first (🔗, the charter's ordering).

The charter is explicit: *"Language is not the starting point. The progression is: Environment →
Perception → Concepts → World Model → Language Grounding."*

That ordering is the entire content of this module. A word acquires meaning here by being
**associated with a concept the system formed on its own** — and a concept is formed from
experience, which comes from perception, which comes from the environment. Nothing is grounded by
being defined in terms of other words, because a closed loop of word-to-word definitions is exactly
the thing that looks like understanding and is not.

:meth:`Grounding.ground` binds a token to whatever concept was active when it was encountered, and
strengthens that binding on repetition. A binding carries:

* **support** — how many times word and concept co-occurred;
* **exclusivity** — how much of this word's mass sits on this one concept, versus spread across
  many. A word bound to eleven concepts equally is not grounded in any of them, and this number is
  what says so;
* **grounded** — support and exclusivity both clearing their thresholds. Anything short of that is
  reported as *ungrounded*, and :meth:`meaning` returns ``None`` for it rather than the
  best-of-a-bad-set.

That last refusal is the point. A grounding layer that always returns its closest match will always
look like it understands every word it has seen once.

:meth:`describe` runs the ordering in reverse — concept → the words that reliably denote it —
which is the only direction in which this system can say anything about what it knows without
borrowing vocabulary it never earned.

Honest: this is co-occurrence binding between token ids and concept clusters. It is grounding in
the technical sense that symbols connect to non-symbolic structure derived from experience, and it
is *not* comprehension. A word grounded in a concept the system formed from a hash-sketch
perception of text is grounded in a shallow thing — the binding is only ever as deep as what
Layers 1 and 4 managed to carve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Binding", "Grounding"]


@dataclass
class Binding:
    """One word-to-concept link, with the two numbers that decide whether it means anything."""

    token: str = ""
    concept: str = ""
    support: float = 0.0
    exclusivity: float = 0.0
    first_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"token": self.token[:60], "concept": self.concept,
                "support": round(self.support, 3), "exclusivity": round(self.exclusivity, 4)}


class Grounding:
    """Binds words to concepts the system formed itself — the last step, never the first."""

    def __init__(self, substrate: Any, *, min_support: float = 3.0,
                 min_exclusivity: float = 0.5, max_tokens: int = 8192,
                 decay: float = 0.999) -> None:
        self.substrate = substrate
        self.min_support = float(min_support)
        self.min_exclusivity = float(min_exclusivity)
        self.max_tokens = int(max_tokens)
        self.decay = float(decay)
        self._counts: Dict[str, Dict[str, float]] = {}       # token -> concept -> weight
        self.groundings = 0
        self.evicted = 0

    # ---- bind ---- #
    def ground(self, token: str, concept: Any) -> Optional[Binding]:
        """Associate a word with the concept that was active when it was encountered."""
        try:
            name = getattr(concept, "name", None) or (str(concept) if concept else None)
            tok = str(token).strip().lower()
            if not tok or not name:
                return None
            if tok not in self._counts and len(self._counts) >= self.max_tokens:
                self._evict()
            row = self._counts.setdefault(tok, {})
            for k in row:
                row[k] *= self.decay                          # older bindings fade, measurably
            row[name] = row.get(name, 0.0) + 1.0
            self.groundings += 1
            return self._binding(tok, name)
        except Exception:  # noqa: BLE001
            return None

    def ground_text(self, text: str, concept: Any) -> List[Binding]:
        """Bind every word of an observation to the concept it arrived under."""
        out = []
        for word in str(text).lower().split():
            w = "".join(ch for ch in word if ch.isalnum())
            if len(w) > 1:
                b = self.ground(w, concept)
                if b is not None:
                    out.append(b)
        return out

    def _evict(self) -> None:
        if not self._counts:
            return
        victim = min(self._counts, key=lambda t: sum(self._counts[t].values()))
        self._counts.pop(victim, None)
        self.evicted += 1

    def _binding(self, token: str, concept: str) -> Binding:
        row = self._counts.get(token, {})
        total = sum(row.values())
        support = row.get(concept, 0.0)
        return Binding(token=token, concept=concept, support=support,
                       exclusivity=float(support / total) if total > 0 else 0.0)

    # ---- read ---- #
    def is_grounded(self, token: str) -> bool:
        b = self.best_binding(token)
        return bool(b and b.support >= self.min_support and b.exclusivity >= self.min_exclusivity)

    def best_binding(self, token: str) -> Optional[Binding]:
        row = self._counts.get(str(token).strip().lower())
        if not row:
            return None
        concept = max(row, key=lambda c: row[c])
        return self._binding(str(token).strip().lower(), concept)

    def meaning(self, token: str) -> Optional[str]:
        """The concept a word denotes — or ``None`` when it has not earned a grounding.

        Returning ``None`` rather than the closest match is the whole design. A layer that always
        answers will always appear to understand every word it has ever seen once.
        """
        b = self.best_binding(token)
        if b is None or b.support < self.min_support or b.exclusivity < self.min_exclusivity:
            return None
        return b.concept

    def describe(self, concept: Any, *, k: int = 5) -> List[Tuple[str, float]]:
        """Concept → the words that reliably denote it. The ordering, run backwards."""
        name = getattr(concept, "name", None) or str(concept)
        out: List[Tuple[str, float]] = []
        for tok, row in self._counts.items():
            w = row.get(name, 0.0)
            total = sum(row.values())
            if w >= self.min_support and total > 0 and (w / total) >= self.min_exclusivity:
                out.append((tok, round(w / total, 4)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out[: max(1, int(k))]

    def vocabulary(self) -> List[str]:
        """Every word this system has actually grounded. Usually much smaller than what it saw."""
        return [t for t in self._counts if self.is_grounded(t)]

    def coverage(self) -> Optional[float]:
        """Fraction of encountered words that are genuinely grounded. ``None`` when none seen."""
        if not self._counts:
            return None
        return float(len(self.vocabulary()) / len(self._counts))

    def stats(self) -> Dict[str, Any]:
        return {"tokens_seen": len(self._counts), "grounded": len(self.vocabulary()),
                "coverage": self.coverage(), "groundings": self.groundings,
                "evicted": self.evicted, "min_support": self.min_support,
                "min_exclusivity": self.min_exclusivity}

    def to_dict(self) -> Dict[str, Any]:
        top = sorted(self._counts.items(), key=lambda kv: sum(kv[1].values()),
                     reverse=True)[:512]
        return {"groundings": self.groundings, "evicted": self.evicted,
                "counts": {t: {c: round(w, 3) for c, w in row.items()} for t, row in top}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.groundings = int(d.get("groundings", 0))
            self.evicted = int(d.get("evicted", 0))
            for t, row in (d.get("counts") or {}).items():
                self._counts[str(t)] = {str(c): float(w) for c, w in row.items()}
        except Exception:  # noqa: BLE001
            pass
