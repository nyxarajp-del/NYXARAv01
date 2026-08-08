"""NYXARA · nyx/selfmodel.py — what she can truthfully say about herself (🪞, NYX V.01).

Asking an ordinary model "what are you good at?" gets you a sentence someone wrote for it. This
module answers the same question from **numbers she measured about herself**: how big her graph
has grown, how much she remembers, which of her faculties her own track record says to trust,
which words she uses without understanding, and which of her engines are actually installed.

Nothing here is a stored description. Every field is read off live state, so the answer changes
as she changes — and when a faculty is missing or a record is too thin to lean on, that is what
it says, rather than filling the gap with something agreeable.

Honest framing: this is **self-measurement, not self-awareness**. It is introspection in the
engineering sense — a mind that can be asked what it is made of and answer accurately. No claim
is made that there is something it is like to be her.

Composes :mod:`nyxara.nyx.metacog` (measured reliability), :mod:`nyxara.nyx.graph` (what she has
connected), :mod:`nyxara.nyx.ground` (what she actually understands), and, when present,
``memory/self_model.py`` and ``identity/soul.py``. Fail-soft throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["SelfReport", "NyxSelfModel"]


@dataclass
class SelfReport:
    """A measured account of herself. Every number comes from live state."""

    concepts: int = 0
    associations: int = 0
    memories: int = 0
    turns: int = 0
    understood_words: int = 0
    learned_words: List[str] = field(default_factory=list)
    trusted: Dict[str, float] = field(default_factory=dict)     # faculty → measured reliability
    untested: List[str] = field(default_factory=list)           # too few samples to lean on
    weakest: List[str] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)        # engines that are not installed
    gaps: List[str] = field(default_factory=list)               # met often, still unconnected
    voice: str = ""                                             # which language rung is live
    fluent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"concepts": self.concepts, "associations": self.associations,
                "memories": self.memories, "turns": self.turns,
                "understood_words": self.understood_words,
                "learned_words": self.learned_words,
                "trusted": {k: round(v, 4) for k, v in self.trusted.items()},
                "untested": self.untested, "weakest": self.weakest,
                "unavailable": self.unavailable, "gaps": self.gaps,
                "voice": self.voice, "fluent": self.fluent}

    def describe(self) -> str:
        """Say it in sentences — assembled from the measurements, never from a template.

        Every clause here is conditional on a real number, so a mind that has learned nothing
        says so, and one that has learned a great deal says that instead.
        """
        parts: List[str] = []
        if self.concepts:
            parts.append(f"I have {self.concepts} concepts connected by {self.associations} "
                         f"associations, and {self.memories} things I can recall")
        else:
            parts.append("I have not learned anything yet")
        if self.turns:
            parts.append(f"across {self.turns} turns of thinking")
        if self.trusted:
            best = max(self.trusted, key=self.trusted.get)
            parts.append(f"the faculty I have most reason to trust is {best} "
                         f"({self.trusted[best]:.0%} of its answers held up)")
        if self.untested:
            parts.append(f"I have too little evidence to judge {', '.join(self.untested)}")
        if self.weakest:
            parts.append(f"I trust {', '.join(self.weakest)} least")
        if self.gaps:
            parts.append(f"and these keep coming up without connecting to anything: "
                         f"{', '.join(self.gaps)}")
        if self.unavailable:
            parts.append(f"these engines are not installed: {', '.join(self.unavailable)}")
        if not self.fluent and self.voice:
            parts.append(f"my voice is running on '{self.voice}', which cannot hold a "
                         f"conversation")
        return "; ".join(parts) + "."


class NyxSelfModel:
    """Answers "who am I / what can I do / what am I bad at" from measurement."""

    def __init__(self, brain: Any, *, inner: Any = None) -> None:
        self.brain = brain
        self._inner = inner                     # memory/self_model.SelfModel, when present

    def report(self) -> SelfReport:
        """Read her own state and say what is actually there."""
        out = SelfReport()
        brain = self.brain
        if brain is None:
            return out
        try:
            self._read_graph(out, brain)
            self._read_memory(out, brain)
            self._read_metacog(out, brain)
            self._read_ground(out, brain)
            self._read_faculties(out, brain)
            self._read_voice(out, brain)
            return out
        except Exception:  # noqa: BLE001 — an unreadable self is an empty report, never a crash
            return out

    @staticmethod
    def _read_graph(out: SelfReport, brain: Any) -> None:
        try:
            stats = brain.graph.stats()
            out.concepts, out.associations = stats.nodes, stats.edges
            out.gaps = brain.gaps(k=4)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _read_memory(out: SelfReport, brain: Any) -> None:
        try:
            out.memories = len(brain.memory)
            out.turns = int(getattr(brain, "turns", 0))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _read_metacog(out: SelfReport, brain: Any) -> None:
        metacog = getattr(brain, "metacog", None)
        if metacog is None:
            return
        try:
            for source, record in metacog.reliability.items():
                if record.trusted:
                    out.trusted[source] = float(record.score)
                else:
                    # Too few samples is not evidence of competence *or* of failure — it is
                    # its own answer, and saying so beats reporting a number nobody earned.
                    out.untested.append(source)
            out.weakest = metacog.weakest(k=2)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _read_ground(out: SelfReport, brain: Any) -> None:
        ground = getattr(brain, "ground", None)
        if ground is None:
            return
        try:
            out.learned_words = list(getattr(ground, "learned", []))[-5:]
            out.understood_words = sum(1 for label in brain.graph.nodes
                                       if ground.understands(label))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _read_faculties(out: SelfReport, brain: Any) -> None:
        workspace = getattr(brain, "workspace", None)
        if workspace is None:
            out.unavailable.append("workspace")
            return
        try:
            out.unavailable.extend(s.name for s in workspace.specialists if not s.available())
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _read_voice(out: SelfReport, brain: Any) -> None:
        dialogue = getattr(brain, "dialogue", None)
        if dialogue is None:
            return
        try:
            surface = dialogue.surface()
            out.voice, out.fluent = surface.provider, surface.fluent
        except Exception:  # noqa: BLE001
            pass

    # ---- questions the Master actually asks ------------------------------- #
    def can(self, what: str) -> Optional[bool]:
        """Can she do this? ``None`` when she genuinely cannot tell — which is common."""
        try:
            workspace = getattr(self.brain, "workspace", None)
            if workspace is None:
                return None
            for spec in workspace.specialists:
                if spec.name == str(what).strip().lower():
                    return bool(spec.available())
            return None
        except Exception:  # noqa: BLE001
            return None

    def weakest(self, *, k: int = 3) -> List[str]:
        """What she is worst at, by her own measurement — not by modesty."""
        try:
            metacog = getattr(self.brain, "metacog", None)
            return metacog.weakest(k=k) if metacog is not None else []
        except Exception:  # noqa: BLE001
            return []

    def describe(self) -> str:
        return self.report().describe()
