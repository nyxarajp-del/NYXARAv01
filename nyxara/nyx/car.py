"""NYXARA · nyx/car.py — continuous autonomous reasoning (♾, NYX V.01 pillar 2).

A mind that only thinks when spoken to is not thinking, it is responding. This module is what
NYX does when nobody is talking to her: it picks a question **out of her own state**, thinks
about it, and folds whatever it learns back in — on the clock, forever, without a prompt.

The question is never drawn from a list. It comes from where her knowledge is actually thin,
which she can measure three ways:

* **structural gaps** — concepts she keeps meeting but has never connected to anything
  (:meth:`~nyxara.nyx.graph.DynamicNeuralGraph.gaps`);
* **ungrounded words** — symbols she has been using without any referent behind them
  (:mod:`nyxara.nyx.ground`);
* **her own weakest faculties** — the specialists her measured track record says to trust least
  (:mod:`nyxara.nyx.metacog`).

That ordering is the Free Energy Principle in practical clothes: go where uncertainty is
highest, because that is where thinking buys the most. Each pick becomes one bounded cycle
through the same machinery a real turn uses, so what she learns alone is learned the same way
as what she learns from the Master.

Honest framing — the claim here is easy to inflate:

* This runs **only while the process is running**. "24/7" is true of ``nyxara-daemon``, not of
  a laptop that is asleep. Beats are counted, so the claim is checkable rather than rhetorical.
* Every step is **budgeted** (``budget_ms``) and rides the existing heartbeat/autonomic clock —
  there is no second thread competing for the mind.
* It **stops when oversight says stop**. A scrammed or paused NYXARA does not quietly keep
  thinking in the background; :meth:`step` becomes a no-op and says so.
* It reasons over *her own* knowledge and simulators. It is not reading the world, and this
  module makes no discovery claims beyond "she connected two things she already had".

Pure standard library. Fail-soft: a background pass must never raise into the idle loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["Wondering", "CarStep", "ContinuousAutonomousReasoning"]


@dataclass
class Wondering:
    """One question she asked herself, and where it came from."""

    text: str
    subject: str = ""
    origin: str = "gap"              # gap | ungrounded | weakness
    uncertainty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "subject": self.subject, "origin": self.origin,
                "uncertainty": round(self.uncertainty, 4)}


@dataclass
class CarStep:
    """One background thought: what she wondered, what came of it, what it cost."""

    ran: bool = False
    halted: bool = False             # oversight said stop — a real and reportable outcome
    reason: str = ""
    wondering: Optional[Wondering] = None
    answer: str = ""
    verified: bool = False
    learned: bool = False            # did anything actually change in her state?
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"ran": self.ran, "halted": self.halted, "reason": self.reason,
                "wondering": self.wondering.to_dict() if self.wondering else None,
                "answer": self.answer[:200], "verified": self.verified,
                "learned": self.learned, "elapsed_ms": round(self.elapsed_ms, 2)}


class ContinuousAutonomousReasoning:
    """Thinks between prompts: picks its own question from where its knowledge is thinnest."""

    def __init__(self, brain: Any, *, budget_ms: float = 250.0, interval_s: float = 30.0,
                 max_questions: int = 3, oversight: Any = None) -> None:
        self.brain = brain
        self.budget_ms = max(1.0, float(budget_ms))
        # A *time* cadence, not a beat count. Several clocks drive this — the heartbeat thread
        # and ``idle_maintenance``, and the heartbeat itself calls idle_maintenance — so counting
        # ticks would mean "every N ticks from however many callers happen to exist". Wall-clock
        # is what the claim is actually about: she thinks on her own every ``interval_s``.
        self.interval_s = max(0.0, float(interval_s))
        self.max_questions = max(1, int(max_questions))
        self.oversight = oversight

        self.beats = 0
        self.steps = 0
        self.learned_total = 0
        self.recent: List[CarStep] = []
        self._last_step_at = 0.0
        # What she has recently wondered about. Without this she picks the same first gap
        # forever and never reaches the second one — busy, but not curious.
        self._asked: List[str] = []

    # ---- the clock -------------------------------------------------------- #
    def beat(self) -> Optional[CarStep]:
        """One tick from whichever clock is driving. Thinks at most once per ``interval_s``."""
        try:
            self.beats += 1
            now = time.monotonic()
            if self._last_step_at and (now - self._last_step_at) < self.interval_s:
                return None
            self._last_step_at = now
            return self.step()
        except Exception:  # noqa: BLE001 — a background beat never raises into the loop
            return None

    def step(self) -> CarStep:
        """One bounded, self-directed thought."""
        out = CarStep()
        started = time.monotonic()
        try:
            if not self._permitted():
                out.halted, out.reason = True, "oversight paused or scrammed"
                return out

            wondering = self.wonder()
            if wondering is None:
                out.reason = "nothing uncertain enough to be worth a thought"
                return out
            out.wondering = wondering

            thought = self.brain.think(wondering.text)
            out.ran = True
            out.answer = str(getattr(thought, "answer", "") or "")
            out.verified = bool(getattr(thought, "verified", False))
            out.learned = self._fold_in(wondering, thought)
            if out.learned:
                self.learned_total += 1
            self.steps += 1
            return out
        except Exception:  # noqa: BLE001 — never raise into the idle loop
            out.reason = "the step failed and was abandoned"
            return out
        finally:
            out.elapsed_ms = (time.monotonic() - started) * 1000.0
            self.recent.append(out)
            del self.recent[:-32]

    # ---- what to wonder about --------------------------------------------- #
    def wonder(self) -> Optional[Wondering]:
        """Pick the question worth the most: go where her own uncertainty is highest.

        Ordered by how *actionable* the uncertainty is — an ungrounded word can be settled by
        grounding it, a structural gap by connecting it, a weak faculty only by exercising it.
        """
        try:
            for pick in (self._ungrounded, self._gap, self._weakness):
                got = pick()
                if got is not None:
                    self._note_asked(got.subject or got.text)
                    return got
            # Everything obvious has been asked recently — let the oldest become fair game
            # again rather than reporting nothing to wonder about forever.
            if self._asked:
                self._asked = self._asked[len(self._asked) // 2:]
            return None
        except Exception:  # noqa: BLE001
            return None

    def _note_asked(self, subject: str) -> None:
        self._asked.append(str(subject))
        del self._asked[:-32]

    def _fresh(self, subject: str) -> bool:
        return str(subject) not in self._asked

    def _ungrounded(self) -> Optional[Wondering]:
        """A word she has been using with nothing behind it is the sharpest kind of gap."""
        ground = getattr(self.brain, "ground", None)
        graph = getattr(self.brain, "graph", None)
        if ground is None or graph is None:
            return None
        for label in list(graph.nodes)[-64:]:
            if self._fresh(label) and not ground.understands(label):
                return Wondering(text=f"what is {label}", subject=label,
                                 origin="ungrounded", uncertainty=1.0)
        return None

    def _gap(self) -> Optional[Wondering]:
        """A concept she keeps meeting and has never connected to anything."""
        graph = getattr(self.brain, "graph", None)
        if graph is None:
            return None
        for label in graph.gaps(k=self.max_questions * 4):
            if not self._fresh(label):
                continue
            neighbours = graph.neighbours(label, k=1)
            if not neighbours:
                return Wondering(text=f"what relates to {label}", subject=label,
                                 origin="gap", uncertainty=0.8)
        return None

    def _weakness(self) -> Optional[Wondering]:
        """The faculty her own record says to trust least — exercise it and find out."""
        metacog = getattr(self.brain, "metacog", None)
        if metacog is None:
            return None
        weakest = metacog.weakest(k=1)
        if not weakest:
            return None
        graph = getattr(self.brain, "graph", None)
        subject = (graph.gaps(k=1) or [""])[0] if graph is not None else ""
        if not subject:
            return None
        return Wondering(text=f"what do I actually know about {subject}", subject=subject,
                         origin="weakness", uncertainty=0.5)

    # ---- folding what she learned back in ---------------------------------- #
    def _fold_in(self, wondering: Wondering, thought: Any) -> bool:
        """Did this thought change anything? Ground a word, connect a gap — or nothing."""
        learned = False
        try:
            answer = str(getattr(thought, "answer", "") or "").strip()
            subject = wondering.subject
            ground = getattr(self.brain, "ground", None)

            # An answer that actually defines the word she was missing grounds it for good.
            if ground is not None and subject and answer and not ground.understands(subject):
                if ground.learn(subject, answer, require_definition=True):
                    learned = True

            # A conclusion worth keeping is kept; an empty one is not dressed up as progress.
            if answer and getattr(thought, "verified", False):
                self.brain.remember(f"wondered-{self.steps}", answer, kind="conclusion")
                learned = True
            return learned
        except Exception:  # noqa: BLE001
            return learned

    # ---- oversight --------------------------------------------------------- #
    def _permitted(self) -> bool:
        """Autonomy buys no extra power: a paused or scrammed mind takes no background work."""
        gate = self.oversight
        if gate is None:
            return True
        try:
            check = getattr(gate, "gate", None)
            if callable(check):
                return bool(check())
            return bool(check) if check is not None else True
        except Exception:  # noqa: BLE001 — if oversight cannot be read, do not act
            return False

    # ---- introspection ------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            ran = [s for s in self.recent if s.ran]
            return {"beats": self.beats, "steps": self.steps,
                    "learned_total": self.learned_total,
                    "interval_s": self.interval_s, "budget_ms": self.budget_ms,
                    "recent_ran": len(ran),
                    "mean_ms": round(sum(s.elapsed_ms for s in ran) / len(ran), 2) if ran else 0.0,
                    "last": self.recent[-1].to_dict() if self.recent else None}
        except Exception:  # noqa: BLE001
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {"beats": self.beats, "steps": self.steps, "learned_total": self.learned_total}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.beats = int(data.get("beats", 0))
            self.steps = int(data.get("steps", 0))
            self.learned_total = int(data.get("learned_total", 0))
            return True
        except Exception:  # noqa: BLE001
            return False
