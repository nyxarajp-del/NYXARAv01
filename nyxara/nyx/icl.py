"""NYXARA · nyx/icl.py — learning a new procedure inside one turn (📐, NYX V.02, gap 6).

NYX V.01's "learning" was two things: a Hebbian weight going up, and a trace being written to
memory. Both are real, and neither is a **new procedure**. Shown

    apple -> APPLE!
    mango -> MANGO!
    car   -> ?

V.01 could remember that it had been shown them. It could not answer ``CAR!``.

:class:`InContextLearner` closes that. It reads demonstrations out of a turn in whatever shape
they arrived, induces a program from them with
:class:`~nyxara.cognition.skill_induction.SkillInductionEngine`, applies it to the probe, and
writes the skill into :mod:`nyxara.nyx.holomem` so it survives the turn, the session and the
process. No weights are updated anywhere; the learning is the program.

Two measured bugs in the machinery underneath are fixed as part of this, because a learner that
sits on top of them inherits them:

* ``solve("7")`` returned ``None`` after a ×2 skill had been learned from ``3 → 6``, because the
  anchorless match floor asked whether ``"7"`` was *lexically* close to ``"3"``. That floor
  exists to stop a skill hijacking an unrelated turn — it should not apply to a probe the caller
  has already parsed out of a demonstration block. ``solve(..., probe=True)`` now says so.
* Field permutation was not in the operation set at all, so ``john smith → Smith, John`` induced
  as ``None``. :func:`~nyxara.cognition.skill_induction._induce_reorder` adds it.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* This is **program induction over a finite operation set** — case, affix, replace, arithmetic,
  field permutation, slot templates, and compositions of them up to a bounded depth. Inside
  that set, two or three examples genuinely teach her something new. Outside it she **declines**
  and says the demonstrations did not reduce to a program she can express. She does not guess.
* Induction is accepted only when the program reproduces **every** demonstration exactly, and
  with three or more demos one is held out to measure real transfer rather than fit.
* Nothing here is a weight update. Calling it "learning" is accurate in the in-context sense and
  would be misleading in the gradient sense, so the distinction is reported in
  :meth:`InContextLearner.stats`.

Pure standard library. Every public method is fail-soft: on any error it degrades to a null
result rather than breaking a turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Demonstration", "Learned", "InContextLearner"]

#: How a demonstration block may be written. All of these are common in real prompts, and V.01
#: understood none of them as demonstrations.
_ARROW = re.compile(r"^\s*(?P<a>.+?)\s*(?:->|→|=>|⇒|:|\||\t)\s*(?P<b>.+?)\s*$")
#: The probe: the row whose answer is missing.
_PROBE_MARKS = ("?", "??", "___", "…", "...")


@dataclass
class Demonstration:
    """One ``input → output`` row lifted out of a turn."""

    given: str = ""
    wanted: str = ""
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"given": self.given, "wanted": self.wanted, "line": self.line}


@dataclass
class Learned:
    """What one turn of in-context learning actually produced."""

    demos: List[Demonstration] = field(default_factory=list)
    probe: str = ""
    skill: str = ""                  # the induced skill's name, "" when nothing was induced
    program: str = ""                # human-readable rendering of the induced program
    answer: str = ""                 # what the program says about the probe
    confidence: float = 0.0
    generalized: bool = False        # True iff it also predicted a held-out demonstration
    learned: bool = False
    refusal: str = ""                # why not, in plain words, when ``learned`` is False

    def to_dict(self) -> Dict[str, Any]:
        return {"demos": [d.to_dict() for d in self.demos], "probe": self.probe,
                "skill": self.skill, "program": self.program, "answer": self.answer,
                "confidence": round(self.confidence, 4), "generalized": self.generalized,
                "learned": self.learned, "refusal": self.refusal}


class InContextLearner:
    """Read demonstrations, induce a procedure, answer the probe, and keep the procedure."""

    #: memory kind under which learned skills are written, so they outlive the process
    TRACE_KIND = "skill"

    def __init__(self, brain: Any = None, *, min_demos: int = 2, max_demos: int = 24,
                 remember: bool = True, engine: Any = None) -> None:
        self.brain = brain
        self.min_demos = max(2, int(min_demos))
        self.max_demos = max(self.min_demos, int(max_demos))
        self.remember = bool(remember)
        self._engine = engine
        self._tried = engine is not None

        self.turns_read = 0
        self.blocks_seen = 0
        self.skills_induced = 0
        self.refusals = 0
        self._skill_seq = 0

    # ---- the engine underneath ------------------------------------------- #
    def engine(self) -> Any:
        """The induction engine, built lazily. ``None`` means the faculty is simply absent."""
        if not self._tried:
            self._tried = True
            try:
                from nyxara.cognition.skill_induction import SkillInductionEngine
                self._engine = SkillInductionEngine(min_demos=self.min_demos)
            except Exception:  # noqa: BLE001 — without it she reads demos and induces nothing
                self._engine = None
        return self._engine

    # ---- reading demonstrations out of a turn ----------------------------- #
    def demonstrations(self, text: str) -> Tuple[List[Demonstration], str]:
        """Lift ``(demos, probe)`` out of ``text``. Empty when the turn is ordinary prose.

        Accepts every shape a demonstration block realistically arrives in: ``a -> b``,
        ``a → b``, ``a: b``, ``a | b``, a tab-separated table, and a final row whose answer is
        ``?``. The probe may also be a trailing bare line after a block of pairs.
        """
        demos: List[Demonstration] = []
        probe = ""
        try:
            lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
            if len(lines) < self.min_demos:
                return [], ""
            trailing: List[str] = []
            for i, line in enumerate(lines):
                match = _ARROW.match(line)
                if match is None:
                    trailing.append(line.strip())
                    continue
                a, b = match.group("a").strip(), match.group("b").strip()
                if not a or not b:
                    continue
                if b in _PROBE_MARKS:
                    probe = a                   # this row *is* the question
                    continue
                demos.append(Demonstration(given=a, wanted=b, line=i))
                if len(demos) >= self.max_demos:
                    break
            if not probe and trailing and demos:
                # A block of pairs followed by a bare line: the bare line is the probe.
                probe = trailing[-1]
            if len(demos) < self.min_demos:
                return [], ""
            return demos, probe
        except Exception:  # noqa: BLE001 — reading a turn never breaks it
            return [], ""

    # ---- the whole act ---------------------------------------------------- #
    def learn(self, text: str, *, name: str = "") -> Learned:
        """Read, induce, apply, remember. The one call :class:`~nyxara.nyx.brain.NyxBrain` makes."""
        out = Learned()
        try:
            self.turns_read += 1
            demos, probe = self.demonstrations(text)
            out.demos, out.probe = demos, probe
            if not demos:
                return out
            self.blocks_seen += 1

            engine = self.engine()
            if engine is None:
                out.refusal = ("I can see the demonstrations, but the induction engine is not "
                               "available in this build, so I cannot turn them into a rule.")
                self.refusals += 1
                return out

            self._skill_seq += 1
            skill_name = str(name).strip() or f"icl-{self._skill_seq}"
            pairs = [(d.given, d.wanted) for d in demos]
            skill = engine.induce(skill_name, pairs)
            if skill is None:
                # This is the honest branch and it matters: the operation set is finite, and a
                # demonstration outside it gets a refusal rather than a plausible-looking guess.
                out.refusal = (
                    f"I read {len(demos)} demonstrations and could not reduce them to a rule I "
                    f"can express. My operation set is case, affix, replace, arithmetic, field "
                    f"reordering and slot templates, composed up to a bounded depth — this is "
                    f"outside it. I would rather say that than invent a pattern.")
                self.refusals += 1
                return out

            self.skills_induced += 1
            out.skill = skill.name
            out.program = skill.render()
            out.confidence = float(skill.confidence)
            out.generalized = bool(skill.generalized)
            out.learned = True

            if probe:
                got = engine.apply_skill(skill.name, probe)
                if got is None:
                    # The named skill did not change the probe; let any skill claim it, with
                    # the probe flag on because this string *was* parsed as the question.
                    got = engine.solve(probe, probe=True)
                if got is not None:
                    out.answer, out.confidence = str(got[0]), float(got[1])

            self._remember(skill, out)
            return out
        except Exception:  # noqa: BLE001 — in-context learning never breaks a turn
            return out

    def apply(self, name: str, text: str) -> Optional[Tuple[str, float]]:
        """Run a skill she already holds against a new input."""
        try:
            engine = self.engine()
            return engine.apply_skill(name, text) if engine is not None else None
        except Exception:  # noqa: BLE001
            return None

    def solve(self, text: str, *, probe: bool = False) -> Optional[Tuple[str, float]]:
        """Let any learned skill claim this turn, if one genuinely fits it."""
        try:
            engine = self.engine()
            return engine.solve(text, probe=probe) if engine is not None else None
        except Exception:  # noqa: BLE001
            return None

    # ---- keeping the procedure ------------------------------------------- #
    def _remember(self, skill: Any, learned: Learned) -> None:
        """Write the skill into holographic memory so it outlives the turn *and* the process.

        The induction engine has its own store; this is the second copy, in her own memory, so
        ``/nyx skills`` after a restart is answering from something she actually kept.
        """
        try:
            if not self.remember or self.brain is None:
                return
            memory = getattr(self.brain, "memory", None)
            if memory is None:
                return
            demos = "; ".join(f"{d.given} -> {d.wanted}" for d in learned.demos[:5])
            memory.remember(f"skill-{skill.name}",
                            f"learned procedure {skill.name}: {skill.render()} "
                            f"(from {demos})", kind=self.TRACE_KIND)
        except Exception:  # noqa: BLE001 — remembering is best-effort, never required
            return

    def skills(self) -> List[Any]:
        try:
            engine = self.engine()
            return list(engine.skills()) if engine is not None else []
        except Exception:  # noqa: BLE001
            return []

    # ---- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            held = self.skills()
            return {
                "turns_read": self.turns_read,
                "demonstration_blocks": self.blocks_seen,
                "skills_induced": self.skills_induced,
                "skills_held": len(held),
                "refusals": self.refusals,
                "generalized": sum(1 for s in held if getattr(s, "generalized", False)),
                "available": self.engine() is not None,
                # Said plainly because the word "learning" is doing two jobs in this repo.
                "note": ("in-context program induction over a finite operation set — no weight "
                         "is updated anywhere; what is learned is a program, and a "
                         "demonstration outside the set is refused, not guessed"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx skills`` prints."""
        try:
            held = self.skills()
            if not held:
                return ("I hold no learned procedures yet. Show me two or three examples "
                        "(`apple -> APPLE!`) and I will try to induce the rule.")
            lines = [f"{len(held)} learned procedure(s):"]
            for skill in held:
                mark = " (predicted a held-out example)" if getattr(
                    skill, "generalized", False) else ""
                lines.append(f"  {skill.render()}   conf "
                             f"{float(getattr(skill, 'confidence', 0.0)):.2f}{mark}")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                "turns_read": self.turns_read, "blocks_seen": self.blocks_seen,
                "skills_induced": self.skills_induced, "refusals": self.refusals,
                "skill_seq": self._skill_seq,
                "skills": [s.to_dict() for s in self.skills()],
            }
        except Exception:  # noqa: BLE001
            return {}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.turns_read = int(data.get("turns_read", 0))
            self.blocks_seen = int(data.get("blocks_seen", 0))
            self.skills_induced = int(data.get("skills_induced", 0))
            self.refusals = int(data.get("refusals", 0))
            self._skill_seq = int(data.get("skill_seq", 0))
            engine = self.engine()
            if engine is None:
                return True
            from nyxara.cognition.skill_induction import InductiveSkill
            for raw in data.get("skills", []) or []:
                if not isinstance(raw, dict):
                    continue
                skill = InductiveSkill.from_dict(raw)
                if skill.name and skill.program:
                    engine._skills[skill.name] = skill
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
