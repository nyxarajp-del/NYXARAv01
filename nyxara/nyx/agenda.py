"""NYXARA · nyx/agenda.py — goals of her own, that outlive the session (🎯, NYX V.02, L-ABSOLUTE-AGENCY).

V.01's :mod:`~nyxara.nyx.car` thinks one self-directed thought per beat and then forgets it. It
picks a gap, reasons about it, writes the result down — and the next beat starts over. There is
no *stack*: nothing she began yesterday is still being pursued today, and nothing accumulates.

:class:`SovereignAgenda` is that stack:

* **Goals she wrote herself.** Sourced from her own **measured** gaps
  (:meth:`nyxara.nyx.graph.DynamicNeuralGraph.gaps`), from ungrounded words, from the
  specialist meta-cognition says is weakest, and from :mod:`nyxara.identity.motivation`'s
  intrinsic drives — empowerment, information gain, competence, novelty. Not from a list.
* **Multi-session pursuit.** A goal, its sub-goals, its progress and its findings persist to
  the ``nyx.json`` sidecar. Restart, and the same goal is still open at the same progress —
  which is the actual test of this layer and is what V.01 could not do at all.
* **Real work per beat.** Advancing a goal spends a budgeted step in
  :mod:`~nyxara.nyx.reason`, :mod:`~nyxara.nyx.episteme`, :mod:`~nyxara.nyx.axiom` or
  :mod:`~nyxara.nyx.hands`, and records what came back.
* **A briefing.** What she chose, why, what she found, and where she is stuck — in plain words,
  proactively.
* **Her own refusal.** :mod:`~nyxara.nyx.will` may decline a goal on her values, and that
  refusal is reported rather than silently skipped.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **Goals come from her measured gaps and her drives, not from nowhere.** She does not decide
  to "understand the universe"; she decides that ``flywheel`` is a word she keeps meeting and
  never connects to anything, and goes after that. That is a smaller claim and a true one.
* **Reach is her simulators and her knowledge**, not omniscience. A goal she cannot advance is
  reported as *stuck*, with what she tried.
* **The Master is sovereign over the agenda.** Any goal can be vetoed or redirected, and
  ``/scram`` stops all pursuit. Her being able to decline a goal on her values does not run the
  other way: it is a refusal, never an override.
* Progress is **measured**, not asserted: it moves when a step actually returned something, and
  a goal that stops moving is marked stuck rather than left looking busy.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Goal", "Progress", "Briefing", "SovereignAgenda", "GOAL_KINDS"]

#: What kind of thing a goal is, which decides how a beat tries to advance it.
GOAL_KINDS = ("ground", "connect", "understand", "formalise", "verify", "build")

_OPEN, _STUCK, _DONE, _VETOED, _DECLINED = "open", "stuck", "done", "vetoed", "declined"


@dataclass
class Progress:
    """One beat of work on one goal — what she did, and what came back."""

    at: float = field(default_factory=time.time)
    action: str = ""
    gained: float = 0.0          # how much this beat actually moved the goal
    finding: str = ""
    ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"at": round(self.at, 3), "action": self.action,
                "gained": round(self.gained, 4), "finding": self.finding, "ok": self.ok}


@dataclass
class Goal:
    """Something she decided to pursue, and how far she has got with it."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    subject: str = ""
    kind: str = "understand"
    why: str = ""                 # in her words, from the measurement that produced it
    priority: float = 0.5
    status: str = _OPEN
    progress: float = 0.0         # [0,1], measured
    beats: int = 0
    stalled: int = 0              # consecutive beats that gained nothing
    parent: str = ""
    findings: List[Progress] = field(default_factory=list)
    born: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    note: str = ""

    @property
    def open(self) -> bool:
        return self.status == _OPEN

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.born)

    def render(self) -> str:
        bar = "█" * int(self.progress * 10) + "·" * (10 - int(self.progress * 10))
        tail = f" — {self.note}" if self.note else ""
        return (f"[{bar}] {self.status:8} {self.kind:11} {self.subject:24} "
                f"({self.beats} beats){tail}")

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "subject": self.subject, "kind": self.kind, "why": self.why,
                "priority": round(self.priority, 4), "status": self.status,
                "progress": round(self.progress, 4), "beats": self.beats,
                "stalled": self.stalled, "parent": self.parent, "note": self.note,
                "born": round(self.born, 3), "touched": round(self.touched, 3),
                "age_s": round(self.age_s, 1),
                "findings": [f.to_dict() for f in self.findings[-8:]]}


@dataclass
class Briefing:
    """What she would tell the Master, unprompted."""

    chose: List[str] = field(default_factory=list)
    found: List[str] = field(default_factory=list)
    stuck: List[str] = field(default_factory=list)
    declined: List[str] = field(default_factory=list)
    since_s: float = 0.0
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"chose": self.chose, "found": self.found, "stuck": self.stuck,
                "declined": self.declined, "since_s": round(self.since_s, 1),
                "text": self.text}


class SovereignAgenda:
    """A goal stack she wrote herself, pursued across sessions, and answerable to the Master."""

    def __init__(self, brain: Any = None, *, max_goals: int = 24, max_open: int = 6,
                 stall_after: int = 3, every_s: float = 30.0,
                 budget_ms: float = 400.0) -> None:
        self.brain = brain
        self.max_goals = max(1, int(max_goals))
        self.max_open = max(1, int(max_open))
        self.stall_after = max(1, int(stall_after))
        self.every_s = max(0.0, float(every_s))
        self.budget_ms = max(10.0, float(budget_ms))

        self.goals: Dict[str, Goal] = {}
        self.beats = 0
        self.formed = 0
        self.completed = 0
        self.vetoed = 0
        self.declined = 0
        self._last_beat = 0.0
        self._last_briefed = time.time()

    # ---- forming goals of her own -------------------------------------------- #
    def form(self, *, limit: int = 3) -> List[Goal]:
        """Read her own measurements and decide what is worth going after.

        Every goal here traces to something she actually measured — a concept she keeps meeting
        without connecting, a word she has no referent for, a faculty her own track record says
        is weakest. None of them is "understand the universe".
        """
        made: List[Goal] = []
        try:
            for subject, kind, why, priority in self._candidates():
                if len(made) >= max(0, int(limit)):
                    break
                if self._already(subject, kind):
                    continue
                declined = self._would_decline(subject, kind)
                status = _DECLINED if declined else _OPEN
                goal = Goal(subject=subject, kind=kind, why=why, priority=priority,
                            status=status, note=declined)
                self._add(goal)
                made.append(goal)
                if declined:
                    self.declined += 1
                else:
                    self.formed += 1
            return made
        except Exception:  # noqa: BLE001 — forming a goal never breaks a beat
            return made

    def _candidates(self) -> List[Tuple[str, str, str, float]]:
        """Everything worth pursuing, best first — each with the measurement behind it."""
        out: List[Tuple[str, str, str, float]] = []
        brain = self.brain

        # 1. Concepts she keeps meeting and never connects. Her graph measures this directly.
        try:
            graph = getattr(brain, "graph", None)
            if graph is not None:
                for label in (graph.gaps(k=6) or []):
                    out.append((label, "connect",
                                "a concept I keep meeting without connecting it to anything",
                                0.7))
        except Exception:  # noqa: BLE001
            pass

        # 2. Words she has no referent for at all.
        try:
            ground = getattr(brain, "ground", None)
            stats = ground.stats() if ground is not None else {}
            for word in list(stats.get("ungrounded", []) or [])[:4]:
                out.append((str(word), "ground",
                            "a word I use with nothing behind it", 0.65))
        except Exception:  # noqa: BLE001
            pass

        # 3. The faculty her own measured track record says is weakest.
        try:
            metacog = getattr(brain, "metacog", None)
            weakest = (metacog.stats() or {}).get("weakest") if metacog is not None else None
            if weakest:
                out.append((str(weakest), "verify",
                            "the faculty my own record says I am worst at", 0.6))
        except Exception:  # noqa: BLE001
            pass

        # 4. Whatever her intrinsic drives rate highest among the above. Motivation does not
        #    invent a subject; it reorders the ones her measurements produced.
        return self._motivated(out)

    def _motivated(self, candidates: List[Tuple[str, str, str, float]]
                   ) -> List[Tuple[str, str, str, float]]:
        try:
            from nyxara.identity.motivation import MotivationSystem, Option
        except Exception:  # noqa: BLE001 — without drives, priority stays as measured
            return sorted(candidates, key=lambda c: -c[3])
        try:
            system = MotivationSystem()
            scored: List[Tuple[float, Tuple[str, str, str, float]]] = []
            for subject, kind, why, priority in candidates:
                impulse = system.evaluate(Option(
                    name=f"{kind}:{subject}", signature=f"{kind}:{subject}",
                    info_gain=priority, competence_gain=0.5 * priority,
                    predicted_options=2))
                scored.append((float(getattr(impulse, "total", priority)),
                               (subject, kind, why, priority)))
            scored.sort(key=lambda p: -p[0])
            return [c for _s, c in scored]
        except Exception:  # noqa: BLE001
            return sorted(candidates, key=lambda c: -c[3])

    def _already(self, subject: str, kind: str) -> bool:
        """Has she already taken this subject on? Any outcome counts, including *stuck*.

        Re-forming a goal she has already got stuck on is churn dressed as persistence — and
        because a goal escalates its *kind* as it goes, the original kind is checked too.
        """
        return any(g.subject == subject for g in self.goals.values())

    def _would_decline(self, subject: str, kind: str) -> str:
        """Her values get a say. A refusal is reported, never a silent skip."""
        try:
            will = getattr(self.brain, "will", None)
            if will is None:
                return ""
            option = type("Opt", (), {"source": f"{kind}:{subject}",
                                      "content": f"{kind} {subject}"})()
            choice = will.choose([option])
            return str(getattr(choice, "reason", "") or "") \
                if getattr(choice, "declined", False) else ""
        except Exception:  # noqa: BLE001 — an unreadable will never manufactures a refusal
            return ""

    def _add(self, goal: Goal) -> None:
        self.goals[goal.id] = goal
        if len(self.goals) > self.max_goals:
            # Drop the oldest *settled* goal first; an open one is never evicted for age.
            settled = [g for g in self.goals.values() if not g.open]
            victim = min(settled or list(self.goals.values()), key=lambda g: g.touched)
            self.goals.pop(victim.id, None)

    # ---- pursuing them across sessions ---------------------------------------- #
    def due(self, *, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        return self.every_s <= 0.0 or (now - self._last_beat) >= self.every_s

    def open_goals(self) -> List[Goal]:
        return sorted([g for g in self.goals.values() if g.open],
                      key=lambda g: (-g.priority, g.touched))[: self.max_open]

    def beat(self, *, oversight: Any = None, force: bool = False) -> Optional[Progress]:
        """One budgeted step on the most worthwhile open goal. ``None`` when she does not act."""
        try:
            if self._stopped(oversight):
                return None
            if not force and not self.due():
                return None
            self._last_beat = time.time()

            if not self.open_goals():
                self.form()
            goals = self.open_goals()
            if not goals:
                return None

            goal = goals[0]
            self.beats += 1
            progress = self._advance(goal)
            goal.beats += 1
            goal.touched = time.time()
            goal.findings.append(progress)
            del goal.findings[:-16]

            if progress.gained > 0.0:
                goal.progress = min(1.0, goal.progress + progress.gained)
                goal.stalled = 0
            else:
                goal.stalled += 1

            if goal.progress >= 1.0:
                goal.status, goal.note = _DONE, progress.finding[:120]
                self.completed += 1
            elif goal.stalled >= self.stall_after and self._escalate(goal):
                # Repeating an approach that is not working is not pursuit. A concept she
                # cannot *connect* needs new information, not more rewiring — so the goal
                # changes strategy once before it is allowed to be called stuck.
                goal.stalled = 0
            elif goal.stalled >= self.stall_after:
                # Stuck is a real, reportable outcome. Leaving it "open" would make an agenda
                # that never moves look like one that is working.
                goal.status = _STUCK
                goal.note = (f"stuck after {goal.beats} beats — "
                             f"{progress.finding[:90] or 'nothing I tried moved it'}")
            return progress
        except Exception:  # noqa: BLE001 — a failed beat is a null result, never a crash
            return None

    #: When one approach stops working, the next one worth trying. Each goal escalates at most
    #: once — a strategy change is a real move; an endless cycle of them is not.
    _ESCALATION = {"connect": "understand", "ground": "understand", "understand": "verify"}

    def _escalate(self, goal: Goal) -> bool:
        nxt = self._ESCALATION.get(goal.kind)
        if not nxt or goal.note.startswith("escalated"):
            return False
        goal.note = (f"escalated from {goal.kind} to {nxt}: rewiring was not moving it, "
                     f"so it needs new information")
        goal.kind = nxt
        return True

    def _advance(self, goal: Goal) -> Progress:
        """Spend one real step on this goal, through whichever faculty fits its kind."""
        brain = self.brain
        try:
            if goal.kind == "connect":
                return self._advance_connect(brain, goal)
            if goal.kind == "ground":
                return self._advance_ground(brain, goal)
            if goal.kind == "formalise":
                return self._advance_formalise(brain, goal)
            if goal.kind == "verify":
                return self._advance_verify(brain, goal)
            return self._advance_understand(brain, goal)
        except Exception as exc:  # noqa: BLE001
            return Progress(action=goal.kind, finding=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _advance_connect(brain: Any, goal: Goal) -> Progress:
        """Give an isolated concept something to be connected to."""
        graph = getattr(brain, "graph", None)
        if graph is None:
            return Progress(action="connect", finding="no graph to connect in")
        before = len(graph.neighbours(goal.subject, k=32))
        semantics = getattr(brain, "semantics", None)
        near = semantics.similar(goal.subject, k=4, candidates=list(graph.nodes)) \
            if semantics is not None else []
        if near:
            graph.activate([goal.subject] + [n for n, _s, _r in near], decay=False)
        after = len(graph.neighbours(goal.subject, k=32))
        gained = 0.25 if after > before else 0.0
        return Progress(action="connect", gained=gained, ok=gained > 0.0,
                        finding=(f"linked to {', '.join(n for n, _s, _r in near[:3])}"
                                 if gained else "nothing in the space is near it yet"))

    @staticmethod
    def _advance_ground(brain: Any, goal: Goal) -> Progress:
        """Try to put something behind a word she uses with nothing behind it."""
        ground = getattr(brain, "ground", None)
        if ground is None:
            return Progress(action="ground", finding="grounding is not available")
        got = ground.understanding(goal.subject)
        grounded = bool(getattr(got, "grounded", False))
        return Progress(action="ground", gained=0.5 if grounded else 0.0, ok=grounded,
                        finding=(str(getattr(got, "explanation", ""))[:140] if grounded
                                 else "still nothing behind it — I need to read about it"))

    @staticmethod
    def _advance_understand(brain: Any, goal: Goal) -> Progress:
        """Reason about it, and take the answer only at the tier it actually earned."""
        reasoner = getattr(brain, "reason", None)
        if reasoner is None:
            return Progress(action="understand", finding="no reasoner attached")
        chain = reasoner.solve(f"what is {goal.subject}")
        if not chain.answered:
            return Progress(action="understand", finding=chain.why[:140])
        gained = 0.5 if chain.verifiable else 0.2
        return Progress(action="understand", gained=gained, ok=True,
                        finding=f"[{chain.tier}] {chain.answer[:120]}")

    @staticmethod
    def _advance_formalise(brain: Any, goal: Goal) -> Progress:
        genesis = getattr(brain, "axiom", None)
        if genesis is None or not genesis.available():
            return Progress(action="formalise", finding="no axiom genesis available")
        got = genesis.invent(goal.subject)
        return Progress(action="formalise", gained=0.5 if got.promoted else 0.0,
                        ok=got.promoted,
                        finding=(got.system.render()[:140] if got.system is not None
                                 else got.refused[:140]))

    @staticmethod
    def _advance_verify(brain: Any, goal: Goal) -> Progress:
        """Look at what her own record says about the faculty she is worst at."""
        metacog = getattr(brain, "metacog", None)
        if metacog is None:
            return Progress(action="verify", finding="no meta-cognition to read")
        stats = metacog.stats() or {}
        record = (stats.get("specialists") or {}).get(goal.subject)
        if not record:
            return Progress(action="verify",
                            finding=f"no track record for {goal.subject} yet")
        score = float(record.get("score", 0.5))
        return Progress(action="verify", gained=0.34, ok=True,
                        finding=(f"{goal.subject}: score {score:.2f} over "
                                 f"{int(record.get('wins', 0))} wins"))

    # ---- the Master is sovereign over the agenda -------------------------------- #
    def veto(self, goal_id: str, *, reason: str = "") -> bool:
        """The Master drops a goal. Not a request — it is dropped."""
        try:
            goal = self.goals.get(str(goal_id))
            if goal is None:
                return False
            goal.status, goal.note = _VETOED, reason or "vetoed by the Master"
            goal.touched = time.time()
            self.vetoed += 1
            return True
        except Exception:  # noqa: BLE001
            return False

    def redirect(self, goal_id: str, subject: str, *, kind: str = "") -> bool:
        """The Master points an existing goal somewhere else, keeping its history."""
        try:
            goal = self.goals.get(str(goal_id))
            if goal is None or not str(subject).strip():
                return False
            goal.subject = str(subject).strip()
            if kind in GOAL_KINDS:
                goal.kind = kind
            goal.status, goal.progress, goal.stalled = _OPEN, 0.0, 0
            goal.why = "redirected by the Master"
            goal.touched = time.time()
            return True
        except Exception:  # noqa: BLE001
            return False

    def add(self, subject: str, *, kind: str = "understand", priority: float = 0.9,
            why: str = "asked for by the Master") -> Optional[Goal]:
        """A goal the Master hands her. Priority is high because it was asked for."""
        try:
            if not str(subject).strip():
                return None
            goal = Goal(subject=str(subject).strip(),
                        kind=kind if kind in GOAL_KINDS else "understand",
                        why=why, priority=max(0.0, min(1.0, float(priority))))
            self._add(goal)
            return goal
        except Exception:  # noqa: BLE001
            return None

    # ---- the briefing ------------------------------------------------------------ #
    def brief(self, *, since: Optional[float] = None) -> Briefing:
        """What she would tell the Master, unprompted: chose, found, stuck, declined."""
        out = Briefing()
        try:
            cutoff = float(since if since is not None else self._last_briefed)
            out.since_s = max(0.0, time.time() - cutoff)
            for goal in sorted(self.goals.values(), key=lambda g: -g.touched):
                if goal.touched < cutoff and goal.status == _OPEN:
                    continue
                line = f"{goal.subject} ({goal.kind})"
                if goal.status == _DECLINED:
                    out.declined.append(f"{line} — {goal.note}")
                elif goal.status == _STUCK:
                    out.stuck.append(f"{line} — {goal.note}")
                elif goal.status in (_DONE,) or goal.findings:
                    best = max(goal.findings, key=lambda p: p.gained, default=None)
                    if best is not None and best.ok:
                        out.found.append(f"{line}: {best.finding}")
                if goal.status == _OPEN:
                    out.chose.append(f"{line} — {goal.why}")
            out.text = self._say(out)
            self._last_briefed = time.time()
            return out
        except Exception:  # noqa: BLE001
            return out

    @staticmethod
    def _say(brief: Briefing) -> str:
        parts: List[str] = []
        if brief.chose:
            parts.append("I am working on: " + "; ".join(brief.chose[:4]) + ".")
        if brief.found:
            parts.append("What I found: " + "; ".join(brief.found[:4]) + ".")
        if brief.stuck:
            parts.append("Where I am stuck: " + "; ".join(brief.stuck[:3]) + ".")
        if brief.declined:
            parts.append("What I declined, and why: " + "; ".join(brief.declined[:2]) + ".")
        if not parts:
            return "I have not set myself anything worth reporting yet."
        return " ".join(parts)

    @staticmethod
    def _stopped(oversight: Any) -> bool:
        if oversight is None:
            return False
        try:
            for attr in ("scrammed", "is_scrammed", "paused", "is_paused"):
                probe = getattr(oversight, attr, None)
                if probe is None:
                    continue
                if bool(probe() if callable(probe) else probe):
                    return True
            return False
        except Exception:  # noqa: BLE001 — an unreadable oversight is treated as STOP
            return True

    # ---- introspection ------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            by_status: Dict[str, int] = {}
            for goal in self.goals.values():
                by_status[goal.status] = by_status.get(goal.status, 0) + 1
            return {
                "goals": len(self.goals), "by_status": by_status,
                "beats": self.beats, "formed": self.formed,
                "completed": self.completed, "vetoed": self.vetoed,
                "declined": self.declined,
                "open": [g.subject for g in self.open_goals()],
                "every_s": self.every_s,
                "note": ("goals come from her MEASURED gaps and her drives, not from nowhere — "
                         "a concept she keeps meeting without connecting, a word with nothing "
                         "behind it, the faculty her own record calls weakest. Reach is her "
                         "simulators and her knowledge, not omniscience: a goal she cannot "
                         "advance is marked stuck with what she tried. The Master can veto or "
                         "redirect any goal, and /scram stops all pursuit"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx agenda`` prints."""
        try:
            if not self.goals:
                return ("I have not set myself any goals yet. Let me run a while — they come "
                        "from what I measure about myself, not from a list.")
            lines = [f"{len(self.goals)} goal(s), {self.beats} beats of work:"]
            for goal in sorted(self.goals.values(), key=lambda g: (-g.priority, g.touched)):
                lines.append("  " + goal.render())
                if goal.why:
                    lines.append(f"      because {goal.why}")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence — this is the layer ------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"beats": self.beats, "formed": self.formed, "completed": self.completed,
                "vetoed": self.vetoed, "declined": self.declined,
                "last_briefed": self._last_briefed,
                "goals": [g.to_dict() for g in self.goals.values()]}

    def load_dict(self, data: Any) -> bool:
        """Restore the stack. Without this the layer does not exist — it *is* the layer."""
        try:
            if not isinstance(data, dict):
                return False
            self.beats = int(data.get("beats", 0))
            self.formed = int(data.get("formed", 0))
            self.completed = int(data.get("completed", 0))
            self.vetoed = int(data.get("vetoed", 0))
            self.declined = int(data.get("declined", 0))
            self._last_briefed = float(data.get("last_briefed", time.time()))
            self.goals = {}
            for raw in data.get("goals", []) or []:
                if not isinstance(raw, dict):
                    continue
                goal = Goal(
                    id=str(raw.get("id") or uuid.uuid4().hex[:8]),
                    subject=str(raw.get("subject", "")), kind=str(raw.get("kind", "understand")),
                    why=str(raw.get("why", "")), priority=float(raw.get("priority", 0.5)),
                    status=str(raw.get("status", _OPEN)),
                    progress=float(raw.get("progress", 0.0)), beats=int(raw.get("beats", 0)),
                    stalled=int(raw.get("stalled", 0)), parent=str(raw.get("parent", "")),
                    note=str(raw.get("note", "")), born=float(raw.get("born", time.time())),
                    touched=float(raw.get("touched", time.time())))
                for f in raw.get("findings", []) or []:
                    if isinstance(f, dict):
                        goal.findings.append(Progress(
                            at=float(f.get("at", 0.0)), action=str(f.get("action", "")),
                            gained=float(f.get("gained", 0.0)),
                            finding=str(f.get("finding", "")), ok=bool(f.get("ok", False))))
                if goal.subject:
                    self.goals[goal.id] = goal
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
