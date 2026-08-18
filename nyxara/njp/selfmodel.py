"""NYXARA · njp/selfmodel.py — what she is actually good at, measured (🪞, NJP V.02).

A self-model that lists capabilities is a brochure. This one carries **numbers she did not choose**:
every capability is a Beta posterior over its own observed successes and failures, so
*"my grounding is reliable and my planning is not"* is a claim backed by counts rather than a
sentiment. That is the difference between metacognition and self-description.

Two things are built on top of that, and they are the reason it is worth measuring:

**Confidence that knows its own worth.** :meth:`trust` discounts a stated confidence by how well
that capability has actually performed. A subsystem that has been wrong eight times out of ten
should not be able to hand up a 0.9 and have it believed, and before this nothing stopped it.

**Learning how to learn.** :class:`MetaLearner` treats her own strategies as arms of a bandit —
recall breadth, settle depth, reasoning depth, hypothesis count — and shifts toward whichever is
*measurably* paying on the task class at hand. Not a preference, not a heuristic: a strategy is
adopted because trials say it wins, and dropped when they stop saying so.

**Resource allocation is itself learned.** :meth:`allocate` reads which faculty has been earning
its cost per task class — memory, simulation, reasoning depth, external information — and moves
compute accordingly. So the architecture is not fixed at what it shipped with: what she spends
effort on is a function of what has repaid effort.

Honest, as everywhere here: this is a **computational** self-model. It is her performance record
and a policy over it. Nothing in this file is introspection, and none of it is consciousness — the
numbers describe behaviour that was observed, which is all they claim to do. She can be wrong
about herself here in exactly the way a badly-calibrated estimate is wrong, and
:meth:`calibration_of` is what surfaces that rather than hiding it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Capability", "Strategy", "Allocation", "SelfModel", "MetaLearner"]


# The faculties worth tracking separately. Each is something that can succeed or fail on its own,
# which is the only thing that makes a separate posterior meaningful.
CAPABILITIES = (
    "grounding",        # did the words become the right structure?
    "recall",           # did the right thing come back?
    "prediction",       # did what she expected actually happen?
    "reasoning",        # was the conclusion right given the facts?
    "planning",         # did the plan achieve what it predicted?
    "tool_use",         # did the tool call do what she expected?
    "language",         # did the phrasing carry the content?
)

#: How deep a chain has to be before it is tracked apart from a short one. A two-hop composition
#: and a five-hop one fail for different reasons and at different rates, and averaging them makes
#: the number useless for deciding whether to believe *this* answer.
_DEEP_CHAIN = 3


def mode_of(derivation: Any) -> str:
    """Which *kind* of reasoning produced this answer, as a capability name.

    ``reasoning`` was one flat posterior, so "how reliable am I here" could only ever be answered
    about reasoning-in-general — and the ways this brain reasons do not fail at the same rate.
    A stored fact read back is nearly always right; a two-hop composition is a conjecture priced
    at 0.70 before anything else discounts it; a schema transfer is defeasible by construction.
    Folding all three into one number means a confident direct answer subsidises a shaky composed
    one, and the composed one is exactly where a caller most needs the warning.

    The label is not invented here. :class:`~nyxara.njp.core.Derivation` has carried ``kind`` from
    the start — ``direct | composed | schema | simulated`` — and a composed answer carries its
    chain in ``support``, so the hop count is already on the object. This only reads them.

    Names are hierarchical (``reasoning:composed:deep``) so :meth:`SelfModel.trust` can fall back
    to the parent while a new mode is still untested.
    """
    kind = str(getattr(derivation, "kind", "") or "").strip().lower()
    if not kind:
        return "reasoning"
    if kind != "composed":
        return f"reasoning:{kind}"
    hops = len(list(getattr(derivation, "support", None) or ()))
    return f"reasoning:composed:{'deep' if hops >= _DEEP_CHAIN else 'short'}"


@dataclass
class Capability:
    """One faculty's record. Beta(α, β) over observed successes and failures.

    A Beta posterior rather than a running mean because it carries **how much evidence there is**
    as well as the rate. Two successes out of two is a mean of 1.0 and means almost nothing; the
    posterior says so, and :attr:`certainty` is what a caller reads to find that out.
    """

    name: str = ""
    alpha: float = 1.0                 # successes + 1 (uniform prior)
    beta: float = 1.0                  # failures + 1
    observations: int = 0
    last_seen: float = field(default_factory=time.time)

    @property
    def level(self) -> float:
        """Posterior mean — the success rate she should expect from this faculty."""
        total = self.alpha + self.beta
        return (self.alpha / total) if total > 0 else 0.5

    @property
    def certainty(self) -> float:
        """How much the level is worth. Grows with evidence, never reaches 1."""
        return 1.0 - (1.0 / (1.0 + (self.alpha + self.beta - 2.0) / 8.0))

    @property
    def reliable(self) -> bool:
        """Good, and known to be good. Both halves are required."""
        return self.level >= 0.7 and self.observations >= _MIN_OBSERVATIONS

    @property
    def weak(self) -> bool:
        """Measurably poor — not merely unproven. An untested faculty is not a weak one."""
        return self.level < 0.5 and self.observations >= _MIN_OBSERVATIONS

    def observe(self, success: float, *, weight: float = 1.0) -> None:
        """Record one outcome. Partial credit is allowed — most outcomes are not binary."""
        success = max(0.0, min(1.0, float(success)))
        weight = max(0.0, float(weight))
        self.alpha += success * weight
        self.beta += (1.0 - success) * weight
        self.observations += 1
        self.last_seen = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": round(self.level, 4),
                "certainty": round(self.certainty, 4), "observations": self.observations,
                "reliable": self.reliable, "weak": self.weak}


@dataclass
class Strategy:
    """One way of doing something, with the record that decides whether she keeps using it."""

    name: str = ""
    arm: str = ""                      # which knob this sets
    value: Any = None                  # what it sets it to
    trials: int = 0
    reward: float = 0.0                # cumulative

    @property
    def mean(self) -> float:
        return (self.reward / self.trials) if self.trials else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arm": self.arm, "value": self.value,
                "trials": self.trials, "mean": round(self.mean, 4)}


@dataclass
class Allocation:
    """How much of a turn's effort each faculty should get, and why."""

    weights: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    learned: bool = False              # False ⇒ still the default split, not yet earned

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": {k: round(v, 4) for k, v in self.weights.items()},
                "reason": self.reason, "learned": self.learned}


_MIN_OBSERVATIONS = 8
_MIN_TRIALS = 5


class SelfModel:
    """Her measured performance record, and what follows from it."""

    def __init__(self, *, ledger: Any = None) -> None:
        # Reuse the repo's CompetenceLedger when it is available: it holds the same Beta shape and
        # is already seeded from her boot-time self-assessment, so the two do not drift apart.
        self.ledger = ledger if ledger is not None else self._build_ledger()
        self.capabilities: Dict[str, Capability] = {
            name: Capability(name=name) for name in CAPABILITIES
        }
        self.state: Dict[str, Any] = {}
        self.updates = 0

    @staticmethod
    def _build_ledger() -> Any:
        try:
            from nyxara.memory.competence import CompetenceLedger
            return CompetenceLedger()
        except Exception:  # noqa: BLE001
            return None

    # ---- observing herself ------------------------------------------------- #
    def observe(self, capability: str, success: float, *, weight: float = 1.0) -> Optional[Capability]:
        """Record one outcome for one faculty. The only way this model changes."""
        try:
            name = str(capability)
            got = self.capabilities.get(name)
            if got is None:
                got = Capability(name=name)
                self.capabilities[name] = got
            got.observe(success, weight=weight)
            self.updates += 1
            if self.ledger is not None:
                try:
                    self.ledger.observe(name, float(success), weight)
                except Exception:  # noqa: BLE001
                    pass
            return got
        except Exception:  # noqa: BLE001
            return None

    def observe_diagnosis(self, diagnosis: Any) -> None:
        """Take a :class:`~nyxara.njp.predict.Diagnosis` as a failure for the organ it blamed.

        This is the join that makes the error taxonomy pay twice: the same attribution that routes
        a repair also updates the record of which faculty keeps needing repairing. An
        ``unattributed`` error updates nothing — it is not evidence against any particular
        faculty, and spreading it across all of them would slowly convince her she is bad at
        everything.
        """
        try:
            kind = str(getattr(diagnosis, "kind", ""))
            if not kind or kind == "unattributed":
                return
            self.observe(_KIND_TO_CAPABILITY.get(kind, kind), 0.0)
        except Exception:  # noqa: BLE001
            pass

    def update_state(self, **kv: Any) -> None:
        """Current goals, uncertainty, active task — the volatile half of the self-model."""
        try:
            self.state.update(kv)
        except Exception:  # noqa: BLE001
            pass

    # ---- what follows from it ----------------------------------------------- #
    def level(self, capability: str) -> float:
        got = self.capabilities.get(str(capability))
        return got.level if got is not None else 0.5

    def trust(self, capability: str, stated: float) -> float:
        """Discount a stated confidence by how well that faculty has actually done.

        Only ever downward, and only once there is enough evidence to justify it: an untested
        faculty is not a bad one, and penalising it would make a cold start permanently timid.
        A faculty that has been wrong eight times in ten should not be able to hand up a 0.9 and
        have it believed, and nothing stopped that before.
        """
        try:
            stated = max(0.0, min(1.0, float(stated)))
            got = self._measured(str(capability))
            if got is None:
                return stated
            return stated * max(0.0, min(1.0, got.level))
        except Exception:  # noqa: BLE001
            return stated

    def _measured(self, capability: str) -> Optional[Capability]:
        """The most specific record for this name that has enough evidence to be worth using.

        Capability names are hierarchical — ``reasoning:composed:deep`` sits under
        ``reasoning:composed`` under ``reasoning`` — and without this walk, splitting a faculty
        would make the model *less* informative rather than more: every new mode starts at zero
        observations, falls under :data:`_MIN_OBSERVATIONS`, and is therefore not discounted at
        all. A brain that has just learned to compose would get a free pass on exactly the
        answers that most deserve doubt.

        So: use the specific record once it has earned the right to speak, and until then borrow
        the parent's. Reliability inherits downwards until the child has its own evidence.
        """
        name = str(capability)
        while name:
            got = self.capabilities.get(name)
            if got is not None and got.observations >= _MIN_OBSERVATIONS:
                return got
            if ":" not in name:
                return None
            name = name.rsplit(":", 1)[0]
        return None

    def weakest(self) -> Optional[Capability]:
        """The faculty most worth improving. Untested ones do not qualify."""
        measured = [c for c in self.capabilities.values()
                    if c.observations >= _MIN_OBSERVATIONS]
        return min(measured, key=lambda c: c.level) if measured else None

    def unreliable_for(self, task: str) -> List[str]:
        """Which faculties this task depends on that she is measurably poor at.

        The reading behind *"is task me meri reliability low hai"* — and it is a report of counts,
        not a mood.
        """
        try:
            needed = _TASK_CAPABILITIES.get(str(task).lower(), ())
            return [n for n in needed
                    if (c := self.capabilities.get(n)) is not None and c.weak]
        except Exception:  # noqa: BLE001
            return []

    def calibration_of(self, capability: str, stated: float) -> float:
        """Gap between a stated confidence and what that faculty actually delivers."""
        try:
            got = self.capabilities.get(str(capability))
            if got is None or got.observations < _MIN_OBSERVATIONS:
                return 0.0
            return abs(float(stated) - got.level)
        except Exception:  # noqa: BLE001
            return 0.0

    def report(self) -> Dict[str, Any]:
        """What she can and cannot do, in the terms she measured it."""
        weak = [c.name for c in self.capabilities.values() if c.weak]
        reliable = [c.name for c in self.capabilities.values() if c.reliable]
        untested = [c.name for c in self.capabilities.values()
                    if c.observations < _MIN_OBSERVATIONS]
        return {"reliable": reliable, "weak": weak, "untested": untested,
                "weakest": self.weakest().name if self.weakest() else None,
                "capabilities": {n: c.to_dict() for n, c in self.capabilities.items()},
                "state": dict(self.state), "updates": self.updates}

    def stats(self) -> Dict[str, Any]:
        return self.report()

    def to_dict(self) -> Dict[str, Any]:
        return {"capabilities": [{"name": c.name, "alpha": c.alpha, "beta": c.beta,
                                  "observations": c.observations}
                                 for c in self.capabilities.values()],
                "state": dict(self.state), "updates": self.updates}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for row in (d.get("capabilities") or []):
                name = str(row.get("name", ""))
                if not name:
                    continue
                self.capabilities[name] = Capability(
                    name=name, alpha=float(row.get("alpha", 1.0)),
                    beta=float(row.get("beta", 1.0)),
                    observations=int(row.get("observations", 0)))
            self.state.update(d.get("state") or {})
            self.updates = int(d.get("updates", 0))
        except Exception:  # noqa: BLE001
            pass


class MetaLearner:
    """Her own strategies as bandit arms — learning *how* to learn, measured.

    Reuses :class:`~nyxara.growth.meta.MetaLearner`'s idea rather than its object, because the
    arms here are NJP's own knobs and the reward is NJP's own turn outcome. What is not reused is
    more important than what is: nothing here decides a strategy is good because it sounds
    principled. A strategy wins when its trials say so.
    """

    def __init__(self, self_model: Any = None, *, explore: float = 0.15,
                 min_trials: int = _MIN_TRIALS) -> None:
        self.self_model = self_model
        # How often she tries something other than her current best. Zero would lock her into
        # whatever won first, which on thin evidence is close to arbitrary.
        self.explore = max(0.0, min(1.0, float(explore)))
        self.min_trials = max(1, int(min_trials))
        self.strategies: Dict[str, Dict[str, Strategy]] = {}   # arm -> name -> strategy
        self._pending: Dict[str, Strategy] = {}
        self.selections = 0
        self.switches = 0
        self._rng_state = 0x2545F4914F6CDD1D

    # ---- arms --------------------------------------------------------------- #
    def register(self, arm: str, name: str, value: Any) -> Strategy:
        """Declare one option for one knob."""
        bucket = self.strategies.setdefault(str(arm), {})
        got = bucket.get(str(name))
        if got is None:
            got = Strategy(name=str(name), arm=str(arm), value=value)
            bucket[str(name)] = got
        return got

    def _random(self) -> float:
        """Deterministic PRNG — exploration must be reproducible or a run cannot be replayed."""
        self._rng_state = (self._rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((self._rng_state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def choose(self, arm: str) -> Optional[Strategy]:
        """Pick a strategy for this knob: the best known, or an exploration."""
        try:
            bucket = self.strategies.get(str(arm)) or {}
            if not bucket:
                return None
            self.selections += 1
            untried = [s for s in bucket.values() if s.trials < self.min_trials]
            if untried:
                chosen = untried[0]          # never prefer a strategy over an untested one
            elif self._random() < self.explore:
                options = list(bucket.values())
                chosen = options[int(self._random() * len(options)) % len(options)]
            else:
                chosen = max(bucket.values(), key=lambda s: s.mean)
            previous = self._pending.get(str(arm))
            if previous is not None and previous.name != chosen.name:
                self.switches += 1
            self._pending[str(arm)] = chosen
            return chosen
        except Exception:  # noqa: BLE001
            return None

    def reward(self, arm: str, value: float) -> None:
        """Tell the chosen strategy how the turn actually went."""
        try:
            chosen = self._pending.get(str(arm))
            if chosen is None:
                return
            chosen.trials += 1
            chosen.reward += max(0.0, min(1.0, float(value)))
        except Exception:  # noqa: BLE001
            pass

    def best(self, arm: str) -> Optional[Strategy]:
        """The current winner for one knob, once it has actually been tried enough."""
        bucket = self.strategies.get(str(arm)) or {}
        tried = [s for s in bucket.values() if s.trials >= self.min_trials]
        return max(tried, key=lambda s: s.mean) if tried else None

    # ---- resource allocation ------------------------------------------------- #
    def allocate(self, task: str = "") -> Allocation:
        """Where this turn's effort should go, from what has been paying.

        Defaults to an even split and says so. A learned allocation is only claimed once the
        capability record has enough evidence to justify moving compute — announcing a "learned"
        allocation derived from four observations would be the same overclaiming this whole file
        exists to avoid.
        """
        out = Allocation(weights={f: 1.0 / len(_FACULTIES) for f in _FACULTIES},
                         reason="even split — nothing has earned a larger share yet")
        try:
            model = self.self_model
            if model is None:
                return out
            levels: Dict[str, float] = {}
            for faculty in _FACULTIES:
                capability = model.capabilities.get(_FACULTY_CAPABILITY.get(faculty, faculty))
                if capability is None or capability.observations < _MIN_OBSERVATIONS:
                    continue
                levels[faculty] = capability.level
            if len(levels) < 2:
                return out
            total = sum(levels.values()) or 1.0
            out.weights = {f: levels.get(f, 0.0) / total for f in _FACULTIES}
            out.learned = True
            best = max(levels, key=levels.get)
            out.reason = f"{best} has been paying best ({levels[best]:.0%} success)"
            return out
        except Exception:  # noqa: BLE001
            return out

    def stats(self) -> Dict[str, Any]:
        return {"selections": self.selections, "switches": self.switches,
                "arms": {arm: {n: s.to_dict() for n, s in bucket.items()}
                         for arm, bucket in self.strategies.items()},
                "best": {arm: (self.best(arm).name if self.best(arm) else None)
                         for arm in self.strategies},
                "allocation": self.allocate().to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        return {"strategies": [s.to_dict() for bucket in self.strategies.values()
                               for s in bucket.values()],
                "counters": {"selections": self.selections, "switches": self.switches}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for row in (d.get("strategies") or []):
                arm, name = str(row.get("arm", "")), str(row.get("name", ""))
                if not arm or not name:
                    continue
                strategy = self.register(arm, name, row.get("value"))
                strategy.trials = int(row.get("trials", 0))
                strategy.reward = float(row.get("mean", 0.0)) * strategy.trials
            counters = d.get("counters") or {}
            self.selections = int(counters.get("selections", 0))
            self.switches = int(counters.get("switches", 0))
        except Exception:  # noqa: BLE001
            pass


# An error kind names the organ that failed; the capability record is keyed by faculty. These are
# the same thing under two names, and the mapping is here so neither module has to know the other's.
_KIND_TO_CAPABILITY: Dict[str, str] = {
    "perception": "grounding", "grounding": "grounding", "memory": "recall",
    "world_model": "prediction", "reasoning": "reasoning", "planning": "planning",
    "language": "language",
}

# Which faculties a kind of task actually leans on. Used only to answer "am I reliable at this",
# never to decide what she is allowed to attempt.
_TASK_CAPABILITIES: Dict[str, Tuple[str, ...]] = {
    "question": ("grounding", "recall", "reasoning"),
    "command": ("planning", "tool_use"),
    "prediction": ("prediction", "reasoning"),
    "conversation": ("grounding", "language"),
}

# What a turn can spend effort on, and the capability whose record justifies spending it there.
_FACULTIES = ("memory", "simulation", "reasoning", "external")
_FACULTY_CAPABILITY: Dict[str, str] = {
    "memory": "recall", "simulation": "prediction",
    "reasoning": "reasoning", "external": "tool_use",
}
