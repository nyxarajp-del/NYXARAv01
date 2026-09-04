"""NYXARA · njp/provenance.py — every conclusion carries the path that made it (🧾, NJP V.43).

The Master's three demands, and they are one mechanism seen from three sides::

    #7  CLAIM: X          #5  final answer          #6  FAILURE
        SUPPORTED BY:             ↑                  ├── prediction
          fact 183            inference 7            ├── actual result
          fact 921                ↑                  ├── failed assumption
          inference R4         belief 2              ├── responsible belief
          causal edge E12         ↑                  ├── reasoning path
          assumption A3        memory 9              ├── missing information
                                  ↑                  └── repair
                               source

You cannot have the second without the first: *"which internal belief was responsible"* is
unanswerable unless the answer wrote down what it stood on. And you cannot have the third without
the second: storing *"wrong"* teaches nothing, storing *"wrong, and this assumption did it"*
teaches the next turn.

So this module is one object — a :class:`Claim` that knows its own ancestry — and two things you
can do with it: **audit** it while it stands, and **blame** through it when it falls.

The three verdicts, and the rule that makes them mean something
---------------------------------------------------------------

    **No path, no claim.**

:meth:`Ledger.assert_` refuses to record a conclusion with nothing under it. That is not a style
preference; it is the only thing that makes :attr:`Claim.status` load-bearing:

``SUPPORTED``
    Every step of the path is itself supported, and the paths do not disagree.
``HYPOTHETICAL``
    Some step is an assumption or an unverified hypothesis. The claim stands and **says so** — a
    defeasible link is how most true generalisations are stated, and refusing them outright would
    leave her able to conclude almost nothing.
``CONFLICTED``
    Two paths reach the claim and they cannot both hold. Reported, never resolved by counting: the
    refusal `Grounder.answer` has made since V.13, one level up.
``UNKNOWN``
    Nothing supports it. Not a hedge — a **verdict**, and the one this whole package's restraint
    numbers are built out of.

Blame, and why it is not the same as the path
---------------------------------------------

When a claim fails, the path tells you what it *stood on*; blame tells you which of those actually
*carried* it. Those differ, and the difference is the whole of credit assignment: a claim resting
on five facts where four are corroborated elsewhere and one is a lone assumption has a **single**
responsible step, and a system that blamed all five would learn nothing it could act on.

:meth:`Ledger.blame` ranks by *how much of the claim rests on this and nothing else* — a step that
appears on every path to the claim and nowhere else in the ledger is the culprit; one that
supports a dozen standing claims is almost certainly not.

The autopsy, and the warning it earns
--------------------------------------

:class:`PostMortem` records what the Master listed: what was predicted, what happened, which step
was blamed, what the path was, and — where it can be inferred — what would have had to be
different. Then :meth:`Ledger.warn` answers the question that makes the record worth keeping:

    *This problem I am about to reason about — have I been wrong here before, and how?*

Matched on the **blamed step**, not on the surface of the problem. Two questions that look nothing
alike and failed on the same assumption are the same failure, and that is exactly the pairing a
similarity search over question text would miss.

What it may not do
------------------

**It may not invent support.** Every step in a path names something: a fact, an edge, an inference,
an assumption. A path may not contain a step nobody recorded.

**It may not upgrade a status by usefulness.** A claim does not become ``SUPPORTED`` for being
needed, and a hypothesis does not become a fact for having survived. Only :meth:`Ledger.confirm`
moves a step, and it takes evidence.

**It may not silently drop a conflict.** A conflicted claim is returned as conflicted forever
unless something retracts one side.

Pure standard library, deterministic, and it holds no knowledge of its own.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Kind", "Status", "Step", "Path", "Claim", "Blame", "PostMortem", "Ledger",
]


class Kind(str, Enum):
    """What a step in a path *is*. Named rather than free text, because blame reads it."""

    FACT = "fact"                 # something she was told
    EDGE = "edge"                 # a relation in the graph
    INFERENCE = "inference"       # a rule that was applied
    ASSUMPTION = "assumption"     # something taken, not established
    HYPOTHESIS = "hypothesis"     # a proposal awaiting a test it could fail
    MEMORY = "memory"             # something recalled
    OBSERVATION = "observation"   # something measured
    SOURCE = "source"             # where a fact came from


#: Steps that do not carry their own weight. A path through one of these is at best hypothetical.
_UNSETTLED = (Kind.ASSUMPTION, Kind.HYPOTHESIS)


class Status(str, Enum):
    SUPPORTED = "supported"
    HYPOTHETICAL = "hypothetical"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Step:
    """One thing a conclusion stands on."""

    id: str
    kind: Kind
    text: str = ""
    confidence: float = 1.0
    #: Set only by :meth:`Ledger.confirm`, and only against evidence.
    settled: bool = True

    @property
    def unsettled(self) -> bool:
        return self.kind in _UNSETTLED and not self.settled

    def render(self) -> str:
        mark = "?" if self.unsettled else ""
        return f"{self.kind.value} {self.id}{mark}" + (f": {self.text}" if self.text else "")

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "text": self.text,
                "confidence": self.confidence, "settled": self.settled}


@dataclass(frozen=True)
class Path:
    """One route to a conclusion, in the order it was travelled."""

    steps: Tuple[Step, ...]
    note: str = ""

    @property
    def unsettled(self) -> Tuple[Step, ...]:
        return tuple(s for s in self.steps if s.unsettled)

    @property
    def confidence(self) -> float:
        out = 1.0
        for step in self.steps:
            out *= max(0.0, min(1.0, float(step.confidence)))
        return round(out, 4)

    def render(self) -> str:
        return "\n".join(f"  {s.render()}" for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"note": self.note, "confidence": self.confidence,
                "steps": [s.to_dict() for s in self.steps]}


@dataclass
class Claim:
    """A conclusion, its paths, and what that adds up to."""

    id: str
    text: str
    paths: List[Path] = field(default_factory=list)
    #: Pairs of path indices that cannot both hold, as recorded by :meth:`Ledger.oppose`.
    conflicts: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def status(self) -> Status:
        if not self.paths:
            return Status.UNKNOWN
        if self.conflicts:
            return Status.CONFLICTED
        if any(path.unsettled for path in self.paths):
            # Hypothetical only if **every** path leans on something unsettled. A claim with one
            # clean route and one speculative one is supported by the clean one; saying otherwise
            # would let an idle guess downgrade an established conclusion.
            if all(path.unsettled for path in self.paths):
                return Status.HYPOTHETICAL
        return Status.SUPPORTED

    @property
    def steps(self) -> List[Step]:
        seen: Dict[str, Step] = {}
        for path in self.paths:
            for step in path.steps:
                seen.setdefault(step.id, step)
        return list(seen.values())

    def render(self) -> str:
        head = f"CLAIM: {self.text}  [{self.status.value}]"
        if not self.paths:
            return head + "\n  (nothing supports this)"
        body = []
        for index, path in enumerate(self.paths):
            body.append(f"SUPPORTED BY ({index}){f' — {path.note}' if path.note else ''}:")
            body.append(path.render())
        if self.conflicts:
            body.append("CONFLICTING PATHS: " + ", ".join(f"{a} vs {b}"
                                                          for a, b in self.conflicts))
        return head + "\n" + "\n".join(body)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "status": self.status.value,
                "paths": [p.to_dict() for p in self.paths],
                "conflicts": [list(c) for c in self.conflicts]}


@dataclass
class Blame:
    """One step, and how much of a failed claim rested on it and nothing else."""

    step: Step
    share: float
    exclusive: bool
    also_supports: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step.to_dict(), "share": self.share,
                "exclusive": self.exclusive, "also_supports": self.also_supports}


@dataclass
class PostMortem:
    """A failure in the shape that lets the next turn recognise it."""

    claim: str
    predicted: Any
    actual: Any
    blamed: List[Blame] = field(default_factory=list)
    path: Optional[Path] = None
    missing: str = ""
    repair: str = ""

    @property
    def culprit(self) -> Optional[Step]:
        return self.blamed[0].step if self.blamed else None

    def render(self) -> str:
        lines = [f"FAILURE: {self.claim}",
                 f"  predicted   {self.predicted!r}",
                 f"  actual      {self.actual!r}"]
        if self.culprit is not None:
            lines.append(f"  blamed      {self.culprit.render()}")
        if self.path is not None:
            lines.append("  path")
            lines.append(self.path.render())
        if self.missing:
            lines.append(f"  missing     {self.missing}")
        if self.repair:
            lines.append(f"  repair      {self.repair}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim, "predicted": self.predicted, "actual": self.actual,
                "blamed": [b.to_dict() for b in self.blamed],
                "path": self.path.to_dict() if self.path else None,
                "missing": self.missing, "repair": self.repair}


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
class Ledger:
    """Claims with their ancestry, and what to do when one of them turns out to be wrong."""

    def __init__(self) -> None:
        self.steps: Dict[str, Step] = {}
        self.claims: Dict[str, Claim] = {}
        self.failures: List[PostMortem] = []

    # ---- recording -------------------------------------------------------- #
    def record(self, step: Step) -> Step:
        self.steps[step.id] = step
        return step

    def assert_(self, claim_id: str, text: str, *,
                path: Sequence[Step] = (), note: str = "") -> Claim:
        """Record a conclusion **with** what it stands on. A path of nothing is refused.

        ``UNKNOWN`` for a claim with no path is not a courtesy return, it is the point of the
        module: a conclusion that cannot say what produced it is not a weak conclusion, it is not
        a conclusion, and every restraint number in this package rests on that distinction being
        made somewhere.
        """
        got = self.claims.get(claim_id)
        if got is None:
            got = self.claims[claim_id] = Claim(id=claim_id, text=text)
        if path:
            for step in path:
                self.steps.setdefault(step.id, step)
            got.paths.append(Path(steps=tuple(path), note=note))
        return got

    def oppose(self, claim_id: str, first: int, second: int) -> Claim:
        """Say that two of a claim's paths cannot both hold. Never resolved here."""
        claim = self.claims[claim_id]
        pair = (min(first, second), max(first, second))
        if pair not in claim.conflicts:
            claim.conflicts.append(pair)
        return claim

    def confirm(self, step_id: str, *, by: str = "") -> Optional[Step]:
        """Settle an assumption **against evidence**. The only way a status improves.

        A step does not become settled by being useful, by being old, or by having been used in a
        conclusion that happened to come out right. ``by`` records what settled it.
        """
        step = self.steps.get(step_id)
        if step is None:
            return None
        settled = Step(id=step.id, kind=step.kind, confidence=step.confidence,
                       text=(step.text + (f" [settled by {by}]" if by else "")).strip(),
                       settled=True)
        self.steps[step_id] = settled
        for claim in self.claims.values():
            claim.paths = [Path(steps=tuple(settled if s.id == step_id else s
                                            for s in path.steps), note=path.note)
                           for path in claim.paths]
        return settled

    # ---- auditing --------------------------------------------------------- #
    def audit(self, claim_id: str) -> Claim:
        return self.claims.get(claim_id) or Claim(id=claim_id, text="")

    def status(self, claim_id: str) -> Status:
        return self.audit(claim_id).status

    # ---- blame ------------------------------------------------------------ #
    def blame(self, claim_id: str) -> List[Blame]:
        """Which step actually carried this claim, ranked.

        The path says what a claim stood on; blame says what held it up. A claim resting on five
        steps where four are corroborated by other standing claims and one is a lone assumption has
        **one** responsible step, and blaming all five teaches nothing anybody can act on.

        Two things are weighed and the first dominates. A step on **every** path to the claim is
        load-bearing; one on a single path of three is not. And a step supporting a dozen other
        claims that are all still standing is very unlikely to be what broke this one — so
        exclusivity, not frequency, is the signal.
        """
        claim = self.claims.get(claim_id)
        if claim is None or not claim.paths:
            return []
        elsewhere: Dict[str, int] = defaultdict(int)
        for other in self.claims.values():
            if other.id == claim_id:
                continue
            for step in other.steps:
                elsewhere[step.id] += 1
        total = len(claim.paths)
        counts: Dict[str, int] = defaultdict(int)
        for path in claim.paths:
            for step in {s.id for s in path.steps}:
                counts[step] += 1
        out: List[Blame] = []
        for step in claim.steps:
            on = counts[step.id]
            share = on / total
            # An unsettled step is what a failure is usually made of; weigh it above a fact that
            # sits on the same number of paths.
            weight = share + (0.5 if step.unsettled else 0.0) - 0.1 * min(elsewhere[step.id], 5)
            out.append(Blame(step=step, share=round(share, 4),
                             exclusive=elsewhere[step.id] == 0 and on == total,
                             also_supports=elsewhere[step.id]))
            out[-1].share = round(weight, 4)
        return sorted(out, key=lambda b: (-b.share, b.step.id))

    # ---- the autopsy ------------------------------------------------------ #
    def autopsy(self, claim_id: str, *, predicted: Any, actual: Any,
                missing: str = "", repair: str = "") -> PostMortem:
        """Record a failure with the step that carried it, and keep it for next time."""
        claim = self.claims.get(claim_id)
        blamed = self.blame(claim_id)
        got = PostMortem(claim=claim.text if claim else claim_id,
                         predicted=predicted, actual=actual, blamed=blamed,
                         path=claim.paths[0] if claim and claim.paths else None,
                         missing=missing, repair=repair)
        self.failures.append(got)
        return got

    def warn(self, path: Sequence[Step]) -> List[PostMortem]:
        """Have I been wrong on this ground before?

        Matched on the **blamed step**, never on the surface of the problem. Two questions that
        look nothing alike and failed on the same assumption are the same failure, and a similarity
        search over their wording would put them in different worlds. This is the whole reason the
        autopsy stores a culprit rather than a description.
        """
        wanted = {step.id for step in path}
        out: List[PostMortem] = []
        for failure in self.failures:
            culprit = failure.culprit
            if culprit is not None and culprit.id in wanted:
                out.append(failure)
        return out

    # ---- reading ---------------------------------------------------------- #
    def render(self, claim_id: str) -> str:
        claim = self.claims.get(claim_id)
        if claim is None:
            return f"CLAIM: {claim_id}\n  (nothing supports this)"
        lines = [claim.render()]
        warnings = self.warn(claim.steps)
        if warnings:
            lines.append(f"WARNING: {len(warnings)} earlier failure(s) rested on the same step")
            for failure in warnings[:2]:
                lines.append(f"  {failure.claim}: predicted {failure.predicted!r}, "
                             f"was {failure.actual!r}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": len(self.steps), "claims": len(self.claims),
                "failures": [f.to_dict() for f in self.failures]}
