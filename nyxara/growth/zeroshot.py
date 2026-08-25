"""NYXARA · growth/zeroshot.py — does a skill she invented work on a task it was not invented for?

**Phase 5's milestone is two claims and only one of them was answerable.** ``skills_created > 0``
is a count of abstractions adopted, and :mod:`nyxara.growth.noesis` produces them —
``abs1 := mul($0, sum_(x))`` is a real new primitive in her language, not a macro comment. The
second claim is the one that means something:

    *and the skills transfer to unseen tasks.*

``NoesisEngine`` has always had the hook for it — ``self.transfer.transfer_all(...)``, called on
every SLEEP with the newly adopted abstractions and this cycle's tasks — and **nothing in the
repository implemented that interface**. The attribute defaulted to ``None``, so the guard above
the call was always false and ``transferred`` was structurally 0 on every cycle ever run. A
milestone that cannot be reached is not a hard milestone; it is an unwired one.

**What transfer has to mean here, to be worth counting.** An abstraction that turns up in the
solution it was extracted from has demonstrated nothing — that is the induction, restated. So a
task counts as transfer only when

* the abstraction does **not** appear in any corpus solution for that task — it was not induced
  there, and this is the "unseen" in "unseen tasks";
* a program *using* the abstraction reproduces the task's examples **exactly**, found by bounded
  enumeration over its holes rather than by being told the answer;
* and that program then survives :class:`~nyxara.growth.redteam.RedTeam` against the task's
  oracle, on adversarial inputs it was never shown.

The last of those is not optional and the first version of this module went without it. Fitting a
handful of examples is exactly how a wrong program looks right: ``abs0 := count_gt(x, $0)`` was
confirmed as transferring to ``count_over_2`` via ``abs0(0)`` — counting elements above *zero*,
which agreed with that task's five examples and is a different function. The oracle is never shown
to the enumeration, which is what keeps this a check rather than a lookup; it is shown to the
falsifier afterwards, which is the role ``redteam`` was written for and the same discipline
``NoesisEngine`` already applies before trusting a solution into its corpus.

**Bounded, and it declines rather than searches harder.** The enumeration fills each hole from a
small set of shapes — the task input, small literals, and single applications of cheap primitives
— with a hard cap on candidates per abstraction. An abstraction that needs a deep argument to
transfer is one this check cannot confirm, and it is reported as not transferred rather than
chased: the number has to mean "verified", and a search that always eventually succeeds would
measure the search instead of the skill.

Pure standard library; imports only the program representation it verifies against. Fail-soft:
any error yields no transfers, never an exception into the SLEEP pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Transferred", "ZeroShotTransfer"]


def _default_red_team() -> Any:
    """The repo's own falsifier, or ``None`` — in which case nothing is ever confirmed."""
    try:
        from nyxara.growth.redteam import RedTeam
        return RedTeam()
    except Exception:  # noqa: BLE001
        return None

#: How many candidate argument fillings may be tried per (abstraction, task) pair. Small on
#: purpose — see the module docstring: this measures the skill, not the search.
_MAX_CANDIDATES = 256

#: Literals offered as hole fillings. The set a hand-written solution would reach for first.
_SMALL_INTS: Tuple[int, ...] = (0, 1, 2, 3, -1)


@dataclass
class Transferred:
    """One abstraction verified on one task it was not induced from."""

    abstraction: str = ""
    task: str = ""
    #: The program that reproduced the task's examples, for the record. A claim of transfer that
    #: cannot show the program is a claim nobody can check.
    program: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"abstraction": self.abstraction, "task": self.task, "program": self.program}


class ZeroShotTransfer:
    """Verifies adopted abstractions against tasks they were not extracted from.

    Duck-typed to the single method :class:`~nyxara.growth.noesis.NoesisEngine` calls, so it can
    be handed in without the engine knowing anything about it.
    """

    def __init__(self, *, max_candidates: int = _MAX_CANDIDATES,
                 red_team: Any = None) -> None:
        self.max_candidates = max(8, int(max_candidates))
        self.red_team = red_team if red_team is not None else _default_red_team()
        self.checked = 0
        #: Candidates that reproduced every example and were then falsified on adversarial
        #: inputs. Reported rather than hidden: it is the count of times example-fitting would
        #: have produced a false claim of transfer.
        self.refuted = 0
        self.transfers: List[Transferred] = []
        #: (abstraction, task) pairs already confirmed, so a skill is not re-counted every cycle
        #: it survives. Transfer is a thing that happened once, not a level.
        self._seen: Set[Tuple[str, str]] = set()
        #: Abstractions adopted but not yet verified — held until tasks arrive that they were
        #: demonstrably not built from. See :meth:`transfer_all`.
        self._pending: Dict[str, Any] = {}

    # ---- the interface NoesisEngine calls ---------------------------------- #
    def transfer_all(self, library: Any, adopted: Sequence[Any],
                     tasks: Sequence[Any]) -> List[Transferred]:
        """Verify abstractions from **earlier** cycles against tasks that arrived after them.

        The held-out discipline is in the timing, and it has to be. ``transfer_all`` is called at
        the end of SLEEP with the abstractions just adopted and the tasks just solved — and the
        adoption was *extracted from those solutions*. Checking them against each other would be
        the induction restated, which is exactly the number this module refuses to produce.

        So an abstraction adopted now is **held** and verified on the next cycle's tasks, which it
        has never seen and which were generated without reference to it. That is what makes a
        confirmation here mean "this skill applies somewhere it was not built for" rather than
        "this skill fits the data it came from".
        """
        out: List[Transferred] = []
        try:
            # Verify what is already pending against tasks that are new to it.
            for name, abstraction in list(self._pending.items()):
                for task in tasks or ():
                    task_name = str(getattr(task, "name", "") or "")
                    if not task_name or (name, task_name) in self._seen:
                        continue
                    self.checked += 1
                    program = self._solves(library, abstraction, task)
                    if program is None:
                        continue
                    got = Transferred(abstraction=name, task=task_name, program=program)
                    self._seen.add((name, task_name))
                    self.transfers.append(got)
                    out.append(got)
            # Then hold this cycle's adoptions for the next one.
            for abstraction in adopted or ():
                name = str(getattr(abstraction, "name", "") or "")
                if name:
                    self._pending[name] = abstraction
        except Exception:  # noqa: BLE001 — a failed check transfers nothing, never breaks SLEEP
            return out
        return out

    # ---- verification ------------------------------------------------------- #
    def _solves(self, library: Any, abstraction: Any, task: Any) -> Optional[str]:
        """A program built on this abstraction that reproduces every example, or ``None``."""
        from nyxara.growth.noesis import Prog, evaluate

        name = str(getattr(abstraction, "name", ""))
        arg_types = tuple(getattr(abstraction, "arg_types", ()) or ())
        ret_type = getattr(abstraction, "ret_type", None)
        # Compared by value: `Type` here is a *name* ("int", "intlist"), not a Python type, so
        # identity happens to work for interned short strings and silently stops working for
        # anything else. Measured, `is` rejected every candidate before a single one was tried.
        if ret_type != getattr(task, "ret_type", None):
            return None
        examples = tuple(getattr(task, "examples", ()) or ())
        if not examples:
            return None
        input_type = getattr(task, "input_type", None)

        tried = 0
        for args in self._fillings(arg_types, input_type, Prog):
            tried += 1
            if tried > self.max_candidates:
                return None
            candidate = Prog(kind="app", rtype=ret_type, name=name, args=tuple(args))
            if not self._reproduces(candidate, examples, library, evaluate):
                continue
            if not self._survives(candidate, task, library):
                self.refuted += 1
                continue
            return candidate.pretty() if hasattr(candidate, "pretty") else name
        return None

    def _survives(self, candidate: Any, task: Any, library: Any) -> bool:
        """Adversarial inputs against the task's oracle. No oracle, no claim.

        Refusing to confirm a transfer that cannot be falsified is the conservative direction and
        the right one: an unfalsifiable confirmation is what makes a metric drift.
        """
        oracle = getattr(task, "oracle", None)
        if self.red_team is None or oracle is None:
            return False
        try:
            verdict = self.red_team.survives(candidate, getattr(task, "input_type", None),
                                             library, oracle=oracle)
            return bool(getattr(verdict, "survived", False))
        except Exception:  # noqa: BLE001 — a battery that cannot run has falsified nothing
            return False

    @staticmethod
    def _reproduces(candidate: Any, examples: Sequence[Tuple[Any, Any]],
                    library: Any, evaluate: Any) -> bool:
        """Every example, exactly. The task's oracle is never consulted — see the docstring."""
        for value, expected in examples:
            try:
                if evaluate(candidate, value, library) != expected:
                    return False
            except Exception:  # noqa: BLE001 — a program that raises has not solved anything
                return False
        return True

    def _fillings(self, arg_types: Sequence[Any], input_type: Any,
                  prog_cls: Any) -> Iterable[List[Any]]:
        """Candidate argument programs per hole, cheapest shapes first.

        Deliberately shallow: the task's own input, and small literals. Anything deeper is a
        search for a solution rather than a check that a skill applies, and the distinction is
        the whole reason this number is worth reporting.
        """
        per_hole: List[List[Any]] = []
        for wanted in arg_types:
            options: List[Any] = []
            if wanted == input_type:
                options.append(prog_cls(kind="var", rtype=wanted, name="x"))
            if wanted == "int":
                options.extend(prog_cls(kind="lit", rtype="int", value=v) for v in _SMALL_INTS)
            elif wanted == "bool":
                options.extend(prog_cls(kind="lit", rtype="bool", value=v)
                               for v in (True, False))
            if not options:
                return
            per_hole.append(options)
        if not per_hole:
            yield []
            return
        yield from self._product(per_hole)

    @staticmethod
    def _product(per_hole: Sequence[Sequence[Any]]) -> Iterable[List[Any]]:
        out: List[List[Any]] = [[]]
        for options in per_hole:
            out = [row + [option] for row in out for option in options]
        yield from out

    # ---- reporting ---------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        by_abstraction: Dict[str, int] = {}
        for row in self.transfers:
            by_abstraction[row.abstraction] = by_abstraction.get(row.abstraction, 0) + 1
        return {
            "checked": self.checked,
            "refuted_by_red_team": self.refuted,
            "transferred": len(self.transfers),
            "abstractions_that_transferred": len(by_abstraction),
            "by_abstraction": by_abstraction,
            "examples": [t.to_dict() for t in self.transfers[:4]],
        }

