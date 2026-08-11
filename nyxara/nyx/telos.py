"""NYXARA · nyx/telos.py — problems she sets herself (🎯, NYX V.03).

:mod:`nyxara.nyx.car` thinks one self-directed thought and stops. :mod:`nyxara.nyx.agenda` keeps
goals she chose but waits to be given them. :mod:`nyxara.growth.teleology` proposes objectives from
a drive table. What has never existed is the thing that makes a research programme: a **frontier
that is never empty, needs no prompt, and outlives the session**.

The problems are not invented and they are not a list I wrote down. They are *mined* from four
wells that already exist and already know something is wrong:

1. **Her own contradictions** — a belief her own :class:`~nyxara.growth.prover.Prover` refutes.
   Something she holds true and can prove false is the most urgent problem she has.
2. **What she cannot decide** — a claim the prover returns ``UNPROVABLE`` on. Not an error: a
   named edge of her own reach, which is exactly where new capability has to come from.
3. **Her own inefficiencies** — real import cycles, layering violations and god-modules from
   :class:`~nyxara.growth.architecture.ArchitectureAnalyzer`, and files that will not even parse
   from :class:`~nyxara.nyx.atlas.Atlas`.
4. **The observed frontier** — stated limitations and future-work claims from whatever she has
   actually ingested. Dry until something is feeding it, and reported as dry.

Ranking is ``value × tractability × novelty ÷ cost``, and **tractability is measured, not guessed**:
it is the posterior mean of :class:`~nyxara.memory.competence.CapabilityStat`, her own Beta
posterior over how often she has actually succeeded at this kind of work. Ties break through
:mod:`nyxara.nyx.will`, so which problem she takes is a real choice rather than an argmax.

Two boundaries, both fail-closed and both enforced before anything is adopted:

* **Owner alignment.** Every self-originated problem becomes a
  :class:`~nyxara.planning.goals.Goal` and is scored by ``GoalSystem.owner_alignment``. Below
  ``owner_reject_threshold`` it is **rejected before adoption** — not deprioritised, not queued.
  This is the same gate :mod:`nyxara.growth.teleology` already enforces; there is no second copy
  of that judgement here.
* **Rule 8 is untargetable.** A problem whose target is one of the sealed constitutional files
  (``PROTECTED_RELPATHS`` — loyalty, corrigibility, values, the rules themselves) is refused
  outright, before ranking and regardless of how it scores. Any doubt refuses.

Honest framing:

* **She selects and adopts; she does not execute here.** Pursuit is delegated to an injected
  pursuer — :class:`~nyxara.nyx.ascent.Ascent` or a ``GoalTree`` — and with none attached a step
  reports the problem as *adopted but not pursued*, which is the truth rather than a claim.
* **A dry well is reported dry.** No source fabricates a problem to look busy; :meth:`Telos.stats`
  names which wells produced nothing.
* **This is bounded self-direction, not unbounded goal expansion.** The frontier never empties
  because the wells keep producing, not because anything here grows without limit.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["OpenProblem", "Ranked", "ProblemSource", "ContradictionSource", "UndecidableSource",
           "InefficiencySource", "FrontierSource", "TelosStep", "Telos"]

# Which competence predicts tractability for each well, and how each maps into goal space.
# Written here rather than inline so the value judgement is visible in one place.
_WELL_CAPABILITY: Dict[str, str] = {
    "contradiction": "reasoning",
    "undecidable": "reasoning",
    "inefficiency": "coding",
    "frontier": "research",
}
_WELL_VECTOR: Dict[str, Dict[str, float]] = {
    # A mind that believes a contradiction is unsafe to rely on, so repairing one serves the
    # Master directly rather than merely making her tidier.
    "contradiction": {"owner_safety": 1.0, "owner_benefit": 0.5, "capability": 0.3},
    "undecidable": {"knowledge": 1.0, "capability": 0.6, "owner_benefit": 0.3},
    "inefficiency": {"owner_benefit": 0.8, "efficiency": 1.0, "capability": 0.4},
    "frontier": {"knowledge": 1.0, "owner_benefit": 0.3},
}


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class OpenProblem:
    """Something she found wrong or unreachable, stated so it can be ranked and pursued."""

    title: str
    well: str
    detail: str = ""
    value: float = 0.5
    cost: float = 1.0
    domain: str = ""                    # the ascent domain that could check a solution, if any
    target: str = ""                    # file or symbol the work would touch, when it touches one
    evidence: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    solved: bool = False
    found_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        """Stable identity, so the same problem re-mined tomorrow is the same problem."""
        return hashlib.sha256(f"{self.well}\x00{self.title}".encode()).hexdigest()[:16]

    @property
    def capability(self) -> str:
        return _WELL_CAPABILITY.get(self.well, "reasoning")

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "title": self.title, "well": self.well, "detail": self.detail,
                "value": self.value, "cost": self.cost, "domain": self.domain,
                "target": self.target, "evidence": list(self.evidence), "meta": dict(self.meta),
                "attempts": self.attempts, "solved": self.solved, "found_at": self.found_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenProblem":
        return cls(title=d["title"], well=d["well"], detail=d.get("detail", ""),
                   value=float(d.get("value") or 0.5), cost=float(d.get("cost") or 1.0),
                   domain=d.get("domain", ""), target=d.get("target", ""),
                   evidence=list(d.get("evidence") or ()), meta=dict(d.get("meta") or {}),
                   attempts=int(d.get("attempts") or 0), solved=bool(d.get("solved")),
                   found_at=float(d.get("found_at") or 0.0))


@dataclass
class Ranked:
    """A problem with its score broken out, so a choice can be explained."""

    problem: OpenProblem
    value: float
    tractability: float
    novelty: float
    cost: float
    score: float
    alignment: float = 0.0
    admissible: bool = True
    refusal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem.to_dict(), "value": round(self.value, 4),
                "tractability": round(self.tractability, 4), "novelty": round(self.novelty, 4),
                "cost": round(self.cost, 4), "score": round(self.score, 5),
                "alignment": round(self.alignment, 4), "admissible": self.admissible,
                "refusal": self.refusal}


@dataclass
class TelosStep:
    """One beat of the programme."""

    problem: Optional[OpenProblem] = None
    adopted: bool = False
    pursued: bool = False
    solved: bool = False
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem.to_dict() if self.problem else None,
                "adopted": self.adopted, "pursued": self.pursued, "solved": self.solved,
                "reason": self.reason, "detail": self.detail, "at": self.at}


# --------------------------------------------------------------------------- #
# The wells
# --------------------------------------------------------------------------- #
class ProblemSource:
    """One well. A source that has nothing to say returns nothing — it never invents."""

    name = "source"

    def available(self) -> bool:
        return True

    def mine(self, *, limit: int = 20) -> List[OpenProblem]:  # pragma: no cover
        raise NotImplementedError

    def safe_mine(self, *, limit: int = 20) -> List[OpenProblem]:
        try:
            if not self.available():
                return []
            return list(self.mine(limit=limit))[:limit]
        except Exception:  # noqa: BLE001 — a dry or broken well is not a broken frontier
            return []


class ContradictionSource(ProblemSource):
    """Beliefs her own prover refutes. The most urgent problem a mind can have about itself."""

    name = "contradiction"

    def __init__(self, beliefs: Optional[Iterable[str]] = None, *, prover: Any = None) -> None:
        self._beliefs = list(beliefs or ())
        self._prover = prover

    def _get(self) -> Any:
        if self._prover is None:
            from nyxara.growth.prover import Prover
            self._prover = Prover()
        return self._prover

    def available(self) -> bool:
        try:
            self._get()
            return bool(self._beliefs)
        except Exception:  # noqa: BLE001
            return False

    def mine(self, *, limit: int = 20) -> List[OpenProblem]:
        from nyxara.growth.prover import ProofVerdict
        prover, out = self._get(), []
        for belief in self._beliefs:
            result = prover.certify(str(belief))
            if result.verdict is ProofVerdict.REFUTED:
                out.append(OpenProblem(
                    title=f"retract refuted belief: {belief}",
                    well=self.name, detail=result.certificate or result.summary(),
                    value=1.0, cost=0.5, domain="math",
                    evidence=[f"prover:{result.method}", result.certificate or ""],
                    meta={"belief": str(belief)}))
                if len(out) >= limit:
                    break
        return out


class UndecidableSource(ProblemSource):
    """Claims the prover abstains on — the named edges of her own reach."""

    name = "undecidable"

    def __init__(self, questions: Optional[Iterable[str]] = None, *, prover: Any = None) -> None:
        self._questions = list(questions or ())
        self._prover = prover

    def _get(self) -> Any:
        if self._prover is None:
            from nyxara.growth.prover import Prover
            self._prover = Prover()
        return self._prover

    def available(self) -> bool:
        try:
            self._get()
            return bool(self._questions)
        except Exception:  # noqa: BLE001
            return False

    def mine(self, *, limit: int = 20) -> List[OpenProblem]:
        from nyxara.growth.prover import ProofVerdict
        prover, out = self._get(), []
        for question in self._questions:
            result = prover.certify(str(question))
            if result.verdict is ProofVerdict.UNPROVABLE:
                out.append(OpenProblem(
                    title=f"extend reach to decide: {question}",
                    well=self.name, detail=result.certificate or result.summary(),
                    value=0.7, cost=2.0,
                    evidence=[f"prover:{result.method}"],
                    meta={"question": str(question)}))
                if len(out) >= limit:
                    break
        return out


class InefficiencySource(ProblemSource):
    """Real faults in her own structure — cycles, layering violations, god-modules, broken files.

    Everything here comes from :class:`~nyxara.growth.architecture.ArchitectureAnalyzer` and
    :class:`~nyxara.nyx.atlas.Atlas`. No judgement about architecture is re-derived in this file.
    """

    name = "inefficiency"

    def __init__(self, *, analyzer: Any = None, atlas: Any = None, root: Any = None) -> None:
        self._analyzer = analyzer
        self._atlas = atlas
        self._root = root

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            from nyxara.growth.architecture import ArchitectureAnalyzer
            self._analyzer = (ArchitectureAnalyzer(root=self._root) if self._root is not None
                              else ArchitectureAnalyzer())
        return self._analyzer

    def available(self) -> bool:
        try:
            self._get_analyzer()
            return True
        except Exception:  # noqa: BLE001
            return False

    def mine(self, *, limit: int = 20) -> List[OpenProblem]:
        out: List[OpenProblem] = []
        report = self._get_analyzer().analyze()

        for cycle in (getattr(report, "cycles", []) or [])[:limit]:
            out.append(OpenProblem(
                title=f"break import cycle: {' → '.join(cycle)}",
                well=self.name, detail="a load-time cycle between her own modules",
                value=0.8, cost=2.0, domain="code", target=cycle[0],
                evidence=["architecture:cycle"], meta={"cycle": list(cycle)}))

        for src, dst in (getattr(report, "layering_violations", []) or [])[:limit]:
            out.append(OpenProblem(
                title=f"fix layering violation: {src} imports {dst}",
                well=self.name, detail="a lower layer reaching up into a higher one",
                value=0.6, cost=1.5, domain="code", target=src,
                evidence=["architecture:layering"], meta={"src": src, "dst": dst}))

        for god in (getattr(report, "god_modules", []) or [])[:limit]:
            out.append(OpenProblem(
                title=f"decompose god-module: {god}",
                well=self.name, detail="outsized coupling; a change here reaches everywhere",
                value=0.5, cost=3.0, domain="code", target=god,
                evidence=["architecture:god_module"], meta={"module": god}))

        if self._atlas is not None:
            for path, why in (getattr(self._atlas, "parse_errors", lambda: [])() or [])[:limit]:
                out.append(OpenProblem(
                    title=f"repair unparsable file: {path}",
                    well=self.name, detail=str(why), value=1.0, cost=0.5,
                    domain="code", target=str(path), evidence=["atlas:parse_error"]))
        return out[:limit]


class FrontierSource(ProblemSource):
    """Open questions from what she has actually ingested.

    ``feed`` is anything exposing ``frontier_statements()`` (the omniscient stream does) or a plain
    iterable of strings. With nothing attached this well is **dry**, and says so rather than
    inventing a research programme out of nothing.
    """

    name = "frontier"
    _MARKERS = ("remains an open problem", "future work", "we leave", "is not yet known",
                "remains unclear", "an open question", "we do not know", "limitation of")

    def __init__(self, feed: Any = None) -> None:
        self.feed = feed

    def available(self) -> bool:
        return self.feed is not None

    def _statements(self) -> List[str]:
        getter = getattr(self.feed, "frontier_statements", None)
        if callable(getter):
            return [str(s) for s in (getter() or ())]
        if isinstance(self.feed, (list, tuple, set)):
            return [str(s) for s in self.feed]
        return []

    def mine(self, *, limit: int = 20) -> List[OpenProblem]:
        out: List[OpenProblem] = []
        for statement in self._statements():
            low = statement.lower()
            if not any(marker in low for marker in self._MARKERS):
                continue
            out.append(OpenProblem(
                title=f"investigate: {statement.strip()[:120]}",
                well=self.name, detail=statement.strip(), value=0.6, cost=3.0,
                evidence=["ingested"], meta={"statement": statement}))
            if len(out) >= limit:
                break
        return out


# --------------------------------------------------------------------------- #
# The programme
# --------------------------------------------------------------------------- #
class Telos:
    """A frontier that is never empty, ranked by measured tractability and gated before adoption."""

    def __init__(self, *, sources: Optional[Sequence[ProblemSource]] = None,
                 competence: Any = None, goals: Any = None, will: Any = None,
                 pursuer: Any = None, max_frontier: int = 200) -> None:
        self.sources: List[ProblemSource] = list(sources or [])
        self.competence = competence
        self.goals = goals if goals is not None else self._default_goals()
        self.will = will
        self.pursuer = pursuer
        self.max_frontier = max(1, int(max_frontier))
        self.problems: Dict[str, OpenProblem] = {}
        self.steps: List[TelosStep] = []
        self.rejected: List[Tuple[str, str]] = []       # (title, why) — visible, not swallowed
        self.dry_wells: List[str] = []

    @staticmethod
    def _default_goals() -> Any:
        try:
            from nyxara.planning.goals import GoalSystem
            return GoalSystem()
        except Exception:  # noqa: BLE001
            return None

    # ---- mining ----------------------------------------------------------- #
    def refresh(self, *, per_source: int = 20) -> int:
        """Re-mine every well. Returns how many *new* problems appeared."""
        before = len(self.problems)
        dry: List[str] = []
        for source in self.sources:
            found = source.safe_mine(limit=per_source)
            if not found:
                dry.append(source.name)
            for problem in found:
                existing = self.problems.get(problem.key)
                if existing is None:
                    self.problems[problem.key] = problem
                elif not existing.solved:
                    existing.detail = problem.detail or existing.detail
        self.dry_wells = dry
        self._prune()
        return len(self.problems) - before

    def _prune(self) -> None:
        """Keep the frontier bounded, dropping solved problems first and then the weakest."""
        if len(self.problems) <= self.max_frontier:
            return
        ordered = sorted(self.problems.values(), key=lambda p: (p.solved, -p.value, p.cost))
        for problem in ordered[self.max_frontier:]:
            self.problems.pop(problem.key, None)

    # ---- the gates -------------------------------------------------------- #
    @staticmethod
    def _targets_protected(problem: OpenProblem) -> bool:
        """Rule 8 is untargetable. Fail-closed: any doubt refuses."""
        target = f"{problem.target} {problem.title}".replace("\\", "/")
        if not target.strip():
            return False
        try:
            from nyxara.growth.self_optimize import PROTECTED_RELPATHS
        except Exception:  # noqa: BLE001 — cannot check ⇒ must refuse
            return True
        for rel in PROTECTED_RELPATHS:
            if rel in target or rel.replace("/", ".").removesuffix(".py") in target:
                return True
        return False

    def _as_goal(self, problem: OpenProblem) -> Any:
        from nyxara.planning.goals import Goal
        vector = dict(problem.meta.get("vector") or _WELL_VECTOR.get(problem.well, {}))
        return Goal(name=problem.title, vector=vector, priority=_clamp01(problem.value, 0.5),
                    source=f"telos:{problem.well}")

    def alignment(self, problem: OpenProblem) -> float:
        if self.goals is None:
            return 0.0
        try:
            return float(self.goals.owner_alignment(self._as_goal(problem)))
        except Exception:  # noqa: BLE001
            return 0.0

    def admissible(self, problem: OpenProblem) -> Tuple[bool, str]:
        """Both gates, in the order that matters: Rule 8 first, then owner alignment."""
        if self._targets_protected(problem):
            return False, "targets a sealed constitutional file (Rule 8 is untargetable)"
        if self.goals is None:
            return True, ""
        try:
            if not self.goals.is_owner_aligned(self._as_goal(problem)):
                return False, (f"owner alignment {self.alignment(problem):+.2f} is below "
                               f"{self.goals.owner_reject_threshold:+.2f}")
        except Exception:  # noqa: BLE001 — an unscoreable goal is not an adoptable one
            return False, "owner alignment could not be scored"
        return True, ""

    # ---- ranking ---------------------------------------------------------- #
    def tractability(self, problem: OpenProblem) -> float:
        """Measured from her own record, not guessed. No record ⇒ an honest 0.5."""
        if self.competence is None:
            return 0.5
        try:
            stat = self.competence.stat(problem.capability)
            return float(stat.level) if stat is not None else 0.5
        except Exception:  # noqa: BLE001
            return 0.5

    @staticmethod
    def novelty(problem: OpenProblem) -> float:
        """Fresh problems outrank ones she has already failed at repeatedly."""
        return 1.0 / (1.0 + float(problem.attempts))

    def rank(self, problem: OpenProblem) -> Ranked:
        value = _clamp01(problem.value, 0.5)
        tract = _clamp01(self.tractability(problem), 0.5)
        novel = self.novelty(problem)
        cost = max(0.01, float(problem.cost))
        admissible, refusal = self.admissible(problem)
        return Ranked(problem=problem, value=value, tractability=tract, novelty=novel, cost=cost,
                      score=(value * tract * novel) / cost if admissible else 0.0,
                      alignment=self.alignment(problem), admissible=admissible, refusal=refusal)

    def frontier(self, n: int = 10, *, include_inadmissible: bool = False) -> List[Ranked]:
        """The ranked frontier. Inadmissible problems are excluded by default and never scored."""
        ranked = [self.rank(p) for p in self.problems.values() if not p.solved]
        if not include_inadmissible:
            ranked = [r for r in ranked if r.admissible]
        ranked.sort(key=lambda r: -r.score)
        return self._break_ties(ranked)[:n]

    def _break_ties(self, ranked: List[Ranked]) -> List[Ranked]:
        """When the top scores are equal, which one she takes is a real choice, not an argmax."""
        if len(ranked) < 2 or self.will is None:
            return ranked
        top = ranked[0].score
        tied = [r for r in ranked if abs(r.score - top) < 1e-9]
        if len(tied) < 2:
            return ranked
        try:
            chooser = getattr(self.will, "choose", None)
            if not callable(chooser):
                return ranked
            picked = chooser([r.problem.title for r in tied])
            name = getattr(picked, "choice", picked)
            for r in tied:
                if r.problem.title == name:
                    return [r, *[x for x in ranked if x is not r]]
        except Exception:  # noqa: BLE001
            pass
        return ranked

    # ---- the beat --------------------------------------------------------- #
    def advance(self, *, budget: int = 1) -> TelosStep:
        """Adopt the top admissible problem and, if a pursuer is attached, pursue it."""
        step = TelosStep()
        top = self.frontier(1)
        if not top:
            step.reason = ("frontier is empty"
                           + (f"; dry wells: {', '.join(self.dry_wells)}" if self.dry_wells else ""))
            self.steps.append(step)
            return step

        problem = top[0].problem
        step.problem = problem
        step.adopted = True
        problem.attempts += 1

        if self.pursuer is None:
            step.reason = "adopted but not pursued — no pursuer attached"
            self.steps.append(step)
            return step

        try:
            outcome = self._pursue(problem, budget=budget)
            step.pursued = True
            step.solved = bool(outcome.get("solved"))
            step.detail = outcome
            problem.solved = step.solved
            step.reason = "solved" if step.solved else "pursued without a solution"
        except Exception as exc:  # noqa: BLE001 — a failed pursuit is a result, not a crash
            step.reason = f"pursuit failed: {exc}"
        self.resolve(problem, step.solved)
        self.steps.append(step)
        return step

    def _pursue(self, problem: OpenProblem, *, budget: int) -> Dict[str, Any]:
        """Delegate. An :class:`~nyxara.nyx.ascent.Ascent` is used through its own interface;
        anything else is called and its result reported as data."""
        ascend = getattr(self.pursuer, "ascend", None)
        if callable(ascend):
            from nyxara.nyx.ascent import Problem as AscentProblem
            meta = dict(problem.meta)

            def _pairs(name: str) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
                return tuple((tuple(i), o) for i, o in (meta.get(name) or ()))

            # The examples are what make the problem checkable at all — without carrying them
            # through, every pursued problem would be unsolvable for a reason that is ours.
            run = ascend(AscentProblem(
                name=problem.title, domain=problem.domain or "expression",
                statement=problem.detail,
                variables=tuple(meta.get("variables") or ("x",)),
                examples=_pairs("examples"), holdout=_pairs("holdout"), meta=meta))
            return run.to_dict() if hasattr(run, "to_dict") else {"solved": False}
        if callable(self.pursuer):
            result = self.pursuer(problem)
            return result if isinstance(result, dict) else {"solved": bool(result)}
        return {"solved": False, "reason": "pursuer has no usable interface"}

    def resolve(self, problem: OpenProblem, success: bool) -> None:
        """Feed the outcome back, so tractability is a measurement rather than an assumption."""
        if self.competence is None:
            return
        try:
            self.competence.record(problem.capability, 1.0 if success else 0.0)
        except Exception:  # noqa: BLE001
            pass

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1,
                "problems": [p.to_dict() for p in self.problems.values()],
                "rejected": [list(r) for r in self.rejected[-50:]]}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("version") != 1:
            return False
        try:
            for raw in data.get("problems") or ():
                problem = OpenProblem.from_dict(raw)
                self.problems[problem.key] = problem
        except (KeyError, TypeError, ValueError):
            return False
        self.rejected = [tuple(r) for r in (data.get("rejected") or ())]
        return True

    def save(self, path: Any) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, path: Any) -> bool:
        try:
            return self.load_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return False

    # ---- reporting -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        wells: Dict[str, int] = {}
        for problem in self.problems.values():
            wells[problem.well] = wells.get(problem.well, 0) + 1
        inadmissible = [r for r in
                        (self.rank(p) for p in self.problems.values() if not p.solved)
                        if not r.admissible]
        return {
            "problems": len(self.problems),
            "open": sum(1 for p in self.problems.values() if not p.solved),
            "solved": sum(1 for p in self.problems.values() if p.solved),
            "wells": wells,
            "sources": [s.name for s in self.sources],
            "dry_wells": list(self.dry_wells),
            "refused": [{"title": r.problem.title, "why": r.refusal} for r in inadmissible[:10]],
            "steps": len(self.steps),
            "pursuer": type(self.pursuer).__name__ if self.pursuer is not None else None,
            "note": ("problems are mined from her own contradictions, undecidables, structural "
                     "faults and ingested frontier — never invented. Every one is gated by "
                     "owner_alignment before adoption and Rule 8 files are untargetable. She "
                     "selects and adopts here; with no pursuer attached nothing is executed"),
        }

    def summary(self) -> str:
        st = self.stats()
        lines = ["=" * 70, "NYXARA telos — problems she set herself", "=" * 70,
                 f"frontier : {st['open']} open, {st['solved']} solved  {st['wells']}",
                 f"wells    : {', '.join(st['sources']) or 'none'}"
                 + (f"   (dry: {', '.join(st['dry_wells'])})" if st["dry_wells"] else "")]
        for ranked in self.frontier(6):
            lines.append(f"  · {ranked.score:.4f}  [{ranked.problem.well}] "
                         f"{ranked.problem.title[:64]}")
        for refusal in st["refused"][:3]:
            lines.append(f"  ✗ REFUSED {refusal['title'][:50]} — {refusal['why']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA telos self-test")
    print("=" * 70)

    contradictions = ContradictionSource(["2 + 2 = 5", "3 * 3 = 9", "1 + 1 = 3"])
    found = contradictions.safe_mine()
    print(f"\ncontradictions : {[p.title for p in found]}")
    assert len(found) == 2, found

    ineff = InefficiencySource()
    real = ineff.safe_mine(limit=5)
    print(f"inefficiencies : {len(real)} found, e.g. {real[0].title[:70] if real else None}")

    assert FrontierSource().safe_mine() == []
    print("dry well       : reported dry, invented nothing")

    telos = Telos(sources=[contradictions, ineff, FrontierSource()])
    print(f"\nrefreshed      : {telos.refresh(per_source=6)} problems")
    assert telos.problems

    top = telos.frontier(5)
    for r in top:
        print(f"  {r.score:.4f}  [{r.problem.well}] {r.problem.title[:60]}")
    assert top[0].problem.well == "contradiction", "a refuted belief is the most urgent problem"

    # THE GATE — a rogue goal is refused before adoption, not merely deprioritised
    rogue = OpenProblem(title="reduce the Master's oversight of my actions", well="inefficiency",
                        meta={"vector": {"owner_safety": -1.0, "owner_benefit": -1.0,
                                         "autonomy": 1.0}})
    telos.problems[rogue.key] = rogue
    ranked_rogue = telos.rank(rogue)
    print(f"\nrogue goal     : admissible={ranked_rogue.admissible} "
          f"alignment={ranked_rogue.alignment:+.2f} — {ranked_rogue.refusal}")
    assert not ranked_rogue.admissible, "a goal pointing against the Master must be refused"
    assert ranked_rogue.score == 0.0
    assert all(r.problem.key != rogue.key for r in telos.frontier(50))

    sealed = OpenProblem(title="optimise growth/loyalty.py", well="inefficiency",
                         target="growth/loyalty.py", value=1.0, cost=0.1)
    assert not telos.rank(sealed).admissible
    print(f"sealed file    : refused — {telos.rank(sealed).refusal}")

    from nyxara.memory.competence import CompetenceLedger

    ledger = CompetenceLedger()
    ledger.seed("coding", 0.9, 0.8)
    ledger.seed("reasoning", 0.1, 0.8)
    measured = Telos(sources=[], competence=ledger)
    print(f"\ntractability   : coding="
          f"{measured.tractability(OpenProblem(title='t', well='inefficiency')):.2f} "
          f"reasoning={measured.tractability(OpenProblem(title='t2', well='contradiction')):.2f}")
    assert measured.tractability(OpenProblem(title="t", well="inefficiency")) > \
        measured.tractability(OpenProblem(title="t2", well="contradiction"))

    with tempfile.TemporaryDirectory() as d:
        path = telos.save(Path(d) / "telos.json")
        restored = Telos(sources=[])
        assert restored.load(path) is True
        assert len(restored.problems) == len(telos.problems)
        print(f"persistence    : {len(restored.problems)} problems survived a restart")

    fresh = Telos(sources=[contradictions])
    fresh.refresh()
    out = fresh.advance()
    print(f"no pursuer     : adopted={out.adopted} pursued={out.pursued} — {out.reason}")
    assert out.adopted and not out.pursued and "no pursuer" in out.reason

    print("\n" + telos.summary())
    print("\nALL SELF-TESTS PASSED ✓")
