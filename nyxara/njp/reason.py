"""NYXARA · njp/reason.py — deliberate reasoning, at a depth the question deserves (🜂, NJP V.02).

Associative recall answers instantly and is right about familiar things. It is also the only mode
NJP had, which meant every question got the same treatment: the fabric settled, something came
back, and whether that was a reflex or a conclusion was indistinguishable from outside.

This adds the deliberate half, and — just as importantly — the **routing** between them.

**The ladder.** Five rungs of increasing cost::

    intuition → association → deliberation → verification → proof/simulation

She starts at the top and descends only while the answer is not yet good enough. Spending maximum
compute on every question is its own kind of uselessness: a mind that runs a proof engine over
"what is my name" is not being careful, it is being broken. Equally, one that answers a
load-bearing question from a reflex is being reckless. So the descent is driven by two measured
things — how uncertain she is, and what it costs to be wrong — and every stop records the rung it
reached, so "she thought hard about this" is a checkable claim.

**The problem state.** A :class:`ProblemState` survives across turns and holds what a multi-step
problem actually needs: known facts, *unknown* facts, assumptions, live hypotheses, intermediate
conclusions, **rejected** hypotheses and what is still pending verification. Keeping the rejected
ones matters — without them she re-proposes the same dead end every time she revisits the problem.

**The debate.** Hard questions do not get one answer generated and defended. They get competing
hypotheses, each scored on evidence, contradiction and simulation, and no answer is emitted until
one is enough clear of the others. A tie is reported as a tie. That is the mechanism behind
"I don't know" being a real output rather than a failure mode.

Nothing here invents its own verification. Contradiction checks go to the caller's belief store,
proofs to :class:`~nyxara.njp.prove.ProofCore`, simulation to
:class:`~nyxara.njp.world.WorldView`, and the final verdict to
:class:`~nyxara.njp.truth.TruthGauntlet`. This module decides *how deep to go* and *how to weigh
what comes back*; it does not re-derive any of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Rung", "Hypothesis", "ProblemState", "Conclusion", "Reasoner"]


class Rung:
    """How deep she went. Ordered — index is cost, and the order is the whole design."""

    INTUITION = "intuition"            # the manifold's pre-cognitive read; no settling
    ASSOCIATION = "association"        # recall + the fabric; the old default
    DELIBERATION = "deliberation"      # competing hypotheses, weighed
    VERIFICATION = "verification"      # contradiction checks against what she believes
    PROOF = "proof"                    # formal proof, or simulation over the world model

    ALL = (INTUITION, ASSOCIATION, DELIBERATION, VERIFICATION, PROOF)

    @staticmethod
    def cost(rung: str) -> int:
        return Rung.ALL.index(rung) if rung in Rung.ALL else 0


@dataclass
class Hypothesis:
    """One candidate answer, with everything that argued for or against it."""

    claim: str = ""
    support: float = 0.0               # evidence for, 0…1
    against: float = 0.0               # evidence against, 0…1
    source: str = ""
    evidence: List[str] = field(default_factory=list)
    contradicted: bool = False
    proved: bool = False
    refuted: bool = False

    @property
    def score(self) -> float:
        """Net standing. A refuted hypothesis is zero however much supported it was."""
        if self.refuted:
            return 0.0
        if self.proved:
            return 1.0
        return max(0.0, min(1.0, self.support - self.against))

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:200], "score": round(self.score, 4),
                "support": round(self.support, 4), "against": round(self.against, 4),
                "source": self.source, "proved": self.proved, "refuted": self.refuted,
                "contradicted": self.contradicted, "evidence": self.evidence[:6]}


@dataclass
class ProblemState:
    """Working memory for one problem, persisting across the turns it takes to solve.

    The **unknowns** and the **rejected** lists are the two that are usually missing and that do
    most of the work. Unknowns are what turns a stuck problem into a question worth asking;
    rejected hypotheses are what stops her walking into the same dead end on every revisit.
    """

    problem: str = ""
    known: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    intermediate: List[str] = field(default_factory=list)
    rejected: List[Hypothesis] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    opened: float = field(default_factory=time.time)
    turns: int = 0
    closed: bool = False

    @property
    def live(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses if not h.refuted]

    @property
    def stuck(self) -> bool:
        """Worked on repeatedly with nothing live left — the state that should ask for help."""
        return self.turns >= 2 and not self.live

    def reject(self, hypothesis: Hypothesis, why: str = "") -> None:
        """Move a hypothesis out of contention, keeping it so it is not proposed again."""
        hypothesis.refuted = True
        if why:
            hypothesis.evidence.append(why)
        if hypothesis in self.hypotheses:
            self.hypotheses.remove(hypothesis)
        self.rejected.append(hypothesis)

    def was_rejected(self, claim: str) -> bool:
        return any(h.claim.strip().lower() == str(claim).strip().lower() for h in self.rejected)

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem[:200], "turns": self.turns, "closed": self.closed,
                "known": self.known[:12], "unknown": self.unknown[:12],
                "assumptions": self.assumptions[:8], "intermediate": self.intermediate[:8],
                "pending": self.pending[:8], "stuck": self.stuck,
                "hypotheses": [h.to_dict() for h in self.hypotheses[:8]],
                "rejected": [h.to_dict() for h in self.rejected[:8]]}


@dataclass
class Conclusion:
    """The outcome of one reasoning pass, with the depth it actually took."""

    answer: str = ""
    rung: str = Rung.INTUITION
    confidence: float = 0.0
    decided: bool = False              # False ⇒ genuinely torn, and says so
    hypotheses: List[Hypothesis] = field(default_factory=list)
    margin: float = 0.0                # how clear the winner was of the runner-up
    why: List[str] = field(default_factory=list)
    ms: float = 0.0

    @property
    def depth(self) -> int:
        return Rung.cost(self.rung)

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer[:300], "rung": self.rung, "depth": self.depth,
                "confidence": round(self.confidence, 4), "decided": self.decided,
                "margin": round(self.margin, 4), "why": self.why[:6],
                "hypotheses": [h.to_dict() for h in self.hypotheses[:6]],
                "ms": round(self.ms, 3)}


# A winner must beat the runner-up by this much before she commits. Below it she is genuinely
# torn, and saying so is the honest output — not picking the marginally higher number.
_DECIDE_MARGIN = 0.15
# Confidence at which a rung's answer is good enough to stop descending.
_ENOUGH = 0.75


class Reasoner:
    """Routes a question to the depth it deserves, and weighs what comes back."""

    def __init__(self, brain: Any = None, *, decide_margin: float = _DECIDE_MARGIN,
                 enough: float = _ENOUGH, max_rung: str = Rung.PROOF,
                 capacity: int = 64) -> None:
        self.brain = brain
        self.decide_margin = float(decide_margin)
        self.enough = float(enough)
        self.max_rung = max_rung if max_rung in Rung.ALL else Rung.PROOF
        self.capacity = max(4, int(capacity))

        self.problems: Dict[str, ProblemState] = {}
        self.passes = 0
        self.by_rung: Dict[str, int] = {r: 0 for r in Rung.ALL}
        self.undecided = 0

    # ---- the routing decision ---------------------------------------------- #
    def depth_for(self, *, uncertainty: float, stakes: float = 0.5,
                  task: str = "question") -> str:
        """How deep this question warrants going, before any work is done.

        Three inputs, and they are not interchangeable. High uncertainty on a trivial question is
        not worth a proof; low uncertainty on a load-bearing one still is, because the cost of
        being confidently wrong scales with what depends on it.

        The third is **her own measured competence**. A faculty this task leans on that she is
        demonstrably poor at forces a deeper rung, because a shallow rung *is* a reflex, and a
        reflex from a faculty that is wrong most of the time is the single most dangerous output
        this system can produce. That reading comes from counts in
        :class:`~nyxara.njp.selfmodel.SelfModel`, not from a mood — and it is what makes the
        self-model consequential rather than a report nobody acts on.
        """
        try:
            uncertainty = max(0.0, min(1.0, float(uncertainty)))
            stakes = max(0.0, min(1.0, float(stakes)))
            demand = uncertainty * 0.6 + stakes * 0.4
            demand = min(1.0, demand + self._incompetence_penalty(task))
            if demand < 0.2:
                target = Rung.INTUITION
            elif demand < 0.4:
                target = Rung.ASSOCIATION
            elif demand < 0.6:
                target = Rung.DELIBERATION
            elif demand < 0.8:
                target = Rung.VERIFICATION
            else:
                target = Rung.PROOF
            return target if Rung.cost(target) <= Rung.cost(self.max_rung) else self.max_rung
        except Exception:  # noqa: BLE001
            return Rung.ASSOCIATION

    def _incompetence_penalty(self, task: str) -> float:
        """How much deeper her own record says this task should be taken.

        Only faculties that are *measurably* weak count. An untested faculty is not a weak one,
        and letting inexperience deepen every search would make a cold brain grind through a proof
        engine for "what is my name".
        """
        try:
            model = getattr(self.brain, "self_model", None)
            if model is None:
                return 0.0
            weak = model.unreliable_for(task)
            return min(0.4, 0.2 * len(weak))
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- the pass ----------------------------------------------------------- #
    def reason(self, problem: str, *, candidates: Optional[Sequence[Hypothesis]] = None,
               uncertainty: float = 0.5, stakes: float = 0.5,
               task: str = "question") -> Conclusion:
        """Work the problem, descending only as far as it needs.

        The loop stops the moment an answer is good enough, which is what makes the ladder worth
        having: an easy question costs one rung, and only a hard or costly one pays for five.
        """
        out = Conclusion()
        t0 = time.perf_counter()
        try:
            state = self.state_for(problem)
            state.turns += 1
            self.passes += 1

            target = self.depth_for(uncertainty=uncertainty, stakes=stakes, task=task)
            for hypothesis in (candidates or []):
                self.propose(state, hypothesis)

            for rung in Rung.ALL:
                if Rung.cost(rung) > Rung.cost(target):
                    break
                out.rung = rung
                self._work(rung, state, out)
                self._settle(state, out)
                if out.decided and out.confidence >= self.enough:
                    break

            self.by_rung[out.rung] = self.by_rung.get(out.rung, 0) + 1
            if not out.decided:
                self.undecided += 1
            out.hypotheses = sorted(state.live, key=lambda h: h.score, reverse=True)[:8]
            return out
        except Exception:  # noqa: BLE001 — a failed pass concludes nothing, never crashes
            return out
        finally:
            out.ms = (time.perf_counter() - t0) * 1000.0

    def _work(self, rung: str, state: ProblemState, out: Conclusion) -> None:
        """Do whatever this rung does. Each is a real call into a real organ, or a no-op."""
        try:
            if rung == Rung.INTUITION:
                self._intuit(state, out)
            elif rung == Rung.ASSOCIATION:
                self._associate(state, out)
            elif rung == Rung.DELIBERATION:
                self._deliberate(state, out)
            elif rung == Rung.VERIFICATION:
                self._verify(state, out)
            elif rung == Rung.PROOF:
                self._prove(state, out)
        except Exception:  # noqa: BLE001 — a rung that fails is a rung that added nothing
            pass

    # ---- the rungs ---------------------------------------------------------- #
    def _intuit(self, state: ProblemState, out: Conclusion) -> None:
        """The manifold's read, without settling. Cheap, and only trusted when it says it is."""
        brain = self.brain
        if brain is None or not hasattr(brain, "anticipate"):
            return
        got = brain.anticipate(state.problem)
        if got is not None and getattr(got, "trusted", False):
            out.why.append("the manifold recognised this without settling")

    def _associate(self, state: ProblemState, out: Conclusion) -> None:
        """Grounded structure and recall — where most questions are correctly answered."""
        brain = self.brain
        if brain is None:
            return
        grounder = getattr(brain, "grounder", None)
        if grounder is None:
            return
        answer = grounder.answer(state.problem)
        if getattr(answer, "answered", False):
            self.propose(state, Hypothesis(
                claim=str(answer.text), support=float(answer.confidence),
                source="grounding", evidence=[str(getattr(answer, "why", ""))]))
            out.why.append("answered from grounded structure")
        else:
            # A known-unknown is worth recording: it is what a curiosity pass acts on.
            if state.problem not in state.unknown:
                state.unknown.append(state.problem)

    def _deliberate(self, state: ProblemState, out: Conclusion) -> None:
        """Bring in what else could be true, so the field is not a field of one.

        A single hypothesis is not a debate, and a debate is the only thing that can produce a
        principled "I am not sure" rather than an unopposed guess.
        """
        brain = self.brain
        if brain is None:
            return
        memory = getattr(brain, "memory", None)
        if memory is None or not hasattr(memory, "candidates"):
            return
        for trace, score in list(memory.candidates(state.problem, k=3)):
            claim = str(getattr(trace, "text", "") or "")
            if not claim or state.was_rejected(claim):
                continue
            self.propose(state, Hypothesis(claim=claim, support=float(score),
                                           source="memory", evidence=["recalled"]))
        if len(state.live) > 1:
            out.why.append(f"weighed {len(state.live)} competing readings")

    def _verify(self, state: ProblemState, out: Conclusion) -> None:
        """Check each hypothesis against what she already believes. Contradiction costs it."""
        brain = self.brain
        grounder = getattr(brain, "grounder", None) if brain is not None else None
        if grounder is None:
            return
        for hypothesis in list(state.live):
            for triple in (getattr(grounder, "facts", {}) or {}).values():
                for fact in triple:
                    if getattr(fact, "superseded", False):
                        continue
                    if self._conflicts(hypothesis.claim, fact):
                        hypothesis.contradicted = True
                        hypothesis.against = min(1.0, hypothesis.against + 0.5)
                        hypothesis.evidence.append(
                            f"contradicts {fact.subject} {fact.predicate} {fact.object}")
                        break
            if hypothesis.contradicted:
                out.why.append("a reading contradicted what she already believes")

    @staticmethod
    def _conflicts(claim: str, fact: Any) -> bool:
        """Does this claim disagree with a recorded fact about the same subject and relation?"""
        try:
            subject = str(getattr(fact, "subject", "")).lower()
            obj = str(getattr(fact, "object", "")).lower()
            low = str(claim).lower()
            if not subject or not obj or subject not in low:
                return False
            return obj not in low
        except Exception:  # noqa: BLE001
            return False

    def _prove(self, state: ProblemState, out: Conclusion) -> None:
        """Formal proof where the claim is expressible; otherwise honest silence, never a verdict."""
        try:
            from nyxara.njp.prove import ProofCore
        except Exception:  # noqa: BLE001
            return
        core = ProofCore()
        for hypothesis in list(state.live):
            certificate = core.prove(hypothesis.claim)
            verdict = str(getattr(certificate, "verdict", "") or "").lower()
            if verdict in ("proved", "proven", "valid"):
                hypothesis.proved = True
                out.why.append("proved formally")
            elif verdict in ("refuted", "invalid"):
                state.reject(hypothesis, why="refuted by the proof core")
                out.why.append("a reading was formally refuted")

    # ---- weighing ------------------------------------------------------------ #
    def propose(self, state: ProblemState, hypothesis: Hypothesis) -> Optional[Hypothesis]:
        """Add a candidate, unless she has already rejected it or already holds it."""
        try:
            if not hypothesis.claim.strip() or state.was_rejected(hypothesis.claim):
                return None
            for existing in state.hypotheses:
                if existing.claim.strip().lower() == hypothesis.claim.strip().lower():
                    existing.support = max(existing.support, hypothesis.support)
                    return existing
            state.hypotheses.append(hypothesis)
            return hypothesis
        except Exception:  # noqa: BLE001
            return None

    def _settle(self, state: ProblemState, out: Conclusion) -> None:
        """Pick a winner — or report that there is not one.

        The margin is what makes "I am not sure" principled. Two readings at 0.51 and 0.49 are a
        tie, and returning the first as though it were an answer would be inventing a decision she
        has not actually made.
        """
        try:
            ranked = sorted(state.live, key=lambda h: h.score, reverse=True)
            if not ranked:
                out.answer, out.confidence, out.decided = "", 0.0, False
                return
            best = ranked[0]
            runner_up = ranked[1].score if len(ranked) > 1 else 0.0
            out.margin = best.score - runner_up
            out.confidence = best.score
            out.decided = bool(best.score > 0.0 and out.margin >= self.decide_margin)
            # A hypothesis standing at zero is not an answer — it is a candidate that nothing
            # supports. Returning its text with `decided=False` invites a caller that only reads
            # `answer` to publish something she has no reason at all to believe, which is the
            # exact failure the whole ladder exists to prevent.
            out.answer = best.claim if best.score > 0.0 else ""
            if not out.decided and len(ranked) > 1 and best.score > 0.0:
                out.why.append(
                    f"two readings are too close to call ({best.score:.2f} vs {runner_up:.2f})")
        except Exception:  # noqa: BLE001
            pass

    # ---- problem state -------------------------------------------------------- #
    def state_for(self, problem: str) -> ProblemState:
        """The working state for this problem, resumed if she has seen it before."""
        key = str(problem).strip().lower()[:200]
        got = self.problems.get(key)
        if got is None:
            got = ProblemState(problem=str(problem))
            self.problems[key] = got
            if len(self.problems) > self.capacity:
                for stale in list(self.problems)[: len(self.problems) - self.capacity]:
                    self.problems.pop(stale, None)
        return got

    def close(self, problem: str) -> Optional[ProblemState]:
        """Mark a problem solved. Its conclusions are what consolidation should keep."""
        state = self.problems.get(str(problem).strip().lower()[:200])
        if state is not None:
            state.closed = True
        return state

    # ---- reporting ------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        open_problems = [s for s in self.problems.values() if not s.closed]
        return {
            "passes": self.passes,
            "by_rung": {r: n for r, n in self.by_rung.items() if n},
            "mean_depth": (round(sum(Rung.cost(r) * n for r, n in self.by_rung.items())
                                 / self.passes, 4) if self.passes else None),
            "undecided": self.undecided,
            "open_problems": len(open_problems),
            "stuck_problems": sum(1 for s in open_problems if s.stuck),
            "decide_margin": self.decide_margin,
        }
