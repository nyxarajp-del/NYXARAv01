"""NYXARA · mind/reasoner.py — model-based reasoning, EFE, strategy & consensus (⬆).

The reasoner is the kernel's general problem-solver. It does not have *one* way to
think; it has several, and it knows when to use which — and when to make them argue.

Three pillars:

1. **Model-based reasoning (expected free energy).** Given a start state and some
   candidate policies, the :class:`ModelBasedReasoner` *imagines* each one through the
   learned :class:`~nyxara.mind.world_model.WorldModel` and scores it by **expected
   free energy** — a blend of *pragmatic* value (reach preferred / high-reward
   states) and *epistemic* value (act where the model is informed). Lowest EFE wins.

2. **Strategy selection.** Different problems want different engines — model-based
   simulation, a verifiable faculty, fast intuition, a heuristic. The
   :class:`StrategySelector` scores each registered :class:`ReasoningStrategy` for
   *applicability* and picks the best fit.

3. **Consensus.** For hard or high-stakes questions, the :class:`ConsensusEngine`
   runs several strategies and aggregates their conclusions by confidence-weighted
   agreement — surfacing the **dissent** when the mind is divided (an abstention
   signal, per Rule 6).

Every result is delivered as a :class:`~nyxara.mind.proposal.Proposal`; the kernel
still disposes.

Depends on :mod:`mind.world_model`, :mod:`mind.faculties`, :mod:`mind.proposal`,
:mod:`memory.provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.mind.faculties import FacultySelector, NoFacultyError, Task
from nyxara.mind.proposal import Proposal, ProposalKind
from nyxara.mind.world_model import Policy, State, WorldModel

__all__ = [
    "ReasoningQuery",
    "Conclusion",
    "ReasoningStrategy",
    "CallableStrategy",
    "ModelBasedReasoner",
    "FacultyReasoner",
    "StrategySelector",
    "ConsensusResult",
    "ConsensusEngine",
    "ReasoningResult",
    "Reasoner",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _consensus_key(answer: Any) -> Any:
    if isinstance(answer, float):
        return round(answer, 6)
    if isinstance(answer, (list, tuple)):
        return tuple(_consensus_key(a) for a in answer)
    try:
        hash(answer)
        return answer
    except TypeError:
        return repr(answer)


# --------------------------------------------------------------------------- #
# Query & conclusion
# --------------------------------------------------------------------------- #
@dataclass
class ReasoningQuery:
    """A question for the reasoner. Fields are used by whichever strategy applies."""

    description: str = ""
    kind: ProposalKind = ProposalKind.ANSWER
    payload: Any = None
    features: Dict[str, Any] = field(default_factory=dict)   # difficulty/stakes/etc.
    # model-based planning inputs
    start: Optional[State] = None
    candidates: Optional[Sequence[Policy]] = None
    preference: Optional[State] = None
    horizon: int = 8
    reward_fn: Optional[Callable[[State, Any, State], float]] = None
    # faculty inputs
    task: Optional[Task] = None


@dataclass
class Conclusion:
    answer: Any
    confidence: float
    rationale: str = ""
    strategy: str = ""
    support: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "confidence": round(self.confidence, 4),
                "strategy": self.strategy, "rationale": self.rationale}


# --------------------------------------------------------------------------- #
# Strategy base + built-ins
# --------------------------------------------------------------------------- #
class ReasoningStrategy:
    name: str = "strategy"

    def applicability(self, query: ReasoningQuery) -> float:  # pragma: no cover
        return 0.0

    def reason(self, query: ReasoningQuery) -> Conclusion:    # pragma: no cover
        raise NotImplementedError


class CallableStrategy(ReasoningStrategy):
    """Wrap any ``reason(query) -> (answer, confidence, rationale)`` as a strategy."""

    def __init__(self, name: str,
                 reason_fn: Callable[[ReasoningQuery], Tuple[Any, float, str]],
                 *, applicability_fn: Optional[Callable[[ReasoningQuery], float]] = None
                 ) -> None:
        self.name = name
        self._reason = reason_fn
        self._appl = applicability_fn

    def applicability(self, query: ReasoningQuery) -> float:
        return self._appl(query) if self._appl else 0.5

    def reason(self, query: ReasoningQuery) -> Conclusion:
        answer, confidence, rationale = self._reason(query)
        return Conclusion(answer, _clamp(confidence), rationale, self.name)


@dataclass
class _PolicyScore:
    policy: Policy
    score: float
    expected_free_energy: float
    total_reward: float
    pragmatic: float
    epistemic: float
    final_state: Optional[State]


class ModelBasedReasoner(ReasoningStrategy):
    """Score candidate policies by expected free energy via the world model."""

    name = "model_based"

    def __init__(self, world_model: WorldModel, *, reward_weight: float = 1.0,
                 preference_weight: float = 1.0, epistemic_weight: float = 0.3) -> None:
        self.wm = world_model
        self.reward_weight = reward_weight
        self.preference_weight = preference_weight
        self.epistemic_weight = epistemic_weight

    def applicability(self, query: ReasoningQuery) -> float:
        if query.start is not None and query.candidates:
            return 0.9
        return 0.0

    def _score(self, query: ReasoningQuery, policy: Policy) -> _PolicyScore:
        traj = self.wm.rollout(query.start, policy, steps=query.horizon,
                               reward_fn=query.reward_fn)
        total_reward = traj.total_reward
        pragmatic = 0.0
        if query.preference is not None and traj.final_state is not None:
            from nyxara.mind.world_model import _dist
            pragmatic = -_dist(traj.final_state, query.preference)
        epistemic = traj.mean_confidence
        score = (self.reward_weight * total_reward
                 + self.preference_weight * pragmatic
                 + self.epistemic_weight * epistemic)
        return _PolicyScore(policy=policy, score=score, expected_free_energy=-score,
                            total_reward=total_reward, pragmatic=pragmatic,
                            epistemic=epistemic, final_state=traj.final_state)

    def evaluate(self, query: ReasoningQuery) -> List[_PolicyScore]:
        scores = [self._score(query, p) for p in (query.candidates or [])]
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def reason(self, query: ReasoningQuery) -> Conclusion:
        scores = self.evaluate(query)
        if not scores:
            return Conclusion(None, 0.0, "no candidate policies", self.name)
        best = scores[0]
        # confidence = model's coverage, scaled by the margin over the runner-up
        margin = 1.0
        if len(scores) > 1:
            spread = abs(best.score - scores[-1].score) + 1e-9
            margin = _clamp((best.score - scores[1].score) / spread + 0.5)
        confidence = _clamp(best.epistemic * (0.5 + 0.5 * margin))
        return Conclusion(
            answer=best.policy, confidence=confidence,
            rationale=f"min expected free energy {best.expected_free_energy:.3f} "
                      f"(reward {best.total_reward:.2f}, epistemic {best.epistemic:.2f})",
            strategy=self.name,
            support={"expected_free_energy": best.expected_free_energy,
                     "total_reward": best.total_reward,
                     "final_state": best.final_state})


class FacultyReasoner(ReasoningStrategy):
    """Delegate to the faculty selector (verifiable engines preferred)."""

    name = "faculty"

    def __init__(self, selector: FacultySelector) -> None:
        self.selector = selector

    def applicability(self, query: ReasoningQuery) -> float:
        return 0.7 if query.task is not None else 0.0

    def reason(self, query: ReasoningQuery) -> Conclusion:
        assert query.task is not None
        try:
            proposal = self.selector.dispatch(query.task)
        except NoFacultyError:
            return Conclusion(None, 0.0, "no faculty available", self.name)
        return Conclusion(answer=proposal.content, confidence=proposal.confidence,
                          rationale=proposal.rationale, strategy=self.name,
                          support={"faculty": proposal.source_faculty})


# --------------------------------------------------------------------------- #
# Strategy selection
# --------------------------------------------------------------------------- #
class StrategySelector:
    def __init__(self, strategies: Sequence[ReasoningStrategy]) -> None:
        self.strategies = list(strategies)

    def applicable(self, query: ReasoningQuery) -> List[Tuple[ReasoningStrategy, float]]:
        scored = [(s, s.applicability(query)) for s in self.strategies]
        scored = [(s, a) for s, a in scored if a > 0.0]
        scored.sort(key=lambda sa: sa[1], reverse=True)
        return scored

    def select(self, query: ReasoningQuery) -> Optional[ReasoningStrategy]:
        ranked = self.applicable(query)
        return ranked[0][0] if ranked else None


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #
@dataclass
class ConsensusResult:
    answer: Any
    confidence: float
    agreement: float                       # winner_weight / total_weight  [0,1]
    conclusions: List[Conclusion]
    dissent: List[Conclusion] = field(default_factory=list)

    @property
    def divided(self) -> bool:
        # an even split (agreement == 0.5) has no majority — treat as divided
        return self.agreement <= 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "confidence": round(self.confidence, 4),
                "agreement": round(self.agreement, 4), "divided": self.divided,
                "n_strategies": len(self.conclusions),
                "dissent": [c.to_dict() for c in self.dissent]}


class ConsensusEngine:
    """Aggregate multiple conclusions by confidence-weighted agreement."""

    def aggregate(self, conclusions: Sequence[Conclusion]) -> ConsensusResult:
        valid = [c for c in conclusions if c.answer is not None and c.confidence > 0]
        if not valid:
            return ConsensusResult(None, 0.0, 0.0, list(conclusions), [])

        # tally confidence-weighted votes per distinct answer
        weights: Dict[Any, float] = {}
        members: Dict[Any, List[Conclusion]] = {}
        total = 0.0
        for c in valid:
            key = _consensus_key(c.answer)
            weights[key] = weights.get(key, 0.0) + c.confidence
            members.setdefault(key, []).append(c)
            total += c.confidence

        winner_key = max(weights, key=weights.get)
        winner_members = members[winner_key]
        agreement = weights[winner_key] / total if total else 0.0
        avg_conf = sum(c.confidence for c in winner_members) / len(winner_members)
        confidence = _clamp(avg_conf * agreement)
        dissent = [c for c in valid if _consensus_key(c.answer) != winner_key]
        return ConsensusResult(answer=winner_members[0].answer, confidence=confidence,
                               agreement=agreement, conclusions=list(conclusions),
                               dissent=dissent)


# --------------------------------------------------------------------------- #
# Reasoner facade
# --------------------------------------------------------------------------- #
class ReasoningMode(str, Enum):
    BEST = "best"
    CONSENSUS = "consensus"


@dataclass
class ReasoningResult:
    conclusion: Conclusion
    proposal: Proposal
    mode: ReasoningMode
    consensus: Optional[ConsensusResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode.value, "conclusion": self.conclusion.to_dict(),
                "proposal": self.proposal.to_dict(),
                "consensus": self.consensus.to_dict() if self.consensus else None}


class Reasoner:
    """Selects strategies, reasons, and (optionally) reaches consensus."""

    def __init__(self, strategies: Optional[Sequence[ReasoningStrategy]] = None) -> None:
        self._strategies: List[ReasoningStrategy] = list(strategies or [])
        self.selector = StrategySelector(self._strategies)
        self.consensus = ConsensusEngine()

    def register(self, strategy: ReasoningStrategy) -> ReasoningStrategy:
        self._strategies.append(strategy)
        self.selector = StrategySelector(self._strategies)
        return strategy

    def _to_proposal(self, query: ReasoningQuery, conclusion: Conclusion) -> Proposal:
        return Proposal(
            kind=query.kind, content=conclusion.answer, source_faculty="reasoner",
            confidence=conclusion.confidence, rationale=conclusion.rationale,
            provenance=Provenance(SourceType.SELF_REFLECTION,
                                  confidence=conclusion.confidence, detail="reasoner",
                                  method="reasoned"),
            metadata={"strategy": conclusion.strategy})

    def reason(self, query: ReasoningQuery, *,
               mode: ReasoningMode = ReasoningMode.BEST) -> ReasoningResult:
        if mode is ReasoningMode.CONSENSUS:
            return self._consensus(query)
        strategy = self.selector.select(query)
        if strategy is None:
            conclusion = Conclusion(None, 0.0, "no applicable strategy", "none")
        else:
            conclusion = strategy.reason(query)
        return ReasoningResult(conclusion, self._to_proposal(query, conclusion),
                               ReasoningMode.BEST)

    def _consensus(self, query: ReasoningQuery) -> ReasoningResult:
        applicable = [s for s, _ in self.selector.applicable(query)]
        conclusions = [s.reason(query) for s in applicable]
        result = self.consensus.aggregate(conclusions)
        conclusion = Conclusion(answer=result.answer, confidence=result.confidence,
                                rationale=f"consensus of {len(conclusions)} strategies "
                                          f"(agreement {result.agreement:.0%})",
                                strategy="consensus",
                                support={"agreement": result.agreement})
        return ReasoningResult(conclusion, self._to_proposal(query, conclusion),
                               ReasoningMode.CONSENSUS, consensus=result)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import random

    print("=" * 70)
    print("NYXARA reasoner self-test")
    print("=" * 70)

    # --- model-based planning over a learned world model --- #
    def true_step(x, a):
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]

    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(400):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (true_step(x, a),), reward=-abs(true_step(x, a)))

    mbr = ModelBasedReasoner(wm)
    q = ReasoningQuery(
        description="get home from x=5",
        start=(5.0,), preference=(0.0,), horizon=6,
        candidates=[["left"] * 6, ["right"] * 6, ["stay"] * 6,
                    ["left", "left", "left", "stay", "stay", "stay"]])
    conc = mbr.reason(q)
    print(f"\nbest policy         : {conc.answer}")
    print(f"  {conc.rationale}")
    assert conc.answer == ["left"] * 6   # straight home maximises reward

    # --- strategy selection --- #
    reasoner = Reasoner([mbr])
    chosen = reasoner.selector.select(q)
    print(f"\nselected strategy   : {chosen.name}")
    assert chosen.name == "model_based"

    # --- consensus across diverse strategies --- #
    # three "experts" weigh in on a yes/no question with differing confidence
    s_yes1 = CallableStrategy("expert_a", lambda q: ("yes", 0.8, "a says yes"),
                              applicability_fn=lambda q: 1.0)
    s_yes2 = CallableStrategy("expert_b", lambda q: ("yes", 0.7, "b says yes"),
                              applicability_fn=lambda q: 1.0)
    s_no = CallableStrategy("contrarian", lambda q: ("no", 0.9, "contrarian says no"),
                            applicability_fn=lambda q: 1.0)
    panel = Reasoner([s_yes1, s_yes2, s_no])
    res = panel.reason(ReasoningQuery(description="should we act?"),
                       mode=ReasoningMode.CONSENSUS)
    print(f"\nconsensus           : {res.consensus.to_dict()}")
    assert res.conclusion.answer == "yes"          # 0.8+0.7 outweighs 0.9
    assert res.consensus.dissent[0].answer == "no" # dissent surfaced (Rule 6)

    # a divided panel flags low agreement
    split = Reasoner([
        CallableStrategy("x", lambda q: ("A", 0.6, ""), applicability_fn=lambda q: 1.0),
        CallableStrategy("y", lambda q: ("B", 0.6, ""), applicability_fn=lambda q: 1.0),
    ])
    res2 = split.reason(ReasoningQuery(), mode=ReasoningMode.CONSENSUS)
    print(f"divided panel       : agreement={res2.consensus.agreement:.2f} "
          f"divided={res2.consensus.divided}")
    assert res2.consensus.divided

    # output is a Proposal
    assert isinstance(res.proposal, Proposal)
    print(f"\noutput is a Proposal : kind={res.proposal.kind.value} "
          f"source={res.proposal.source_faculty}")

    print("\nALL SELF-TESTS PASSED ✓")
