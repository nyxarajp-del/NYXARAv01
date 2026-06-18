"""NYXARA · growth/autonomous_scientist.py — the Autonomous Scientist (Level 10c).

Next-level intelligence is not *learning* information — it is **creating** it.

The :class:`~nyxara.growth.scientist.Scientist` runs the scientific method once, on a
question it is *handed*. The Autonomous Scientist closes the loop and drives it itself:

    Observe → Hypothesis → Experiment → Result → Update model
       ↑                                              │
       └──────────────────────────────────────────────┘

1. **Observe**  — she chooses *her own* next question: a follow-up harvested from the last
   conclusion, a gap in her self-knowledge, or — with no external prompting at all — a fresh,
   genuinely *testable* proposition from her own generator. Picking what to study is itself an
   act of creating information: no one handed her the question.
2. **Hypothesis / Experiment / Result** — delegated to the proven
   :class:`~nyxara.growth.scientist.Scientist` (compose, never duplicate): it forms a
   falsifiable hypothesis, runs a *safe* sandboxed experiment, and compares result to prediction.
3. **Update model** — the result is folded back into a :class:`BeliefModel` (her evolving map of
   what she has settled and how sure she is), into the kernel's
   :class:`~nyxara.mind.world_model.WorldModel` as a real transition, and the conclusion's
   suggested next hypothesis is pushed onto the frontier — so the *next* Observe builds on the
   last. Knowledge / graph / memory are already recorded by the Scientist she composes.

Everything degrades gracefully: with no Scientist, world model, knowledge or memory it still
generates testable propositions, verifies them in a sandbox, and updates an in-process belief
model — so it works fully offline (and in CI). Nothing it does touches the world or the gates:
every experiment is sandboxed, exactly as the Scientist already guarantees.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

__all__ = [
    "AutonomousScientist",
    "Belief",
    "BeliefModel",
    "DiscoveryCycle",
    "DiscoveryReport",
    "QuestionOrigin",
]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class QuestionOrigin(str, Enum):
    """Where an observed question came from — the provenance of an inquiry."""
    FOLLOW_UP = "follow_up"   # spawned by a previous conclusion's next hypothesis
    GAP = "gap"               # a known-unknown surfaced from self-knowledge
    SEED = "seed"             # self-generated, genuinely testable proposition
    INVENTION = "invention"   # a theory/optimization she invented via meta-research


# --------------------------------------------------------------------------- #
# The model that gets updated
# --------------------------------------------------------------------------- #
@dataclass
class Belief:
    """One settled (or revised) proposition — a Beta-Bernoulli posterior over 'is it true?'.

    ``alpha``/``beta`` are the accumulated pseudo-counts of (confidence-weighted) supporting vs
    refuting evidence. The posterior mean ``alpha/(alpha+beta)`` is the probability the
    proposition holds; ``verdict``/``confidence`` are derived from it, so repeated consistent
    evidence *sharpens* the belief and conflicting evidence keeps it honestly near 0.5 — rather
    than the old naive average that a single new reading could swing."""
    variable: str
    statement: str
    verdict: str
    confidence: float
    evidence_count: int = 1
    revisions: int = 0
    last_reasoning: str = ""
    alpha: float = 1.0          # supporting pseudo-count (Beta prior 1,1 = uniform)
    beta: float = 1.0           # refuting pseudo-count

    @property
    def support_prob(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "statement": self.statement,
            "verdict": self.verdict,
            "confidence": round(float(self.confidence), 3),
            "support_prob": round(float(self.support_prob), 3),
            "evidence_count": self.evidence_count,
            "revisions": self.revisions,
            "alpha": round(float(self.alpha), 3),
            "beta": round(float(self.beta), 3),
            "last_reasoning": self.last_reasoning,
        }


@dataclass
class BeliefModel:
    """NYXARA's evolving map of what she has settled — *the model that updates*.

    Keyed by the hypothesis ``variable`` so re-investigating the same proposition *revises* a
    belief (blending confidence, counting a revision on a flipped verdict) rather than piling up
    duplicates. This is the concrete, inspectable thing that visibly changes each cycle.
    """
    beliefs: Dict[str, Belief] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.beliefs)

    def get(self, variable: str) -> Optional[Belief]:
        return self.beliefs.get(variable)

    def update(self, report: Any) -> Dict[str, Any]:
        """Fold an InvestigationReport into the model. Returns a delta describing the change."""
        hyps = getattr(report, "hypotheses", None) or []
        conclusion = getattr(report, "conclusion", None)
        if not hyps or conclusion is None:
            return {"changed": False, "new": False, "revised": False}

        hyp = hyps[0]
        variable = (getattr(hyp, "variable", "") or getattr(hyp, "statement", "")
                    or getattr(report, "question", ""))[:120]
        verdict = getattr(getattr(conclusion, "verdict", None), "value",
                          str(getattr(conclusion, "verdict", "inconclusive")))
        confidence = float(getattr(conclusion, "confidence", 0.0) or 0.0)
        reasoning = getattr(conclusion, "reasoning", "") or ""
        statement = getattr(hyp, "statement", "") or getattr(report, "question", "")

        existing = self.beliefs.get(variable)
        if existing is None:
            belief = Belief(variable=variable, statement=statement, verdict="inconclusive",
                            confidence=0.5, evidence_count=1, last_reasoning=reasoning)
            _apply_evidence(belief, verdict, confidence)
            self.beliefs[variable] = belief
            return {"changed": True, "new": True, "revised": False,
                    "variable": variable, "verdict": belief.verdict,
                    "support_prob": round(belief.support_prob, 3)}

        # revise an existing belief: fold the new evidence into the Beta posterior
        prior_verdict = existing.verdict
        existing.evidence_count += 1
        _apply_evidence(existing, verdict, confidence)
        flipped = existing.verdict != prior_verdict
        if flipped:
            existing.revisions += 1
        existing.last_reasoning = reasoning
        existing.statement = statement
        return {"changed": True, "new": False, "revised": True,
                "variable": variable, "verdict": existing.verdict, "flipped": flipped,
                "support_prob": round(existing.support_prob, 3)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": len(self.beliefs),
            "beliefs": [b.to_dict() for b in self.beliefs.values()],
        }


def _apply_evidence(belief: "Belief", verdict: str, confidence: float,
                    margin: float = 0.05) -> None:
    """Fold one confidence-weighted verdict into a belief's Beta posterior and re-derive its
    verdict/confidence from the posterior mean. SUPPORTED adds to ``alpha``, REFUTED to ``beta``;
    INCONCLUSIVE is non-directional (it only counts toward evidence). The verdict is the side of
    0.5 (with a small dead-band) the posterior now favours — so accumulated evidence, not the
    latest single reading, decides."""
    c = max(0.0, min(1.0, float(confidence)))
    if verdict == "supported":
        belief.alpha += c
    elif verdict == "refuted":
        belief.beta += c
    # inconclusive: no directional shift
    p = belief.support_prob
    if p > 0.5 + margin:
        belief.verdict, belief.confidence = "supported", p
    elif p < 0.5 - margin:
        belief.verdict, belief.confidence = "refuted", 1.0 - p
    else:
        belief.verdict, belief.confidence = "inconclusive", 0.5


# --------------------------------------------------------------------------- #
# Records of the loop
# --------------------------------------------------------------------------- #
@dataclass
class DiscoveryCycle:
    """One turn of the autonomous loop: observe → investigate → update model."""
    index: int
    question: str
    origin: QuestionOrigin
    report: Any = None                       # the Scientist's InvestigationReport
    belief_delta: Dict[str, Any] = field(default_factory=dict)
    world_transition_added: bool = False
    created_information: bool = False        # did this cycle settle a *new* belief?

    def to_dict(self) -> Dict[str, Any]:
        rep = self.report
        return {
            "index": self.index,
            "question": self.question,
            "origin": self.origin.value,
            "verdict": (getattr(getattr(getattr(rep, "conclusion", None), "verdict", None),
                                "value", None) if rep is not None else None),
            "report": rep.to_dict() if hasattr(rep, "to_dict") else None,
            "belief_delta": self.belief_delta,
            "world_transition_added": self.world_transition_added,
            "created_information": self.created_information,
        }


@dataclass
class DiscoveryReport:
    """The result of an autonomous discovery run: the chain of cycles and what it created."""
    timestamp: float = field(default_factory=time.time)
    cycles: List[DiscoveryCycle] = field(default_factory=list)
    new_beliefs: int = 0
    revised_beliefs: int = 0
    world_transitions_added: int = 0
    model_size: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cycles_run": len(self.cycles),
            "new_beliefs": self.new_beliefs,
            "revised_beliefs": self.revised_beliefs,
            "world_transitions_added": self.world_transitions_added,
            "model_size": self.model_size,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "cycles": [c.to_dict() for c in self.cycles],
        }


# --------------------------------------------------------------------------- #
# Autonomous Scientist
# --------------------------------------------------------------------------- #
class AutonomousScientist:
    """Self-driven scientific discovery: Observe → Hypothesis → Experiment → Result → Update.

    Parameters
    ----------
    scientist:    a :class:`~nyxara.growth.scientist.Scientist` to run hypothesis/experiment/
                  result (optional; one is created on demand if absent).
    world_model:  a :class:`~nyxara.mind.world_model.WorldModel` to fold results into as
                  transitions (optional).
    memory:       MemoryStore — passed to an on-demand Scientist (optional).
    knowledge:    KnowledgeBase — passed to an on-demand Scientist (optional).
    gap_source:   a zero-arg callable returning a mapping/iterable of known-unknowns to mine for
                  questions during Observe (optional; e.g. ``core.known_unknowns``).
    """

    def __init__(self, *, scientist: Any = None, world_model: Any = None,
                 memory: Any = None, knowledge: Any = None,
                 gap_source: Optional[Callable[[], Any]] = None,
                 meta_researcher: Any = None, max_frontier: int = 64) -> None:
        self.scientist = scientist
        self.world_model = world_model
        self.memory = memory
        self.knowledge = knowledge
        self.gap_source = gap_source
        self.meta_researcher = meta_researcher
        self.model = BeliefModel()
        self._frontier: Deque[tuple] = deque(maxlen=max(8, int(max_frontier)))
        self._seen: set = set()           # questions already observed (no churn on repeats)
        self._seed_n = 0                  # advances the self-generated proposition stream
        self._cycles: List[DiscoveryCycle] = []
        self._cycle_count = 0

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def discover(self, cycles: int = 3) -> DiscoveryReport:
        """Run ``cycles`` full autonomous loops. Always returns a report."""
        t0 = time.monotonic()
        report = DiscoveryReport()
        for _ in range(max(1, int(cycles))):
            cycle = self.step()
            if cycle is None:
                break
            report.cycles.append(cycle)
            if cycle.belief_delta.get("new"):
                report.new_beliefs += 1
            if cycle.belief_delta.get("revised"):
                report.revised_beliefs += 1
            if cycle.world_transition_added:
                report.world_transitions_added += 1
        report.model_size = len(self.model)
        report.elapsed_ms = (time.monotonic() - t0) * 1000
        return report

    def step(self) -> Optional[DiscoveryCycle]:
        """Run exactly one Observe → … → Update cycle. Returns the cycle (always succeeds)."""
        question, origin = self._observe()
        cycle = DiscoveryCycle(index=self._cycle_count, question=question, origin=origin)
        self._cycle_count += 1
        try:
            # Hypothesis + Experiment + Result — delegate to the Scientist (reuse, not rebuild)
            sci = self._ensure_scientist()
            report = sci.investigate(question)
            cycle.report = report
            # Update model
            prior = len(self.model)
            cycle.belief_delta = self.model.update(report)
            cycle.created_information = bool(cycle.belief_delta.get("new"))
            cycle.world_transition_added = self._update_world_model(report, prior)
            self._enqueue_follow_up(report)
        except Exception as exc:  # noqa: BLE001 — a failed cycle is data, not a crash
            cycle.belief_delta = {"changed": False, "error": str(exc)}
        self._cycles.append(cycle)
        return cycle

    def all_cycles(self) -> List[DiscoveryCycle]:
        return list(self._cycles)

    def belief_model(self) -> BeliefModel:
        return self.model

    # ---------------------------------------------------------------------- #
    # Meta-research: invent → test → (gated) integrate, then fold into beliefs
    # ---------------------------------------------------------------------- #
    def meta_discover(self, topic: str) -> Dict[str, Any]:
        """Run one meta-research pass: invent new theories/optimizations, verify them in the
        sandbox, (when authorised) integrate them — and fold each validated invention into the
        belief model as genuinely *created* information. Always returns a dict."""
        mr = self._ensure_meta_researcher()
        if mr is None:
            return {"error": "meta-researcher unavailable", "topic": topic}
        report = mr.run(topic)
        # fold each validated invention into the belief model (origin = INVENTION)
        for cand in report.candidates:
            if not cand.validated:
                continue
            variable = f"invention:{cand.title}"[:120]
            if variable in self.model.beliefs:
                continue
            self.model.beliefs[variable] = Belief(
                variable=variable, statement=cand.title, verdict="supported",
                confidence=0.7, evidence_count=1,
                last_reasoning=cand.experiment_result)
            cycle = DiscoveryCycle(index=self._cycle_count, question=cand.title,
                                   origin=QuestionOrigin.INVENTION,
                                   belief_delta={"changed": True, "new": True},
                                   created_information=True)
            self._cycle_count += 1
            self._cycles.append(cycle)
        d = report.to_dict()
        d["beliefs_held"] = len(self.model)
        return d

    def _ensure_meta_researcher(self) -> Any:
        if self.meta_researcher is None:
            try:
                from nyxara.growth.meta_research import MetaResearcher
                self.meta_researcher = MetaResearcher(memory=self.memory,
                                                      knowledge=self.knowledge)
            except Exception:  # noqa: BLE001 — meta-research is a capability, never required
                return None
        return self.meta_researcher

    # ---------------------------------------------------------------------- #
    # Stage 1 — Observe: choose the next question (create the inquiry)
    # ---------------------------------------------------------------------- #
    def _observe(self) -> tuple:
        """Pick the next question: a pending follow-up, then a self-knowledge gap, then a fresh
        self-generated proposition. Never returns nothing — there is always something to test."""
        # 1) drain the frontier (follow-ups + gaps already queued), skipping seen questions
        while self._frontier:
            question, origin = self._frontier.popleft()
            if question and question not in self._seen:
                self._seen.add(question)
                return question, origin

        # 2) mine a fresh gap from self-knowledge, if a source is wired
        gap = self._next_gap()
        if gap is not None and gap not in self._seen:
            self._seen.add(gap)
            return gap, QuestionOrigin.GAP

        # 3) generate a new, genuinely testable proposition (works with zero external input)
        seed = self._next_seed()
        self._seen.add(seed)
        return seed, QuestionOrigin.SEED

    def _next_gap(self) -> Optional[str]:
        if self.gap_source is None:
            return None
        try:
            gaps = self.gap_source() or {}
            # accept either a mapping (topic -> note) or an iterable of topics
            topics = list(gaps.keys()) if isinstance(gaps, dict) else list(gaps)
            for topic in topics:
                topic = str(topic).strip()
                if topic and topic not in self._seen:
                    # phrase the gap as a question the Scientist can attempt to ground
                    return f"What is known about {topic}?"
        except Exception:  # noqa: BLE001 — gap mining is best-effort, never fatal
            return None
        return None

    def _next_seed(self) -> str:
        """A deterministic stream of *checkable* propositions she poses to herself.

        These are genuinely falsifiable and run end-to-end through the Scientist's sandbox, so
        even with no LLM, no tools and no knowledge she keeps autonomously generating and
        *verifying* new propositions — accumulating information she created, not merely learned.
        """
        templates = (
            lambda i: f"Is {self._nth_candidate_prime(i)} a prime number?",
            lambda i: "Is the sum of two even numbers always even?",
            # phrased to avoid the word "even" so the Scientist routes it to the odd-parity
            # proposition (its even branch is matched first on any text containing "even").
            lambda i: "Is the sum of two odd numbers divisible by 2?",
            lambda i: f"is {2 + i} + {3 + i} = {5 + 2 * i}?",  # always true: tests arithmetic
            # a real Monte-Carlo statistical experiment (significance-tested)
            lambda i: "Is a fair coin's probability of heads 0.5?",
            lambda i: "Do two dice sum to 7 with probability 1/6?",
            # more deterministic, falsifiable math — divisibility + perfect squares
            lambda i: f"Is {4 * (i + 1)} divisible by 4?",
            lambda i: f"Is {(i + 2) * (i + 2)} a perfect square?",
        )
        i = self._seed_n
        self._seed_n += 1
        return templates[i % len(templates)](i // len(templates))

    @staticmethod
    def _nth_candidate_prime(i: int) -> int:
        """A growing stream of integers to test for primality (mix of primes and composites)."""
        return 11 + i  # 11,12,13,… — a deterministic march of SUPPORTED/REFUTED verdicts

    # ---------------------------------------------------------------------- #
    # Stage 5 — Update model
    # ---------------------------------------------------------------------- #
    def _update_world_model(self, report: Any, prior_size: int) -> bool:
        """Fold the result into the kernel's world model as an *epistemic* transition.

        State is the belief-count before the cycle, the action is ``"investigate"``, the next
        state is the belief-count after, and the reward encodes the outcome (positive when a
        hypothesis was SUPPORTED, negative when REFUTED). So ``len(world_model)`` grows and its
        stats honestly reflect that the mind has learned from its own experiments.

        This is deliberately distinct from the *sensorimotor* dynamics fed by
        :mod:`nyxara.sim.real_environment` (real filesystem/compute transitions): they live in
        separate action buckets (``"investigate"`` vs. ``noop``/``create_file``/…), so the kNN
        world model keeps the epistemic signal and the real-environment dynamics cleanly apart.
        """
        if self.world_model is None:
            return False
        try:
            conclusion = getattr(report, "conclusion", None)
            verdict = getattr(getattr(conclusion, "verdict", None), "value",
                              str(getattr(conclusion, "verdict", "inconclusive")))
            confidence = float(getattr(conclusion, "confidence", 0.0) or 0.0)
            reward = {"supported": confidence, "refuted": -confidence}.get(verdict, 0.0)
            after = len(self.model)
            self.world_model.observe((float(prior_size),), "investigate",
                                     (float(after),), reward=reward)
            return True
        except Exception:  # noqa: BLE001 — model update is best-effort, never fatal
            return False

    def _enqueue_follow_up(self, report: Any) -> None:
        """Push the conclusion's suggested next hypothesis onto the frontier — the loop's momentum."""
        try:
            conclusion = getattr(report, "conclusion", None)
            nxt = (getattr(conclusion, "next_hypothesis", "") or "").strip()
            if nxt and nxt not in self._seen:
                self._frontier.append((nxt, QuestionOrigin.FOLLOW_UP))
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    def _ensure_scientist(self) -> Any:
        if self.scientist is None:
            from nyxara.growth.scientist import Scientist
            self.scientist = Scientist(knowledge=self.knowledge, memory=self.memory)
        return self.scientist


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.world_model import WorldModel

    print("=" * 70)
    print("NYXARA autonomous-scientist self-test")
    print("=" * 70)

    wm = WorldModel()
    auto = AutonomousScientist(world_model=wm)   # fully offline: no llm/knowledge/memory
    rep = auto.discover(cycles=6)

    for c in rep.cycles:
        d = c.to_dict()
        print(f"\n[{d['index']}] ({d['origin']}) {d['question']}")
        print(f"    verdict: {d['verdict']}  created_info={d['created_information']}")

    print(f"\nbeliefs held       : {len(auto.belief_model())}")
    print(f"new / revised      : {rep.new_beliefs} / {rep.revised_beliefs}")
    print(f"world transitions  : {len(wm)}")

    # the loop must create information and update the model
    assert len(auto.belief_model()) > 0, "no beliefs were formed"
    assert rep.new_beliefs > 0, "no new information was created"
    assert len(wm) == len(rep.cycles), "world model did not update every cycle"

    # re-running must revise (not duplicate) beliefs about repeated propositions
    auto2 = AutonomousScientist()
    auto2.discover(cycles=4)
    size_after_first = len(auto2.belief_model())
    auto2._seed_n = 0  # rewind the generator to re-pose the same propositions
    auto2._seen.clear()
    auto2._frontier.clear()
    auto2.discover(cycles=4)
    assert len(auto2.belief_model()) == size_after_first, "repeats must revise, not duplicate"

    print("\nALL SELF-TESTS PASSED ✓")
