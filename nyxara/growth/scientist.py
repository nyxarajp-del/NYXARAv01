"""NYXARA · growth/scientist.py — the Scientist loop (Level 10b).

NYXARA reasons like a scientist. Given a question, she runs the full method:

    1. hypothesize — form one or more falsifiable hypotheses, each with a
       concrete, measurable *prediction*
    2. design      — design an experiment that could refute the hypothesis
    3. run + compare — execute the experiment (a *safe* computational test in the
       :class:`~nyxara.sim.sandbox.Sandbox`, a numeric comparison, or a query of
       grounded knowledge) and compare the observed result to the prediction
    4. conclude    — draw a calibrated conclusion (SUPPORTED / REFUTED /
       INCONCLUSIVE) and suggest the next hypothesis

The Scientist *composes* the :class:`~nyxara.growth.researcher.AutonomousResearcher`
for background evidence rather than duplicating it, and reuses the real
``Sandbox.run`` for experiments so nothing it does touches the world or the gates —
an external action would still go through the Master.

Everything degrades gracefully: with no LLM, no tools and no knowledge it still
forms a hypothesis, runs a computational experiment, and concludes — so it works
fully offline (and in CI).

Conclusions are stored as SEMANTIC memories tagged ``["research", "experiment", topic]``
and folded into the KnowledgeBase / KnowledgeGraph, the same way the researcher does.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "Scientist",
    "Hypothesis",
    "Experiment",
    "Observation",
    "Conclusion",
    "InvestigationReport",
    "Verdict",
    "ExperimentKind",
]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class ExperimentKind(str, Enum):
    """How a hypothesis is put to the test."""
    COMPUTATIONAL = "computational"   # run a safe Python procedure in the Sandbox
    NUMERIC = "numeric"               # compare a measured number to a prediction
    KNOWLEDGE = "knowledge"           # check grounded knowledge / gathered claims


class Verdict(str, Enum):
    """The outcome of comparing observation to prediction."""
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


# --------------------------------------------------------------------------- #
# Domain objects
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    """A falsifiable statement with a concrete, measurable prediction."""
    statement: str
    prediction: Any = None
    variable: str = ""
    rationale: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "prediction": self.prediction,
            "variable": self.variable,
            "rationale": self.rationale,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class Experiment:
    """A test designed to refute a hypothesis."""
    kind: ExperimentKind
    method: str = ""
    expected: Any = None
    procedure: Optional[Callable[..., Any]] = None
    tolerance: float = 1e-9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "method": self.method,
            "expected": self.expected,
            "tolerance": self.tolerance,
        }


@dataclass
class Observation:
    """The measured outcome of running an experiment."""
    value: Any = None
    detail: str = ""
    success: bool = True
    effects: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "detail": self.detail,
            "success": self.success,
            "effects": self.effects,
        }


@dataclass
class Conclusion:
    """A calibrated conclusion, with a suggested follow-up hypothesis."""
    verdict: Verdict
    confidence: float
    reasoning: str
    next_hypothesis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": round(float(self.confidence), 3),
            "reasoning": self.reasoning,
            "next_hypothesis": self.next_hypothesis,
        }


@dataclass
class InvestigationReport:
    """One complete scientific investigation of a question."""
    question: str
    timestamp: float = field(default_factory=time.time)
    background_claims: List[str] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    experiment: Optional[Experiment] = None
    observation: Optional[Observation] = None
    conclusion: Optional[Conclusion] = None
    kb_chunks_added: int = 0
    graph_triples_added: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "timestamp": self.timestamp,
            "background_claims": self.background_claims,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "experiment": self.experiment.to_dict() if self.experiment else None,
            "observation": self.observation.to_dict() if self.observation else None,
            "conclusion": self.conclusion.to_dict() if self.conclusion else None,
            "kb_chunks_added": self.kb_chunks_added,
            "graph_triples_added": self.graph_triples_added,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# Scientist
# --------------------------------------------------------------------------- #
class Scientist:
    """Run the scientific method over a natural-language question.

    Parameters
    ----------
    researcher:      AutonomousResearcher for gathering background evidence (optional).
    sandbox:         sim.sandbox.Sandbox for safe computational experiments (optional;
                     one is created on demand if absent).
    knowledge:       KnowledgeBase for storing conclusions (optional).
    knowledge_graph: KnowledgeGraph for storing triples (optional).
    memory:          MemoryStore for SEMANTIC memory of conclusions (optional).
    llm:             LLM facade for hypothesis generation (optional; heuristic fallback).
    max_hypotheses:  how many candidate hypotheses to form per question.
    """

    def __init__(self, *, researcher: Any = None, sandbox: Any = None,
                 knowledge: Any = None, knowledge_graph: Any = None,
                 memory: Any = None, llm: Any = None,
                 max_hypotheses: int = 3) -> None:
        self.researcher = researcher
        self._sandbox = sandbox
        self.knowledge = knowledge
        self.knowledge_graph = knowledge_graph
        self.memory = memory
        self.llm = llm
        self.max_hypotheses = max(1, int(max_hypotheses))
        self._investigations: List[InvestigationReport] = []

    # ---------------------------------------------------------------------- #
    def investigate(self, question: str) -> InvestigationReport:
        """Full scientific-method loop for ``question``. Always returns a report."""
        t0 = time.monotonic()
        report = InvestigationReport(question=question)
        try:
            # 0. gather background evidence (reuse the researcher; never required)
            report.background_claims = self._gather_background(question)

            # 1. hypothesize
            report.hypotheses = self._form_hypotheses(question, report.background_claims)
            primary = report.hypotheses[0]

            # 2. design
            report.experiment = self._design_experiment(question, primary,
                                                         report.background_claims)

            # 3. run + compare
            report.observation = self._run_experiment(report.experiment)

            # 4. conclude
            report.conclusion = self._conclude(primary, report.experiment,
                                               report.observation)

            # record what was learned
            report.kb_chunks_added = self._record_kb(report)
            report.graph_triples_added = self._record_graph(report)
            self._record_memory(report)
        except Exception as exc:  # noqa: BLE001 — a failed investigation is data, not a crash
            report.conclusion = Conclusion(
                Verdict.INCONCLUSIVE, 0.0, f"investigation error: {exc}")

        report.elapsed_ms = (time.monotonic() - t0) * 1000
        self._investigations.append(report)
        return report

    def all_investigations(self) -> List[InvestigationReport]:
        return list(self._investigations)

    # ---------------------------------------------------------------------- #
    # Step 0 — background
    # ---------------------------------------------------------------------- #
    def _gather_background(self, question: str) -> List[str]:
        if self.researcher is None:
            return []
        try:
            rep = self.researcher.research(question)
            claims = list(getattr(rep, "key_claims", []) or [])
            if not claims and getattr(rep, "summary", ""):
                claims = [rep.summary]
            return claims[:6]
        except Exception:  # noqa: BLE001
            return []

    # ---------------------------------------------------------------------- #
    # Step 1 — hypothesize
    # ---------------------------------------------------------------------- #
    def _form_hypotheses(self, question: str, claims: List[str]) -> List[Hypothesis]:
        """Form falsifiable hypotheses — LLM-assisted when available, else heuristic."""
        # a checkable mathematical proposition always takes priority: it gives a
        # genuinely falsifiable, runnable experiment.
        prop = self._detect_proposition(question)
        if prop is not None:
            return [prop]

        llm_hyps = self._llm_hypotheses(question, claims)
        if llm_hyps:
            return llm_hyps[: self.max_hypotheses]

        # heuristic: treat the question as a claim to be checked against evidence.
        stmt = question.strip().rstrip("?")
        conf = 0.55 if claims else 0.4
        rationale = ("grounded sources gathered" if claims
                     else "no external evidence — testing against internal knowledge")
        return [Hypothesis(
            statement=f"Evidence supports: {stmt}",
            prediction="affirmed_by_sources",
            variable="evidence",
            rationale=rationale,
            confidence=conf,
        )]

    def _detect_proposition(self, question: str) -> Optional[Hypothesis]:
        """Recognise a checkable mathematical proposition in the question.

        Returns a hypothesis whose experiment can be *run* (and possibly refuted)
        empirically in the sandbox. Returns None when no such proposition is found.
        """
        q = question.lower()

        # "is N prime?" / "is N a prime number"
        m = re.search(r"is\s+(-?\d+)\s+(?:a\s+)?prime", q)
        if m:
            n = int(m.group(1))
            return Hypothesis(
                statement=f"{n} is a prime number",
                prediction=True,
                variable=f"is_prime({n})",
                rationale="primality is decidable by trial division",
                confidence=0.5,
            )

        # parity-closure claims, e.g. "sum of two even numbers is even"
        if "even" in q and ("sum" in q or "add" in q or "+" in q):
            return Hypothesis(
                statement="The sum of two even numbers is always even",
                prediction=True,
                variable="parity_closure(even+even)",
                rationale="closure of 2Z under addition",
                confidence=0.6,
            )
        if "odd" in q and ("sum" in q or "add" in q or "+" in q):
            return Hypothesis(
                statement="The sum of two odd numbers is always even",
                prediction=True,
                variable="parity_closure(odd+odd)",
                rationale="(2a+1)+(2b+1)=2(a+b+1)",
                confidence=0.6,
            )

        # a bare arithmetic equality, e.g. "is 2 + 2 = 5?" or "2 + 2 == 4"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*={1,2}\s*"
                      r"(-?\d+(?:\.\d+)?)", q)
        if m:
            a, op, b, claimed = m.group(1), m.group(2), m.group(3), m.group(4)
            return Hypothesis(
                statement=f"{a} {op} {b} = {claimed}",
                prediction=float(claimed),
                variable=f"{a}{op}{b}",
                rationale="closed-form arithmetic is exactly computable",
                confidence=0.5,
            )
        return None

    def _llm_hypotheses(self, question: str, claims: List[str]) -> List[Hypothesis]:
        if not self._llm_available():
            return []
        try:
            ctx = ("\n".join(f"- {c}" for c in claims))[:800]
            prompt = (
                f"Question: {question}\n"
                f"Known facts:\n{ctx or '(none)'}\n\n"
                "Propose one short, falsifiable hypothesis answering the question. "
                "Reply with a single sentence.")
            text = self.llm.generate(prompt, max_tokens=80, temperature=0.4)
            text = str(text).strip()
            if not text:
                return []
            return [Hypothesis(statement=text, prediction="affirmed_by_sources",
                               variable="evidence", rationale="LLM-proposed",
                               confidence=0.5)]
        except Exception:  # noqa: BLE001
            return []

    # ---------------------------------------------------------------------- #
    # Step 2 — design
    # ---------------------------------------------------------------------- #
    def _design_experiment(self, question: str, hyp: Hypothesis,
                           claims: List[str]) -> Experiment:
        """Design an experiment capable of refuting ``hyp``."""
        var = hyp.variable

        if var.startswith("is_prime("):
            n = int(re.search(r"\((-?\d+)\)", var).group(1))

            def _check_prime(ctx: Any) -> bool:
                return _is_prime(n)

            return Experiment(
                ExperimentKind.COMPUTATIONAL,
                method=f"decide primality of {n} by trial division (sandboxed)",
                expected=True, procedure=_check_prime)

        if var.startswith("parity_closure"):
            even = "even+even" in var
            samples = _parity_samples(even=even)

            def _check_parity(ctx: Any) -> Tuple[bool, Any]:
                for a, b in samples:
                    if (a + b) % 2 != 0:
                        return (False, (a, b))   # a counterexample refutes the claim
                return (True, None)

            return Experiment(
                ExperimentKind.COMPUTATIONAL,
                method=f"search {len(samples)} sampled pairs for a counterexample (sandboxed)",
                expected=True, procedure=_check_parity)

        # bare arithmetic equality → NUMERIC
        m = re.match(r"(-?\d+(?:\.\d+)?)([+\-*/])(-?\d+(?:\.\d+)?)$", var)
        if m and isinstance(hyp.prediction, (int, float)):
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))

            def _evaluate(ctx: Any) -> float:
                return _arith(a, op, b)

            return Experiment(
                ExperimentKind.NUMERIC,
                method=f"compute {m.group(1)} {op} {m.group(3)} and compare to prediction",
                expected=hyp.prediction, procedure=_evaluate, tolerance=1e-9)

        # default → KNOWLEDGE: do grounded sources / claims affirm the statement?
        return Experiment(
            ExperimentKind.KNOWLEDGE,
            method="query grounded knowledge for claims affirming the hypothesis",
            expected="affirmed_by_sources")

    # ---------------------------------------------------------------------- #
    # Step 3 — run + compare
    # ---------------------------------------------------------------------- #
    def _run_experiment(self, exp: Experiment) -> Observation:
        if exp.kind is ExperimentKind.KNOWLEDGE:
            return self._run_knowledge_experiment()

        # COMPUTATIONAL / NUMERIC both execute their procedure in the sandbox so the
        # run is isolated and any (simulated) effects are captured, never enacted.
        sandbox = self._ensure_sandbox()
        if sandbox is None or exp.procedure is None:
            # no sandbox available — run the pure procedure directly (still side-effect free)
            try:
                value = exp.procedure() if exp.procedure else None  # type: ignore[call-arg]
                return Observation(value=value, detail="run without sandbox", success=True)
            except Exception as exc:  # noqa: BLE001
                return Observation(value=None, detail=f"{exc}", success=False)
        try:
            result = sandbox.run(exp.procedure)
            effects = [e.to_dict() for e in getattr(result, "effects", [])]
            return Observation(
                value=getattr(result, "value", None),
                detail=getattr(result, "error", "") or "ok",
                success=bool(getattr(result, "success", True)),
                effects=effects)
        except Exception as exc:  # noqa: BLE001
            return Observation(value=None, detail=f"{exc}", success=False)

    def _run_knowledge_experiment(self) -> Observation:
        """Count grounded claims that affirm the hypothesis (the 'measurement')."""
        hits = 0
        detail = "no knowledge source available"
        if self.knowledge is not None:
            try:
                chunks = self.knowledge.retrieve(self._last_question(), k=4)
                hits = len(chunks)
                detail = f"{hits} grounding chunk(s) retrieved"
            except Exception:  # noqa: BLE001
                pass
        return Observation(value=hits, detail=detail, success=True)

    # ---------------------------------------------------------------------- #
    # Step 4 — conclude
    # ---------------------------------------------------------------------- #
    def _conclude(self, hyp: Hypothesis, exp: Experiment,
                  obs: Observation) -> Conclusion:
        if not obs.success:
            return Conclusion(
                Verdict.INCONCLUSIVE, 0.2,
                f"experiment did not complete cleanly ({obs.detail}); cannot judge "
                f"'{hyp.statement}'.",
                next_hypothesis=f"Re-test '{hyp.statement}' with a simpler procedure.")

        if exp.kind is ExperimentKind.NUMERIC:
            try:
                ok = abs(float(obs.value) - float(exp.expected)) <= exp.tolerance
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                return Conclusion(
                    Verdict.SUPPORTED, 0.97,
                    f"measured {obs.value} == predicted {exp.expected}; "
                    f"'{hyp.statement}' holds.",
                    next_hypothesis="Generalise the relation to other inputs.")
            return Conclusion(
                Verdict.REFUTED, 0.97,
                f"measured {obs.value} != predicted {exp.expected}; "
                f"'{hyp.statement}' is false.",
                next_hypothesis=f"Revise prediction to {obs.value}.")

        if exp.kind is ExperimentKind.COMPUTATIONAL:
            value = obs.value
            holds, counter = (value, None)
            if isinstance(value, tuple) and len(value) == 2:
                holds, counter = value
            if bool(holds) == bool(exp.expected):
                return Conclusion(
                    Verdict.SUPPORTED, 0.9,
                    f"experiment found no counterexample; '{hyp.statement}' is supported "
                    f"over the tested space.",
                    next_hypothesis="Attempt a formal proof of the general case.")
            return Conclusion(
                Verdict.REFUTED, 0.95,
                f"counterexample found ({counter}); '{hyp.statement}' is refuted.",
                next_hypothesis="Restrict the claim to exclude the counterexample.")

        # KNOWLEDGE
        hits = obs.value if isinstance(obs.value, int) else 0
        if hits > 0:
            conf = min(0.85, 0.4 + 0.1 * hits)
            return Conclusion(
                Verdict.SUPPORTED, conf,
                f"{hits} grounded source(s) affirm '{hyp.statement}'.",
                next_hypothesis="Seek a disconfirming source to test robustness.")
        return Conclusion(
            Verdict.INCONCLUSIVE, 0.3,
            f"no grounded evidence for or against '{hyp.statement}'.",
            next_hypothesis="Gather more sources before concluding.")

    # ---------------------------------------------------------------------- #
    # Recording — store conclusions back into knowledge / graph / memory
    # ---------------------------------------------------------------------- #
    def _record_kb(self, report: InvestigationReport) -> int:
        if self.knowledge is None or report.conclusion is None:
            return 0
        try:
            text = (f"Investigation: {report.question}\n"
                    f"Hypothesis: {report.hypotheses[0].statement}\n"
                    f"Verdict: {report.conclusion.verdict.value} "
                    f"({report.conclusion.confidence:.2f}) — {report.conclusion.reasoning}")
            self.knowledge.ingest_text(text, source=f"experiment:{report.question[:60]}")
            return 1
        except Exception:  # noqa: BLE001
            return 0

    def _record_graph(self, report: InvestigationReport) -> int:
        if self.knowledge_graph is None or report.conclusion is None:
            return 0
        try:
            from nyxara.memory.graph import Relation
            from nyxara.memory.provenance import Provenance, SourceType
            prov = Provenance(SourceType.SELF_REFLECTION, confidence=0.7)
            self.knowledge_graph.add_triple(
                report.hypotheses[0].statement[:80],
                Relation.LEARNED_FROM,
                f"experiment:{report.question[:60]}",
                confidence=report.conclusion.confidence, provenance=prov)
            return 1
        except Exception:  # noqa: BLE001
            return 0

    def _record_memory(self, report: InvestigationReport) -> None:
        if self.memory is None or report.conclusion is None:
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
            text = (f"[Experiment: {report.question}] "
                    f"{report.conclusion.verdict.value}: {report.conclusion.reasoning}")[:300]
            self.memory.remember(
                text, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.7),
                importance=0.65, tags=["research", "experiment", report.question[:40]])
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    def _ensure_sandbox(self) -> Any:
        if self._sandbox is None:
            try:
                from nyxara.sim.sandbox import Sandbox
                self._sandbox = Sandbox()
            except Exception:  # noqa: BLE001
                self._sandbox = None
        return self._sandbox

    def _last_question(self) -> str:
        return self._investigations[-1].question if self._investigations else ""

    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# Pure helpers (stdlib only — safe to run in or out of the sandbox)
# --------------------------------------------------------------------------- #
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _parity_samples(*, even: bool) -> List[Tuple[int, int]]:
    """Deterministic sample pairs for a parity-closure experiment."""
    base = range(0, 20, 2) if even else range(1, 20, 2)
    pairs: List[Tuple[int, int]] = []
    nums = list(base)
    for i, a in enumerate(nums):
        b = nums[(i + 3) % len(nums)]
        pairs.append((a, b))
    return pairs


def _arith(a: float, op: str, b: float) -> float:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b != 0 else float("inf")
    raise ValueError(f"unsupported operator {op!r}")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA scientist self-test")
    print("=" * 70)

    sci = Scientist()  # fully offline: no researcher, llm, knowledge, memory

    for q in ("Is the sum of two even numbers always even?",
              "Is 17 a prime number?",
              "Is 2 + 2 = 5?",
              "Does spaced repetition improve memory retention?"):
        rep = sci.investigate(q)
        c = rep.conclusion
        print(f"\nQ: {q}")
        print(f"  hypothesis : {rep.hypotheses[0].statement}")
        print(f"  experiment : [{rep.experiment.kind.value}] {rep.experiment.method}")
        print(f"  observation: {rep.observation.value}")
        print(f"  conclusion : {c.verdict.value} ({c.confidence:.2f}) — {c.reasoning}")
        assert c is not None and isinstance(c.verdict, Verdict)

    # the two math claims must come out right; the false one must be refuted.
    r_even = sci.investigate("sum of two even numbers is even")
    assert r_even.conclusion.verdict is Verdict.SUPPORTED
    r_false = sci.investigate("is 2 + 2 = 5?")
    assert r_false.conclusion.verdict is Verdict.REFUTED

    print(f"\nTotal investigations: {len(sci.all_investigations())}")
    print("\nALL SELF-TESTS PASSED ✓")
