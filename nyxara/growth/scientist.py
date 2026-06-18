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

import ast
import builtins
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

try:                                       # scipy powers real statistical tests; optional
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:                          # noqa: BLE001
    _scipy_stats = None  # type: ignore
    _HAS_SCIPY = False

try:                                       # sklearn powers TF-IDF semantic grounding; optional
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    _HAS_SKLEARN = True
except Exception:                          # noqa: BLE001
    _TfidfVectorizer = None  # type: ignore
    _cosine_similarity = None  # type: ignore
    _HAS_SKLEARN = False

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
    STATISTICAL = "statistical"       # Monte-Carlo + a real significance test (p-value)
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
    metrics: Dict[str, Any] = field(default_factory=dict)   # e.g. p-value, statistic, n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "detail": self.detail,
            "success": self.success,
            "effects": self.effects,
            "metrics": self.metrics,
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
                 max_hypotheses: int = 3, allow_llm_experiments: bool = False,
                 significance: float = 0.05, mc_trials: int = 4000) -> None:
        self.researcher = researcher
        self._sandbox = sandbox
        self.knowledge = knowledge
        self.knowledge_graph = knowledge_graph
        self.memory = memory
        self.llm = llm
        self.max_hypotheses = max(1, int(max_hypotheses))
        # author + run a real Python experiment via the LLM (gated: needs a real provider);
        # the code is AST-whitelisted and run with restricted builtins, then sandboxed
        self.allow_llm_experiments = bool(allow_llm_experiments)
        self.significance = float(significance)        # α for statistical conclusions
        self.mc_trials = max(100, int(mc_trials))      # Monte-Carlo sample size
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
            report.observation = self._run_experiment(
                report.experiment, question, primary)

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
        """Recognise a checkable proposition in the question.

        Returns a hypothesis whose experiment can be *run* (and possibly refuted) — a sandboxed
        computational test, an exact numeric comparison, or a Monte-Carlo statistical test.
        Returns None when no such proposition is found.
        """
        q = question.lower()

        # a probabilistic/statistical claim → a real Monte-Carlo significance test
        stat = self._detect_statistical(q)
        if stat is not None:
            return stat

        # "is N prime?" / "is N a prime number"
        m = re.search(r"is\s+(-?\d+)\s+(?:a\s+)?prime", q)
        if m:
            n = int(m.group(1))
            return Hypothesis(
                statement=f"{n} is a prime number", prediction=True,
                variable=f"is_prime({n})",
                rationale="primality is decidable by trial division", confidence=0.5)

        # "is N divisible by M" / "is N a multiple of M"
        m = re.search(r"is\s+(-?\d+)\s+(?:divisible by|a multiple of)\s+(-?\d+)", q)
        if m:
            n, d = int(m.group(1)), int(m.group(2))
            return Hypothesis(
                statement=f"{n} is divisible by {d}", prediction=True,
                variable=f"divisible({n},{d})",
                rationale="divisibility is decidable by the modulo operation", confidence=0.5)

        # "is N even/odd"
        m = re.search(r"is\s+(-?\d+)\s+(even|odd)\b", q)
        if m:
            n, parity = int(m.group(1)), m.group(2)
            return Hypothesis(
                statement=f"{n} is {parity}", prediction=True,
                variable=f"parity({n},{parity})",
                rationale="parity is decidable by n mod 2", confidence=0.5)

        # "is N a perfect square"
        m = re.search(r"is\s+(-?\d+)\s+a\s+perfect\s+square", q)
        if m:
            n = int(m.group(1))
            return Hypothesis(
                statement=f"{n} is a perfect square", prediction=True,
                variable=f"perfect_square({n})",
                rationale="a perfect square has an integer square root", confidence=0.5)

        # "is A greater than/less than/equal to B"  (and >, <, >=, <= symbols)
        m = re.search(r"is\s+(-?\d+(?:\.\d+)?)\s+"
                      r"(>=|<=|>|<|greater than or equal to|less than or equal to|"
                      r"greater than|less than|equal to)\s+(-?\d+(?:\.\d+)?)", q)
        if m:
            a, op_raw, b = float(m.group(1)), m.group(2), float(m.group(3))
            op = {"greater than": ">", "less than": "<", "equal to": "==",
                  "greater than or equal to": ">=",
                  "less than or equal to": "<="}.get(op_raw, op_raw)
            return Hypothesis(
                statement=f"{m.group(1)} {op} {m.group(3)}", prediction=True,
                variable=f"compare({m.group(1)},{op},{m.group(3)})",
                rationale="a numeric comparison is exactly decidable", confidence=0.5)

        # parity-closure claims, e.g. "sum of two even numbers is even"
        if "even" in q and ("sum" in q or "add" in q or "+" in q):
            return Hypothesis(
                statement="The sum of two even numbers is always even", prediction=True,
                variable="parity_closure(even+even)",
                rationale="closure of 2Z under addition", confidence=0.6)
        if "odd" in q and ("sum" in q or "add" in q or "+" in q):
            return Hypothesis(
                statement="The sum of two odd numbers is always even", prediction=True,
                variable="parity_closure(odd+odd)",
                rationale="(2a+1)+(2b+1)=2(a+b+1)", confidence=0.6)

        # a bare arithmetic equality, e.g. "is 2 + 2 = 5?" or "2 + 2 == 4"
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*={1,2}\s*"
                      r"(-?\d+(?:\.\d+)?)", q)
        if m:
            a, op, b, claimed = m.group(1), m.group(2), m.group(3), m.group(4)
            return Hypothesis(
                statement=f"{a} {op} {b} = {claimed}", prediction=float(claimed),
                variable=f"{a}{op}{b}",
                rationale="closed-form arithmetic is exactly computable", confidence=0.5)
        return None

    def _detect_statistical(self, q: str) -> Optional[Hypothesis]:
        """Recognise a probabilistic claim that a Monte-Carlo experiment can test."""
        prob = _parse_probability(q)        # a stated probability like 0.5, 1/6, 'half'

        # a fair coin's heads rate
        if "coin" in q and prob is not None:
            return Hypothesis(
                statement=f"a fair coin lands heads with probability {prob:.4g}",
                prediction=prob, variable=f"stat:coin_heads:{prob}",
                rationale="estimable by simulating Bernoulli(0.5) flips", confidence=0.5)

        # two dice summing to a target value
        m = re.search(r"(?:two dice|2 dice|pair of dice).*?sum.*?(?:to|=|equals?)\s+(\d+)", q)
        if m is None:
            m = re.search(r"sum.*?(?:of\s+)?two dice.*?(\d+)", q)
        if m and prob is not None:
            target = int(m.group(1))
            return Hypothesis(
                statement=f"two fair dice sum to {target} with probability {prob:.4g}",
                prediction=prob, variable=f"stat:dice_sum:{target}:{prob}",
                rationale="estimable by simulating two uniform dice", confidence=0.5)

        # a fair die's expected (mean) value
        if ("die" in q or "dice" in q) and ("expected" in q or "average" in q or "mean" in q):
            mval = re.search(r"(\d+(?:\.\d+)?)", q)
            if mval is not None:
                claimed = float(mval.group(1))
                return Hypothesis(
                    statement=f"a fair six-sided die has expected value {claimed:g}",
                    prediction=claimed, variable=f"stat:die_mean:{claimed}",
                    rationale="estimable by simulating uniform die rolls", confidence=0.5)
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
            return Experiment(
                ExperimentKind.COMPUTATIONAL,
                method=f"decide primality of {n} by trial division (sandboxed)",
                expected=True, procedure=lambda ctx: _is_prime(n))

        if var.startswith("divisible("):
            n, d = (int(x) for x in re.search(r"\((-?\d+),(-?\d+)\)", var).groups())
            return Experiment(
                ExperimentKind.COMPUTATIONAL,
                method=f"test whether {d} divides {n} (sandboxed)",
                expected=True, procedure=lambda ctx: (d != 0 and n % d == 0))

        if var.startswith("parity(") and "_closure" not in var:
            mm = re.search(r"\((-?\d+),(even|odd)\)", var)
            n, parity = int(mm.group(1)), mm.group(2)
            want_even = parity == "even"
            return Experiment(
                ExperimentKind.COMPUTATIONAL,
                method=f"test whether {n} is {parity} via n mod 2 (sandboxed)",
                expected=True, procedure=lambda ctx: ((n % 2 == 0) == want_even))

        if var.startswith("perfect_square("):
            n = int(re.search(r"\((-?\d+)\)", var).group(1))
            return Experiment(
                ExperimentKind.COMPUTATIONAL,
                method=f"test whether {n} has an integer square root (sandboxed)",
                expected=True, procedure=lambda ctx: _is_perfect_square(n))

        if var.startswith("compare("):
            mm = re.search(r"\((-?\d+(?:\.\d+)?),(>=|<=|>|<|==),(-?\d+(?:\.\d+)?)\)", var)
            a, op, b = float(mm.group(1)), mm.group(2), float(mm.group(3))
            return Experiment(
                ExperimentKind.COMPUTATIONAL,
                method=f"evaluate {mm.group(1)} {op} {mm.group(3)} (sandboxed)",
                expected=True, procedure=lambda ctx: _compare(a, op, b))

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

        if var.startswith("stat:"):
            return Experiment(
                ExperimentKind.STATISTICAL,
                method=f"Monte-Carlo ({self.mc_trials} trials) + significance test at α={self.significance}",
                expected=hyp.prediction,
                procedure=lambda ctx: _simulate_statistic(var, self.mc_trials),
                tolerance=self.significance)

        # bare arithmetic equality → NUMERIC
        m = re.match(r"(-?\d+(?:\.\d+)?)([+\-*/])(-?\d+(?:\.\d+)?)$", var)
        if m and isinstance(hyp.prediction, (int, float)):
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            return Experiment(
                ExperimentKind.NUMERIC,
                method=f"compute {m.group(1)} {op} {m.group(3)} and compare to prediction",
                expected=hyp.prediction, procedure=lambda ctx: _arith(a, op, b), tolerance=1e-9)

        # an open question → let the LLM author a real, sandboxed Python experiment (gated)
        llm_exp = self._llm_experiment(question, hyp)
        if llm_exp is not None:
            return llm_exp

        # default → KNOWLEDGE: do grounded sources / claims affirm the statement?
        return Experiment(
            ExperimentKind.KNOWLEDGE,
            method="query grounded knowledge for claims affirming the hypothesis",
            expected="affirmed_by_sources")

    # ---------------------------------------------------------------------- #
    # Step 3 — run + compare
    # ---------------------------------------------------------------------- #
    def _run_experiment(self, exp: Experiment, question: str = "",
                        hypothesis: Optional[Hypothesis] = None) -> Observation:
        if exp.kind is ExperimentKind.KNOWLEDGE:
            return self._run_knowledge_experiment(question, hypothesis)

        if exp.kind is ExperimentKind.STATISTICAL:
            return self._run_statistical_experiment(exp)

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

    def _run_knowledge_experiment(self, question: str = "",
                                  hypothesis: Optional[Hypothesis] = None) -> Observation:
        """Measure how many retrieved chunks *actually affirm* the hypothesis.

        Retrieval similarity alone is not evidence: a chunk that merely comes back
        for a query says nothing about whether it supports the claim. So we retrieve
        for the *current* question and then count only the chunks whose content
        genuinely overlaps the hypothesis's key terms — an unrelated chunk affirms
        nothing, and the verdict stays honest (INCONCLUSIVE) rather than falsely
        SUPPORTED.
        """
        if self.knowledge is None:
            return Observation(value=0, detail="no knowledge source available",
                               success=True)
        try:
            chunks = self.knowledge.retrieve(question, k=4)
        except Exception:  # noqa: BLE001
            return Observation(value=0, detail="retrieval failed", success=True)

        statement = hypothesis.statement if hypothesis is not None else question
        texts = [(getattr(c, "text", "") or "") for c in chunks]
        affirming, scores = _semantic_affirm(statement, texts)
        hits = len(affirming)
        backend = "tf-idf cosine" if _HAS_SKLEARN else "term-overlap"
        detail = (f"{len(chunks)} chunk(s) retrieved, {hits} semantically affirm "
                  f"the hypothesis ({backend})" if chunks else "no grounding chunks retrieved")
        return Observation(value=hits, detail=detail, success=True,
                           metrics={"similarities": [round(s, 3) for s in scores],
                                    "grounding": backend})

    def _run_statistical_experiment(self, exp: Experiment) -> Observation:
        """Run a Monte-Carlo simulation and a real significance test against the claim."""
        if exp.procedure is None:
            return Observation(value=None, detail="no statistical procedure", success=False)
        try:
            sim = exp.procedure(None)                       # {"samples", "test", "h0", ...}
        except Exception as exc:  # noqa: BLE001
            return Observation(value=None, detail=f"simulation failed: {exc}", success=False)
        samples = sim.get("samples") or []
        h0 = float(sim.get("h0", exp.expected if isinstance(exp.expected, (int, float)) else 0.0))
        test = sim.get("test", "ttest")
        n = len(samples)
        if n == 0:
            return Observation(value=None, detail="no samples drawn", success=False)
        estimate = float(sum(samples) / n)
        pvalue, statistic = _significance_test(samples, h0, test)
        detail = (f"estimate={estimate:.4f} vs claim={h0:.4f} over n={n} "
                  f"({test}, p={pvalue:.4g})")
        return Observation(value=estimate, detail=detail, success=True,
                           metrics={"pvalue": pvalue, "statistic": statistic, "n": n,
                                    "h0": h0, "estimate": estimate, "test": test})

    # ---------------------------------------------------------------------- #
    # LLM-authored experiments — a real, AST-restricted, sandboxed Python test
    # ---------------------------------------------------------------------- #
    def _llm_experiment(self, question: str, hyp: Hypothesis) -> Optional[Experiment]:
        """Have the LLM author a pure-Python ``experiment()`` that tests the hypothesis.

        Triple-safe: the code must pass an AST whitelist (no imports, no attribute access, no
        dunders, only safe builtins), is executed with a restricted ``__builtins__``, and is run
        inside the :class:`Sandbox` with a wall-clock timeout. Gated: only when explicitly
        enabled AND a real (non-mock) provider is available. Returns None on any doubt — the
        caller then falls back to the knowledge experiment."""
        if not (self.allow_llm_experiments and self._llm_available()):
            return None
        prompt = (
            "Write a single Python function `experiment()` (no arguments) that empirically "
            "tests this hypothesis and RETURNS either a bool (True if it holds) or a tuple "
            "(holds: bool, counterexample). Use ONLY plain arithmetic, comparisons, loops and "
            "these builtins: range, len, sum, abs, min, max, sorted, int, float, bool, round, "
            "pow, divmod, enumerate, zip, all, any. NO imports, NO file/network/system access, "
            "NO attribute access. Keep it under 40 lines and bounded (no infinite loops).\n\n"
            f"Hypothesis: {hyp.statement}\nQuestion: {question}\n\n"
            "Reply with ONLY the function definition.")
        try:
            code = _strip_code_block(str(self.llm.generate(prompt, max_tokens=400,
                                                           temperature=0.1)))
        except Exception:  # noqa: BLE001
            return None
        if not _is_safe_experiment_code(code):
            return None

        def _procedure(ctx: Any) -> Any:
            return _run_safe_experiment(code)

        return Experiment(
            ExperimentKind.COMPUTATIONAL,
            method="run an LLM-authored, AST-restricted Python experiment (sandboxed)",
            expected=True, procedure=_procedure)

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

        if exp.kind is ExperimentKind.STATISTICAL:
            p = float(obs.metrics.get("pvalue", 1.0))
            est = obs.metrics.get("estimate", obs.value)
            h0 = obs.metrics.get("h0", exp.expected)
            alpha = exp.tolerance or self.significance
            if p >= alpha:        # fail to reject H0 → the data is consistent with the claim
                return Conclusion(
                    Verdict.SUPPORTED, min(0.95, 0.6 + 0.3 * p),
                    f"estimate {est:.4f} is statistically consistent with the claimed "
                    f"{h0} (p={p:.3g} ≥ α={alpha}); '{hyp.statement}' is supported.",
                    next_hypothesis="Increase the sample size to tighten the estimate.")
            return Conclusion(
                Verdict.REFUTED, min(0.99, 1.0 - p),
                f"estimate {est:.4f} differs significantly from the claimed {h0} "
                f"(p={p:.3g} < α={alpha}); '{hyp.statement}' is refuted.",
                next_hypothesis=f"Revise the claimed value toward {est:.4f}.")

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
_STOPWORDS = frozenset({
    "evidence", "supports", "support", "does", "do", "did", "is", "are", "was",
    "were", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "not",
    "that", "this", "it", "its", "be", "been", "has", "have", "had", "with",
    "always", "ever", "than", "then", "as", "by", "at", "from", "can", "will",
    "would", "should", "could", "may", "might", "own", "more", "most", "any",
})


def _key_terms(statement: str) -> List[str]:
    """Content words of a hypothesis statement — the terms a chunk must mention
    to count as affirming it (stopwords and short tokens dropped)."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", statement.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def _chunk_affirms(chunk: Any, terms: List[str]) -> bool:
    """A chunk affirms the hypothesis only if its text genuinely overlaps the
    hypothesis's key terms. With no key terms there is nothing to affirm."""
    if not terms:
        return False
    text = (getattr(chunk, "text", "") or "").lower()
    matched = sum(1 for t in set(terms) if t in text)
    # require a real overlap: at least half the key terms (min 1, cap at 2 so a
    # long claim isn't impossible to ground), never zero.
    needed = max(1, min(2, (len(set(terms)) + 1) // 2))
    return matched >= needed


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


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _compare(a: float, op: str, b: float) -> bool:
    return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b, "==": a == b}.get(op, False)


# ---- statistical: parsing, Monte-Carlo simulation, significance testing ---- #
def _parse_probability(q: str) -> Optional[float]:
    """Extract a stated probability from text: 'half', '50%', '1/6', '0.5'."""
    if "half" in q:
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", q)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    m = re.search(r"\b0?\.\d+\b", q)
    if m:
        return float(m.group(0))
    return None


def _simulate_statistic(variable: str, trials: int) -> Dict[str, Any]:
    """Draw Monte-Carlo samples for a ``stat:*`` proposition. Seeded ⇒ deterministic."""
    rng = random.Random(0xC0FFEE)
    parts = variable.split(":")
    kind = parts[1] if len(parts) > 1 else ""
    if kind == "coin_heads":
        h0 = float(parts[2])
        samples = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(trials)]
        return {"samples": samples, "test": "binom", "h0": h0}
    if kind == "dice_sum":
        target, h0 = int(parts[2]), float(parts[3])
        samples = [1.0 if (rng.randint(1, 6) + rng.randint(1, 6)) == target else 0.0
                   for _ in range(trials)]
        return {"samples": samples, "test": "binom", "h0": h0}
    if kind == "die_mean":
        h0 = float(parts[2])
        samples = [float(rng.randint(1, 6)) for _ in range(trials)]
        return {"samples": samples, "test": "ttest", "h0": h0}
    return {"samples": [], "test": "ttest", "h0": 0.0}


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (stdlib fallback when scipy is absent)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _significance_test(samples: List[float], h0: float, test: str) -> Tuple[float, float]:
    """Return (two-sided p-value, statistic). Uses scipy when present, else a normal approx."""
    n = len(samples)
    if test == "binom":
        k = int(round(sum(samples)))
        if _HAS_SCIPY and hasattr(_scipy_stats, "binomtest"):
            res = _scipy_stats.binomtest(k, n, h0)
            return float(res.pvalue), k / n if n else 0.0
        p0 = min(max(h0, 1e-9), 1 - 1e-9)
        se = math.sqrt(p0 * (1 - p0) / n) if n else 1.0
        z = ((k / n) - p0) / se if se else 0.0
        return 2.0 * (1.0 - _phi(abs(z))), k / n if n else 0.0
    # one-sample t-test against h0
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / max(1, n - 1)
    if var <= 1e-15:        # degenerate (zero-variance) sample: exact consistency check
        return (1.0 if abs(mean - h0) <= 1e-12 else 0.0), 0.0
    if _HAS_SCIPY:
        t, p = _scipy_stats.ttest_1samp(samples, h0)
        if math.isnan(p):
            p = 1.0 if abs(mean - h0) <= 1e-12 else 0.0
        return float(p), float(t)
    se = math.sqrt(var / n) if n else 1.0
    t = (mean - h0) / se if se else 0.0
    return 2.0 * (1.0 - _phi(abs(t))), t


# ---- semantic grounding: TF-IDF cosine (sklearn) with a term-overlap fallback ---- #
def _semantic_affirm(statement: str, texts: List[str],
                     threshold: float = 0.18) -> Tuple[List[str], List[float]]:
    """Which chunk texts genuinely affirm the statement, and each one's similarity.

    Uses TF-IDF cosine similarity (real semantic overlap, term-weighted) when sklearn is
    available; falls back to key-term overlap otherwise. Similarity, not mere retrieval, is what
    counts as affirmation — so an unrelated chunk grounds nothing."""
    if not texts:
        return [], []
    if _HAS_SKLEARN:
        try:
            matrix = _TfidfVectorizer().fit_transform([statement, *texts])
            sims = _cosine_similarity(matrix[0:1], matrix[1:]).ravel()
            scores = [float(s) for s in sims]
            affirming = [texts[i] for i, s in enumerate(scores) if s >= threshold]
            return affirming, scores
        except Exception:  # noqa: BLE001 — degrade to the deterministic term overlap
            pass
    terms = _key_terms(statement)
    affirming = [t for t in texts if _text_affirms(t, terms)]
    return affirming, [1.0 if t in affirming else 0.0 for t in texts]


def _text_affirms(text: str, terms: List[str]) -> bool:
    """Term-overlap affirmation on a raw string (the offline fallback for grounding)."""
    if not terms:
        return False
    low = (text or "").lower()
    matched = sum(1 for t in set(terms) if t in low)
    needed = max(1, min(2, (len(set(terms)) + 1) // 2))
    return matched >= needed


# ---- LLM-authored experiments: AST whitelist + restricted, time-bounded execution ---- #
_SAFE_BUILTINS = frozenset({
    "range", "len", "sum", "abs", "min", "max", "sorted", "int", "float", "bool", "round",
    "pow", "divmod", "enumerate", "zip", "all", "any", "map", "filter", "list", "tuple",
    "dict", "set", "str", "reversed", "frozenset", "bytes",
})
# everything that could touch the world, import, reflect, or loop unbounded is rejected
_FORBIDDEN_NODES: Tuple[type, ...] = (
    ast.Import, ast.ImportFrom, ast.Attribute, ast.Global, ast.Nonlocal, ast.ClassDef,
    ast.With, ast.Try, ast.Raise, ast.Delete, ast.While,
)
for _opt in ("AsyncFunctionDef", "AsyncWith", "AsyncFor", "Await", "Yield", "YieldFrom"):
    _node = getattr(ast, _opt, None)
    if _node is not None:
        _FORBIDDEN_NODES = _FORBIDDEN_NODES + (_node,)


def _strip_code_block(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        body = t.splitlines()[1:]
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        t = "\n".join(body)
    return t


def _is_safe_experiment_code(code: str) -> bool:
    """True only if ``code`` defines ``experiment()`` and uses a strict, side-effect-free subset
    (no imports, attribute access, dunders, unbounded loops, or non-whitelisted calls)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if not any(isinstance(n, ast.FunctionDef) and n.name == "experiment" for n in tree.body):
        return False
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            return False
        if isinstance(node, ast.Name) and "__" in node.id:
            return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False                          # no attribute or computed callees
            if node.func.id not in _SAFE_BUILTINS and node.func.id not in defined:
                return False
    return True


def _run_safe_experiment(code: str, *, timeout: float = 2.0) -> Any:
    """Execute validated ``experiment()`` with restricted builtins under a wall-clock timeout."""
    safe = {n: getattr(builtins, n) for n in _SAFE_BUILTINS if hasattr(builtins, n)}
    namespace: Dict[str, Any] = {"__builtins__": safe}
    try:
        exec(compile(code, "<llm-experiment>", "exec"), namespace, namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return (False, f"compile/exec error: {exc}")
    fn = namespace.get("experiment")
    if not callable(fn):
        return (False, "no experiment() defined")
    box: Dict[str, Any] = {}

    def _target() -> None:
        try:
            box["v"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["e"] = exc

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return (False, "experiment timed out")
    if "e" in box:
        return (False, f"experiment error: {box['e']}")
    return box.get("v")


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
              "Is 100 divisible by 4?",
              "Is a fair coin's probability of heads 0.5?",
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

    # a real statistical experiment: a fair coin's heads rate is consistent with 0.5
    r_coin = sci.investigate("Is a fair coin's probability of heads 0.5?")
    assert r_coin.experiment.kind is ExperimentKind.STATISTICAL
    assert r_coin.conclusion.verdict is Verdict.SUPPORTED
    assert "pvalue" in r_coin.observation.metrics

    print(f"\nTotal investigations: {len(sci.all_investigations())}")
    print("\nALL SELF-TESTS PASSED ✓")
