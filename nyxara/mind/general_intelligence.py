"""NYXARA · mind/general_intelligence.py — domain-aware General Intelligence (Level 4+).

The rest of the mind already holds real reasoning power — verifiable math/logic faculties,
a symbolic math engine and prover, a hypothesis-testing Scientist, RAG with honest
abstention, world-model rollouts, a real code sandbox, governed web tools, an LLM council,
and the six-part :class:`~nyxara.mind.strategic.StrategicIntelligence` framework. What was
missing is **domain awareness**: every problem flowed through the *same* generic reasoner.

This module makes NYXARA solve a problem the way the *right kind of expert* would. It:

  1. **classifies** the problem into a domain — coding, maths, science, business, robotics,
     medicine, design, law — or flags it *novel* when nothing fits;
  2. **frames** it with that domain's methodology (the expert lens) and binds it to the
     existing real engine best suited to it (coding→sandbox, maths→faculties/prover,
     science→Scientist, medicine/law→RAG+web+cite, business→strategic, …);
  3. applies domain **constraints** — knowledge-heavy domains must cite sources and
     honestly abstain (with a professional-consultation caveat) rather than confabulate;
  4. **adapts** to a completely new field by building a first-principles expert on the fly
     and *learning a profile* so the field is recognised, and improves, next time.

Two surfaces:

* :meth:`GeneralIntelligence.frame` — cheap, stateless, *advisory*. Returns a
  :class:`DomainFrame` the orchestrator prepends to the reasoning context on every turn so
  the council/reasoner thinks as the right expert. It proposes; the kernel still disposes.
* :meth:`GeneralIntelligence.solve` — runs the bound engine end-to-end and returns a
  :class:`DomainSolution` (the ``/v1/solve`` and ``/solve`` path).

Everything degrades gracefully: with no LLM, no council, no tools and no knowledge it still
classifies, frames and produces a structured offline answer, so it runs fully in CI.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "Domain",
    "DomainFrame",
    "DomainSolution",
    "DomainClassifier",
    "DomainExpert",
    "GeneralIntelligence",
]

_WORD = re.compile(r"[A-Za-z][A-Za-z+#.\-]*")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class Domain(str, Enum):
    """The field a problem belongs to. ``GENERAL`` is the fallback / novel-field bucket."""
    CODING = "coding"
    MATH = "math"
    SCIENCE = "science"
    BUSINESS = "business"
    ROBOTICS = "robotics"
    MEDICINE = "medicine"
    DESIGN = "design"
    LAW = "law"
    GENERAL = "general"


# Keyword signals per domain. Weighted: a strong term scores 2, a softer term 1. The
# classifier sums matched weights and normalises — deterministic, offline, explainable.
_SIGNALS: Dict[Domain, Dict[str, int]] = {
    Domain.CODING: {
        "code": 2, "function": 2, "python": 2, "javascript": 2, "java": 2, "c++": 2,
        "rust": 2, "go": 1, "bug": 2, "compile": 2, "algorithm": 2, "program": 2,
        "debug": 2, "api": 1, "regex": 2, "loop": 1, "array": 1, "class": 1,
        "variable": 1, "script": 2, "refactor": 2, "stacktrace": 2, "syntax": 2,
        "implement": 1, "sql": 2, "database": 1, "git": 1, "compiler": 2,
    },
    Domain.MATH: {
        "solve": 1, "equation": 2, "integral": 2, "derivative": 2, "calculus": 2,
        "algebra": 2, "geometry": 2, "probability": 2, "matrix": 2, "theorem": 2,
        "prove": 2, "factor": 1, "polynomial": 2, "sum": 1, "product": 1, "prime": 2,
        "sequence": 2, "limit": 1, "logarithm": 2, "trigonometry": 2, "compute": 1,
        "calculate": 1, "percent": 2, "arithmetic": 2,
    },
    Domain.SCIENCE: {
        "physics": 2, "chemistry": 2, "biology": 2, "experiment": 2, "hypothesis": 2,
        "molecule": 2, "atom": 2, "energy": 1, "velocity": 2, "reaction": 1, "cell": 1,
        "gene": 2, "evolution": 1, "quantum": 2, "thermodynamics": 2, "gravity": 2,
        "electron": 2, "force": 1, "mass": 1, "temperature": 1, "observe": 1,
        "scientific": 2, "theory": 1,
    },
    Domain.BUSINESS: {
        "business": 2, "market": 2, "revenue": 2, "profit": 2, "strategy": 2,
        "customer": 2, "pricing": 2, "startup": 2, "investment": 2, "growth": 1,
        "marketing": 2, "sales": 2, "competitor": 2, "roi": 2, "budget": 1,
        "stakeholder": 2, "scale": 1, "product-market": 2, "valuation": 2, "churn": 2,
        "acquisition": 1, "company": 1,
    },
    Domain.ROBOTICS: {
        "robot": 2, "actuator": 2, "sensor": 2, "motor": 2, "kinematics": 2,
        "trajectory": 2, "servo": 2, "lidar": 2, "control": 1, "pid": 2, "arm": 1,
        "gripper": 2, "navigation": 2, "autonomous": 1, "drone": 2, "torque": 2,
        "joint": 1, "odometry": 2, "slam": 2, "embodied": 2, "manipulator": 2,
    },
    Domain.MEDICINE: {
        "patient": 2, "symptom": 2, "diagnosis": 2, "disease": 2, "treatment": 2,
        "medicine": 2, "drug": 2, "dose": 2, "dosage": 2, "therapy": 2, "clinical": 2,
        "infection": 2, "tumor": 2, "blood": 1, "heart": 1, "cancer": 2, "fever": 2,
        "prescription": 2, "surgery": 2, "medical": 2, "pain": 1, "antibiotic": 2,
        "pathology": 2,
    },
    Domain.DESIGN: {
        "design": 2, "layout": 2, "ui": 2, "ux": 2, "color": 1, "typography": 2,
        "wireframe": 2, "logo": 2, "brand": 2, "aesthetic": 2, "interface": 1,
        "palette": 2, "font": 2, "visual": 1, "prototype": 1, "usability": 2,
        "composition": 1, "spacing": 1, "mockup": 2, "accessibility": 1,
    },
    Domain.LAW: {
        "law": 2, "legal": 2, "contract": 2, "liability": 2, "court": 2, "statute": 2,
        "clause": 2, "lawsuit": 2, "plaintiff": 2, "defendant": 2, "jurisdiction": 2,
        "regulation": 2, "compliance": 2, "tort": 2, "patent": 1, "copyright": 2,
        "negligence": 2, "rights": 1, "breach": 1, "litigation": 2, "constitutional": 2,
        "attorney": 2,
    },
}

# Domains whose answers must be grounded (cite or abstain) and carry a caveat.
_KNOWLEDGE_HEAVY = frozenset({Domain.MEDICINE, Domain.LAW, Domain.SCIENCE})

# The expert persona and ordered methodology each domain reasons through.
_METHODOLOGY: Dict[Domain, Tuple[str, Tuple[str, ...]]] = {
    Domain.CODING: ("a senior software engineer", (
        "Restate the requirement and the exact inputs/outputs and edge cases.",
        "Design the approach; pick data structures and note complexity.",
        "Implement clean, correct code.",
        "Run it on representative inputs and the edge cases — actually execute, don't assume.",
        "Verify the output, fix failures, and state the remaining limitations.",
    )),
    Domain.MATH: ("a rigorous mathematician", (
        "Identify what is given and what must be found.",
        "Choose an exact method (symbolic algebra, calculus, logic, number theory).",
        "Solve step by step, keeping every step reversible.",
        "Verify the result with an independent check or by substitution.",
        "State the answer exactly; flag any assumption that was needed.",
    )),
    Domain.SCIENCE: ("a careful research scientist", (
        "State the question and the relevant principles.",
        "Form a falsifiable hypothesis with a measurable prediction.",
        "Design and run the smallest experiment or calculation that could refute it.",
        "Compare result to prediction: supported, refuted, or inconclusive.",
        "Conclude with calibrated confidence and the evidence behind it.",
    )),
    Domain.BUSINESS: ("a sharp business strategist", (
        "Name the real objective and the constraints.",
        "Stress-test the central premise — is the assumed market/behaviour real?",
        "Find the leverage point, not the comfortable compromise.",
        "Lay out concrete, ordered execution steps with the key metric for each.",
        "Name the worst case and the cheapest way to reverse course.",
    )),
    Domain.ROBOTICS: ("a robotics and control engineer", (
        "Define the task, the embodiment, the sensors and the actuators.",
        "Model the dynamics: state, action, and the reward/cost.",
        "Plan a trajectory or control law; roll it out before committing.",
        "Account for sensing noise, latency, and safety limits.",
        "Specify the failure behaviour and the safe-stop condition.",
    )),
    Domain.MEDICINE: ("a cautious clinical reasoner", (
        "Gather the presentation: symptoms, history, and red flags.",
        "Build a differential — the plausible causes, most-likely first.",
        "Ground each statement in cited evidence; do not assert beyond it.",
        "Flag urgent red flags that need immediate in-person care.",
        "Add a clear caveat: this is information, not a diagnosis — consult a clinician.",
    )),
    Domain.DESIGN: ("a principled product designer", (
        "Clarify the user, the goal, and the context of use.",
        "State the constraints and the design principles that apply.",
        "Propose a concrete structure (hierarchy, layout, flow).",
        "Justify each choice against usability and accessibility.",
        "Give the next iteration step and what to test with users.",
    )),
    Domain.LAW: ("a precise legal analyst", (
        "Issue — state the precise legal question.",
        "Rule — the relevant statute/principle, grounded in cited sources.",
        "Application — apply the rule to these facts, both sides.",
        "Conclusion — the reasoned answer, with its uncertainty.",
        "Add a clear caveat: this is general information, not legal advice — consult a lawyer.",
    )),
    Domain.GENERAL: ("a first-principles generalist", (
        "Strip the problem to its first principles and unknowns.",
        "Map it onto the nearest understood domain by analogy.",
        "Form a testable approach and try the smallest version of it.",
        "Check the result against reality; keep what survives.",
        "Record what was learned so the new field is understood next time.",
    )),
}

# Which existing engine each domain's solver leans on (advisory label + real dispatch).
_ENGINE: Dict[Domain, str] = {
    Domain.CODING: "code_sandbox",
    Domain.MATH: "faculties",
    Domain.SCIENCE: "scientist",
    Domain.BUSINESS: "strategic",
    Domain.ROBOTICS: "world_model",
    Domain.MEDICINE: "rag",
    Domain.DESIGN: "council",
    Domain.LAW: "rag",
    Domain.GENERAL: "adaptive",
}


# --------------------------------------------------------------------------- #
# Domain objects
# --------------------------------------------------------------------------- #
@dataclass
class DomainFrame:
    """The advisory framing for a problem: its domain, the expert lens, the engine, and
    the honesty/safety constraints. Prepended to the reasoning context every turn."""
    problem: str
    domain: Domain = Domain.GENERAL
    confidence: float = 0.0
    novel: bool = False
    persona: str = ""
    methodology: List[str] = field(default_factory=list)
    engine: str = ""
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "confidence": round(float(self.confidence), 3),
            "novel": self.novel,
            "persona": self.persona,
            "methodology": list(self.methodology),
            "engine": self.engine,
            "constraints": list(self.constraints),
        }

    def to_prompt_text(self) -> str:
        """Render the frame as a context block the reasoner reads at the top of its memory."""
        tag = f"{self.domain.value}{' · novel field' if self.novel else ''}"
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.methodology, 1))
        cons = ("\n  Constraints: " + "; ".join(self.constraints)) if self.constraints else ""
        return (f"[Domain expertise — {tag} (confidence {self.confidence:.0%})]\n"
                f"  Reason as {self.persona}. Work the problem like this:\n{steps}{cons}")


@dataclass
class DomainSolution:
    """The result of running a domain expert end-to-end."""
    problem: str
    domain: Domain = Domain.GENERAL
    confidence: float = 0.5
    novel: bool = False
    methodology: List[str] = field(default_factory=list)
    engine: str = ""
    answer: str = ""
    verified: bool = False
    abstained: bool = False
    citations: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "domain": self.domain.value,
            "confidence": round(float(self.confidence), 3),
            "novel": self.novel,
            "methodology": list(self.methodology),
            "engine": self.engine,
            "answer": self.answer,
            "verified": self.verified,
            "abstained": self.abstained,
            "citations": list(self.citations),
            "detail": dict(self.detail),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class DomainClassifier:
    """Map a problem to a :class:`Domain` with a confidence in ``[0, 1]``.

    Deterministic keyword scoring runs first and always works. ``learned`` profiles
    (novel fields NYXARA has met before) are scored alongside the built-ins, so an
    adapted field is recognised next time. When the top score is weak and an LLM is
    available, an optional refine pass breaks ties; otherwise the result stands.
    """

    def __init__(self, *, threshold: float = 0.18, llm: Any = None,
                 use_llm_refine: bool = True) -> None:
        self.threshold = float(threshold)
        self.llm = llm
        self.use_llm_refine = bool(use_llm_refine)
        # label -> {keyword: weight}; learned novel domains accrete here.
        self.learned: Dict[str, Dict[str, int]] = {}

    def learn(self, label: str, problem: str) -> None:
        """Remember a novel field by its distinctive terms so it is recognised again."""
        label = (label or "").strip().lower()
        if not label:
            return
        bag = self.learned.setdefault(label, {})
        for tok in _tokens(problem):
            if len(tok) >= 4:
                bag[tok] = min(2, bag.get(tok, 0) + 1)

    def _score(self, toks: List[str], signals: Dict[str, int]) -> int:
        return sum(signals.get(t, 0) for t in toks)

    @staticmethod
    def _structural(problem: str) -> Dict[Domain, int]:
        """Signals from structure, not vocabulary: a bare ``2+3*4`` is maths, a fenced code
        block is coding, even with no keyword present."""
        boost: Dict[Domain, int] = {}
        # maths: arithmetic expression, an equation, or a percentage
        if (re.search(r"\d+\s*[-+*/^]\s*\d+", problem)
                or re.search(r"[a-zA-Z0-9]\s*=\s*[a-zA-Z0-9]", problem)
                or re.search(r"\d+\s*%", problem)
                or re.search(r"\b(derivative|integral)\b", problem.lower())):
            boost[Domain.MATH] = boost.get(Domain.MATH, 0) + 3
        # coding: a fenced block or obvious source tokens
        if (re.search(r"```", problem)
                or re.search(r"\b(def|import|class|return|print)\s", problem)
                or re.search(r"[(){};]", problem) and re.search(r"\b(function|var|let|const)\b",
                                                                problem.lower())):
            boost[Domain.CODING] = boost.get(Domain.CODING, 0) + 3
        return boost

    def classify(self, problem: str) -> Tuple[Domain, float, bool, Optional[str]]:
        """Return ``(domain, confidence, novel, learned_label)``.

        ``novel`` is True when nothing clears the threshold (a candidate new field).
        ``learned_label`` is set when a previously-learned novel field matched best.
        """
        toks = _tokens(problem)
        if not toks:
            return Domain.GENERAL, 0.0, True, None

        scores: Dict[Domain, int] = {
            d: self._score(toks, sig) for d, sig in _SIGNALS.items()
        }
        for d, b in self._structural(problem).items():
            scores[d] = scores.get(d, 0) + b
        best = max(scores, key=lambda d: scores[d])
        best_raw = scores[best]

        # learned novel fields compete on the same footing
        learned_best, learned_raw = None, 0
        for label, sig in self.learned.items():
            raw = self._score(toks, sig)
            if raw > learned_raw:
                learned_best, learned_raw = label, raw

        # confidence: matched weight as a fraction of a small reference mass
        denom = max(4, len(toks))
        if learned_raw > best_raw:
            conf = min(0.95, learned_raw / denom)
            return Domain.GENERAL, conf, False, learned_best

        conf = min(0.95, best_raw / denom)
        if best_raw == 0 or conf < self.threshold:
            refined = self._refine(problem) if self.use_llm_refine else None
            if refined is not None:
                return refined, 0.55, False, None
            return Domain.GENERAL, conf, True, None
        return best, conf, False, None

    def _refine(self, problem: str) -> Optional[Domain]:
        """Break a weak/ambiguous call with the LLM, if a real provider is available."""
        if not _llm_real(self.llm):
            return None
        names = ", ".join(d.value for d in Domain if d is not Domain.GENERAL)
        prompt = (f"Classify this problem into exactly one domain from: {names}, or 'general' "
                  f"if none fit.\n\nProblem: {problem}\n\nReply with only the one word.")
        try:
            out = (self.llm.generate(prompt, system=_CLASSIFY_SYSTEM,
                                     temperature=0.0, max_tokens=8) or "").strip().lower()
        except Exception:  # noqa: BLE001
            return None
        for d in Domain:
            if d.value in out and d is not Domain.GENERAL:
                return d
        return None


# --------------------------------------------------------------------------- #
# Experts — one per domain, each bound to a real existing engine
# --------------------------------------------------------------------------- #
class DomainExpert:
    """Base expert: holds the persona/methodology/engine for a domain and a default
    ``solve`` that synthesises a structured answer (LLM when present, else offline).
    Concrete subclasses override ``solve`` to drive their bound engine for real."""

    domain: Domain = Domain.GENERAL

    def __init__(self, gi: "GeneralIntelligence") -> None:
        self.gi = gi

    # ---- framing (cheap, advisory) ---- #
    def constraints(self) -> List[str]:
        if self.domain in _KNOWLEDGE_HEAVY:
            return ["ground every claim in a cited source",
                    "abstain honestly when the evidence is missing — do not confabulate",
                    "add the professional-consultation caveat"]
        if self.domain is Domain.CODING:
            return ["actually run the code and report real output, not a guess"]
        if self.domain is Domain.MATH:
            return ["prefer an exact, verifiable method over a probabilistic guess"]
        return []

    def frame(self, problem: str, confidence: float, novel: bool = False) -> DomainFrame:
        persona, steps = _METHODOLOGY.get(self.domain, _METHODOLOGY[Domain.GENERAL])
        return DomainFrame(problem=problem, domain=self.domain, confidence=confidence,
                           novel=novel, persona=persona, methodology=list(steps),
                           engine=_ENGINE.get(self.domain, "council"),
                           constraints=self.constraints())

    # ---- default end-to-end solve (overridden by most subclasses) ---- #
    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        sol.answer = self.gi._synthesise(problem, frame)
        sol.confidence = 0.5 if self.gi._llm_real() else 0.35
        return sol


class CodingExpert(DomainExpert):
    domain = Domain.CODING

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        code = self.gi._extract_code(problem)
        if not code and self.gi._llm_real():
            code = self.gi._generate_code(problem)
        if not code:
            sol.answer = self.gi._synthesise(problem, frame)
            sol.confidence = 0.3
            sol.detail["note"] = "no model available to synthesise code; gave an approach"
            return sol
        run = self.gi._run_code(code)
        sol.detail["code"] = code
        sol.detail["execution"] = run
        if run.get("ok"):
            out = (run.get("stdout") or "").strip()
            val = run.get("value")
            sol.answer = (f"Code ran successfully.\n\n```python\n{code}\n```\n\n"
                          f"Output:\n{out or val or '(no stdout; see value)'}")
            sol.verified = True
            sol.confidence = 0.8
        else:
            sol.answer = (f"Code did not run cleanly — here is the attempt and the error so it "
                          f"can be fixed.\n\n```python\n{code}\n```\n\nError:\n"
                          f"{(run.get('error') or run.get('stderr') or '').strip()}")
            sol.confidence = 0.4
        return sol


class MathExpert(DomainExpert):
    domain = Domain.MATH

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        exact = self.gi._faculty_solve(problem)
        if exact is not None:
            answer, conf = exact
            sol.answer = f"{answer}"
            sol.verified = True
            sol.confidence = max(0.7, float(conf))
            sol.detail["method"] = "verifiable_faculty"
            return sol
        # no decidable method matched — synthesise, but say it is not certified
        sol.answer = self.gi._synthesise(problem, frame)
        sol.confidence = 0.45 if self.gi._llm_real() else 0.3
        sol.detail["method"] = "reasoned (not exactly verified)"
        return sol


class ScienceExpert(DomainExpert):
    domain = Domain.SCIENCE

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        report = self.gi._investigate(problem)
        if report is not None:
            sol.detail["investigation"] = report
            concl = report.get("conclusion") or {}
            verdict = concl.get("verdict", "inconclusive")
            reasoning = concl.get("reasoning", "")
            sol.answer = f"Verdict: {verdict}. {reasoning}".strip()
            sol.confidence = float(concl.get("confidence", 0.5))
            sol.verified = verdict in ("supported", "refuted")
        else:
            sol.answer = self.gi._synthesise(problem, frame)
            sol.confidence = 0.4
        # science is knowledge-heavy: ground/caveat the answer
        self.gi._apply_grounding(problem, frame, sol)
        return sol


class BusinessExpert(DomainExpert):
    domain = Domain.BUSINESS

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        analysis = self.gi._strategize(problem)
        if analysis is not None:
            sol.detail["analysis"] = analysis
            sol.answer = (analysis.get("direct_answer", "") + "\n\nOptimised: "
                          + analysis.get("optimized_solution", "")).strip()
            sol.confidence = float(analysis.get("confidence", 0.5))
        else:
            sol.answer = self.gi._synthesise(problem, frame)
            sol.confidence = 0.45
        return sol


class RoboticsExpert(DomainExpert):
    domain = Domain.ROBOTICS

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        sol.answer = self.gi._synthesise(problem, frame)
        sol.confidence = 0.5 if self.gi._llm_real() else 0.35
        sol.detail["world_model"] = "available" if self.gi.world_model is not None else "absent"
        return sol


class GroundedExpert(DomainExpert):
    """Medicine and law: answer only from cited evidence; abstain + caveat otherwise."""

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        sol.answer = self.gi._synthesise(problem, frame) if self.gi._llm_real() else ""
        sol.confidence = 0.4
        self.gi._apply_grounding(problem, frame, sol)
        return sol


class MedicineExpert(GroundedExpert):
    domain = Domain.MEDICINE


class LawExpert(GroundedExpert):
    domain = Domain.LAW


class DesignExpert(DomainExpert):
    domain = Domain.DESIGN

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        sol.answer = self.gi._synthesise(problem, frame)
        sol.confidence = 0.5 if self.gi._llm_real() else 0.35
        return sol


class AdaptiveExpert(DomainExpert):
    """Novel fields: reason from first principles, test the smallest version, and *learn*
    a profile so the field is recognised — and improves — next time."""

    domain = Domain.GENERAL

    def solve(self, problem: str, frame: DomainFrame) -> DomainSolution:
        sol = self.gi._blank_solution(problem, frame)
        principles = self.gi._first_principles(problem)
        sol.detail["first_principles"] = principles
        report = self.gi._investigate(problem)
        if report is not None:
            sol.detail["investigation"] = report
        # OWN FACULTIES FIRST — before reaching for the base LLM, try to generalize the new
        # field herself. The unified cascade (skill-induction from any examples in the prompt,
        # relational transfer, open-world law modelling) is tried first; it subsumes bare transfer.
        # When it fires, the reasoning content is hers; the LLM is not consulted. Honest: each
        # stage declines when it cannot answer, so bare transfer and then the LLM still follow.
        transferred = None
        if self.gi.own_faculties_first:
            transferred = self.gi._generalize_solve(problem) or self.gi._transfer_solve(problem)
        if transferred is not None:
            answer, conf, detail = transferred
            sol.answer = answer
            sol.confidence = conf
            sol.detail.update(detail)
        else:
            sol.answer = self.gi._synthesise(problem, frame) or (
                "From first principles: " + "; ".join(principles))
            sol.confidence = 0.4 if self.gi._llm_real() else 0.3
        # adapt: learn this field so it is recognised next time (Rule 4 — capability grows)
        label = self.gi._label_for(problem)
        if label:
            self.gi.learn_domain(label, problem)
            sol.detail["learned_field"] = label
        return sol


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #
class GeneralIntelligence:
    """Domain-aware general problem solver.

    Composes the mind's existing real engines and routes each problem to the right one.
    Every dependency is optional; with none of them it still classifies, frames and
    answers deterministically and offline.
    """

    def __init__(self, *, reasoner: Any = None, council: Any = None, scientist: Any = None,
                 world_model: Any = None, memory: Any = None, knowledge: Any = None,
                 tools: Any = None, strategic: Any = None, self_model: Any = None,
                 transfer_engine: Any = None, generalization_engine: Any = None,
                 llm: Any = None, threshold: float = 0.18,
                 use_llm_refine: bool = True, allow_web_grounding: bool = True,
                 auto_discover: bool = True, own_faculties_first: bool = True) -> None:
        self.reasoner = reasoner
        self.council = council
        self.scientist = scientist
        self.world_model = world_model
        self.memory = memory
        self.knowledge = knowledge
        self.tools = tools
        self.strategic = strategic
        self.self_model = self_model
        # her own cross-domain generalizer (mind/transfer.py) — used to reason about a NOVEL
        # field herself (structure-mapping from a known domain) before falling to the base LLM
        self.transfer_engine = transfer_engine
        # her unified own-faculty generalizer (mind/generalization.py) — the full cascade
        # (skill-induction from examples, relational transfer, open-world modelling), used to solve
        # a novel field herself before reaching for the base LLM
        self.generalization_engine = generalization_engine
        self.own_faculties_first = bool(own_faculties_first)
        self.llm = llm if llm is not None else getattr(reasoner, "llm", None)
        self.allow_web_grounding = bool(allow_web_grounding)
        self.auto_discover = bool(auto_discover)
        self.classifier = DomainClassifier(threshold=threshold, llm=self.llm,
                                           use_llm_refine=use_llm_refine)
        self._experts: Dict[Domain, DomainExpert] = {
            Domain.CODING: CodingExpert(self),
            Domain.MATH: MathExpert(self),
            Domain.SCIENCE: ScienceExpert(self),
            Domain.BUSINESS: BusinessExpert(self),
            Domain.ROBOTICS: RoboticsExpert(self),
            Domain.MEDICINE: MedicineExpert(self),
            Domain.DESIGN: DesignExpert(self),
            Domain.LAW: LawExpert(self),
            Domain.GENERAL: AdaptiveExpert(self),
        }
        self._frames: List[DomainFrame] = []

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def frame(self, problem: str) -> DomainFrame:
        """Classify + frame a problem (cheap, advisory). Always returns a frame."""
        problem = (problem or "").strip()
        domain, conf, novel, _label = self.classifier.classify(problem)
        expert = self._experts.get(domain, self._experts[Domain.GENERAL])
        fr = expert.frame(problem, conf, novel=novel)
        self._frames.append(fr)
        return fr

    def solve(self, problem: str) -> Dict[str, Any]:
        """Classify, route to the bound engine, and run it end-to-end. Returns a dict."""
        t0 = time.monotonic()
        problem = (problem or "").strip()
        domain, conf, novel, _label = self.classifier.classify(problem)
        expert = self._experts.get(domain, self._experts[Domain.GENERAL])
        fr = expert.frame(problem, conf, novel=novel)
        try:
            sol = expert.solve(problem, fr)
        except Exception as exc:  # noqa: BLE001 — a solve never crashes a turn
            sol = self._blank_solution(problem, fr)
            sol.answer = f"Could not complete the {domain.value} solve cleanly ({exc})."
            sol.confidence = 0.2
        sol.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return sol.to_dict()

    def learn_domain(self, label: str, problem: str) -> None:
        """Persist a learned novel field: recognised by the classifier next time, and
        written to long-term memory so it survives restarts (best-effort)."""
        if not self.auto_discover:
            return
        self.classifier.learn(label, problem)
        if self.memory is not None:
            try:
                from nyxara.memory.store import MemoryType
                self.memory.remember(
                    f"Learned to recognise the '{label}' field from: {problem[:160]}",
                    mem_type=MemoryType.SEMANTIC, tags=["domain_profile", label],
                    importance=0.6)
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass

    def all_frames(self) -> List[DomainFrame]:
        return list(self._frames)

    # ---------------------------------------------------------------------- #
    # Engine bindings (each reuses an existing real subsystem)
    # ---------------------------------------------------------------------- #
    def _faculty_solve(self, problem: str) -> Optional[Tuple[str, float]]:
        """Exact math/logic via the verifiable faculties (mind/reasoning_faculties)."""
        try:
            from nyxara.mind.reasoning_faculties import solve_with_faculties
            return solve_with_faculties(problem)
        except Exception:  # noqa: BLE001
            return None

    def _ensure_transfer(self) -> Any:
        """The relational-transfer engine (built lazily if none was injected)."""
        if self.transfer_engine is None:
            try:
                from nyxara.mind.transfer import RelationalTransferEngine
                self.transfer_engine = RelationalTransferEngine()
            except Exception:  # noqa: BLE001 — transfer is advisory
                self.transfer_engine = None
        return self.transfer_engine

    def _transfer_solve(self, problem: str) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """Reason about a NOVEL field herself: structure-map it onto a domain she already
        understands and project the known structure across. Returns ``(answer, confidence,
        detail)`` — the answer is derived by NYXARA's own structure-mapper, not the base LLM —
        or ``None`` when no structure maps (the caller then falls to the LLM)."""
        eng = self._ensure_transfer()
        if eng is None:
            return None
        try:
            tr = eng.generalize(problem)
        except Exception:  # noqa: BLE001 — transfer is advisory, never fatal
            return None
        if tr is None:
            return None
        conf = min(0.85, 0.5 + 0.1 * float(tr.structural_score))
        # a loose (analogical) transfer maps a new domain's own vocabulary onto a known base by
        # structure alone — her own reasoning, but a conjecture, so its confidence is held down
        if getattr(tr, "analogical", False):
            conf = min(conf, 0.6)
        return tr.render(), conf, {"transfer": tr.to_dict(), "own_faculty": "relational_transfer",
                                   "analogical": bool(getattr(tr, "analogical", False))}

    def _ensure_generalization(self) -> Any:
        """The unified own-faculty cascade (built lazily if none was injected).

        Composes her exact faculties + first-principles, skill-induction from in-prompt demos,
        relational transfer, from-scratch domain-genesis, and open-world law modelling — so the
        own-faculty path is populated *by default*, not only when an orchestrator injects it. This
        is what makes the base LLM a genuine last resort rather than the first thing reached."""
        if self.generalization_engine is None:
            try:
                from nyxara.mind.generalization import GeneralizationEngine
                self.generalization_engine = GeneralizationEngine(
                    transfer_engine=self._ensure_transfer())
            except Exception:  # noqa: BLE001 — the cascade is a capability, never required
                self.generalization_engine = None
        return self.generalization_engine

    def _generalize_solve(self, problem: str) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """Solve a novel field with her unified own-faculty cascade (mind/generalization.py):
        exact faculties + first-principles, skill-induction from any examples in the prompt,
        relational transfer, from-scratch domain-genesis, open-world law modelling. Returns
        ``(answer, confidence, detail)`` — content derived by her own faculties, never the base
        LLM — or ``None`` when none of them can honestly answer."""
        eng = self._ensure_generalization()
        if eng is None:
            return None
        try:
            res = eng.generalize(problem)
        except Exception:  # noqa: BLE001 — the cascade is advisory, never fatal
            return None
        if res is None:
            return None
        return res.render(), float(res.confidence), {
            "own_faculty": res.source, "generalization": res.to_dict(),
            "analogical": bool(res.analogical)}

    def _extract_code(self, problem: str) -> str:
        """Pull a fenced ```python code block out of the problem, if present."""
        m = re.search(r"```(?:python|py)?\s*(.+?)```", problem, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _generate_code(self, problem: str) -> str:
        """Ask the LLM for a runnable Python solution; strip fences. Returns '' on failure."""
        prompt = (f"Write a single self-contained Python program that solves this. Print the "
                  f"result to stdout. Return ONLY code, no prose.\n\nTask: {problem}")
        try:
            out = self.llm.generate(prompt, system=_CODE_SYSTEM, temperature=0.2,
                                    max_tokens=700) or ""
        except Exception:  # noqa: BLE001
            return ""
        m = re.search(r"```(?:python|py)?\s*(.+?)```", out, re.DOTALL | re.IGNORECASE)
        return (m.group(1) if m else out).strip()

    def _run_code(self, code: str) -> Dict[str, Any]:
        """Execute code in the real sandbox and return its outcome as data."""
        try:
            from nyxara.agency.code_sandbox import run_python
            return run_python(code, timeout_s=5.0).to_dict()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"sandbox unavailable: {exc}"}

    def _investigate(self, problem: str) -> Optional[Dict[str, Any]]:
        """Run the Scientist's hypothesis loop on the problem."""
        if self.scientist is None:
            return None
        try:
            return self.scientist.investigate(problem).to_dict()
        except Exception:  # noqa: BLE001
            return None

    def _strategize(self, problem: str) -> Optional[Dict[str, Any]]:
        """Run the six-part strategic framework (business / general analysis)."""
        if self.strategic is None:
            return None
        try:
            return self.strategic.analyze(problem).to_dict()
        except Exception:  # noqa: BLE001
            return None

    def _first_principles(self, problem: str) -> List[str]:
        """Borrow the strategic faculty's first-principles pass, else a local heuristic."""
        if self.strategic is not None:
            try:
                return self.strategic._first_principles(problem)  # noqa: SLF001 (intentional reuse)
            except Exception:  # noqa: BLE001
                pass
        return ["the problem's framing is itself untested; reduce it to what is actually known"]

    def _apply_grounding(self, problem: str, frame: DomainFrame, sol: DomainSolution) -> None:
        """Knowledge-heavy domains: gather cited evidence (knowledge + governed web),
        abstain honestly when nothing is found, and always append the caveat."""
        evidence: List[str] = []
        citations: List[str] = []
        # 1) internal knowledge base
        if self.knowledge is not None:
            try:
                for ch in self.knowledge.retrieve(problem, k=4):
                    evidence.append(getattr(ch, "text", ""))
                    src = getattr(ch, "source", "")
                    if src:
                        citations.append(src)
            except Exception:  # noqa: BLE001
                pass
        # 2) governed web grounding (still passes the permission/governor pipeline)
        if self.allow_web_grounding and self.tools is not None:
            web = self._web_search(problem)
            for hit in web:
                title = hit.get("title") or hit.get("url") or ""
                if hit.get("snippet"):
                    evidence.append(hit["snippet"])
                if hit.get("url"):
                    citations.append(hit["url"])
                elif title:
                    citations.append(title)
        sol.citations = sorted({c for c in citations if c})
        if not sol.citations and not sol.answer:
            # nothing grounded and no model spoke — abstain honestly
            sol.abstained = True
            sol.answer = ("I don't have grounded sources for this, so I won't guess. "
                          "Please consult a qualified professional.")
            sol.confidence = min(sol.confidence, 0.2)
        elif sol.citations and self._llm_real():
            # re-synthesise the answer strictly from the gathered evidence
            grounded = self._synthesise_grounded(problem, frame, evidence)
            if grounded:
                sol.answer = grounded
                sol.confidence = max(sol.confidence, 0.55)
        # the caveat is non-negotiable for knowledge-heavy domains
        caveat = _CAVEAT.get(frame.domain)
        if caveat and caveat not in sol.answer:
            sol.answer = (sol.answer + "\n\n" + caveat).strip()

    def _web_search(self, query: str) -> List[Dict[str, Any]]:
        """Best-effort governed web search via the tool registry."""
        try:
            if self.tools.get("web_search") is None:
                return []
            from nyxara.agency.permissions import Authority
            res = self.tools.invoke("web_search", {"query": query, "max_results": 4},
                                    authority=Authority.AUTONOMOUS)
            if getattr(res, "ok", False) and isinstance(res.value, list):
                return [h for h in res.value if isinstance(h, dict)]
        except Exception:  # noqa: BLE001 — web grounding is best-effort
            pass
        return []

    # ---------------------------------------------------------------------- #
    # Synthesis helpers
    # ---------------------------------------------------------------------- #
    def _synthesise(self, problem: str, frame: DomainFrame) -> str:
        """A domain-framed answer, own-faculties FIRST and the base LLM only as a last resort.

        Order (the whole point of the last-resort rewiring): (1) try NYXARA's own unified
        generalizer — exact faculties, first-principles derivation, skill-induction, relational
        transfer, from-scratch domain-genesis — so a problem she *can* reason about herself never
        reaches the LLM; (2) only if every own faculty declines, consult the base LLM, and pass its
        output through :meth:`_verify_answer` so the verdict is hers, not the model's; (3) with no
        LLM, a structured offline answer."""
        own = self._own_faculty_answer(problem)
        if own is not None:
            return own
        if self._llm_real():
            steps = " ".join(f"{i}) {s}" for i, s in enumerate(frame.methodology, 1))
            prompt = (f"Solve this as {frame.persona}. Method: {steps}\n\nProblem: {problem}\n\n"
                      "Give a direct, high-signal answer.")
            try:
                out = self.llm.generate(prompt, system=_SOLVE_SYSTEM, temperature=0.3,
                                        max_tokens=600)
                if out and out.strip():
                    return self._verify_answer(problem, out.strip())
            except Exception:  # noqa: BLE001
                pass
        return self._synthesise_offline(problem, frame)

    def _own_faculty_answer(self, problem: str) -> Optional[str]:
        """The answer from NYXARA's own faculties (cascade → bare transfer), or ``None`` to defer.

        Reused everywhere the LLM would otherwise be reached, so *every* domain — not just the
        novel-field path — prefers her own verified/structural reasoning before the base model."""
        got = self._generalize_solve(problem) or self._transfer_solve(problem)
        return got[0] if got is not None else None

    def _verify_answer(self, problem: str, answer: str) -> str:
        """Cross-check the base LLM's answer with NYXARA's OWN exact faculties before it stands.

        Any concrete arithmetic equality the model asserts ("2 + 3 * 4 = 20") is recomputed
        exactly; a false one is flagged and corrected, a correct one is endorsed — so even when the
        LLM drafts, the *verdict* is hers. Free of a checkable claim, the answer passes through
        unchanged (nothing to verify, so nothing is asserted)."""
        notes: List[str] = []
        for lhs, claimed in self._checkable_equalities(answer):
            actual = self._eval_arith(lhs)
            if actual is None:
                continue
            if abs(actual - claimed) < 1e-9:
                notes.append(f"✓ verified {lhs.strip()} = {_fmt_num(actual)}")
            else:
                notes.append(f"⚠ corrected: {lhs.strip()} = {_fmt_num(actual)}, not {_fmt_num(claimed)}")
        if notes:
            return answer + "\n\n[NYXARA self-check — her own faculties] " + "; ".join(notes)
        return answer

    @staticmethod
    def _checkable_equalities(text: str) -> List[Tuple[str, float]]:
        """Recover ``<arithmetic> = <number>`` claims from prose (numbers and operators only)."""
        out: List[Tuple[str, float]] = []
        for m in re.finditer(r"([0-9][0-9\s+\-*/().^%]*[0-9)])\s*=\s*(-?\d+(?:\.\d+)?)", text):
            lhs = m.group(1)
            if not re.search(r"[-+*/^%]", lhs):    # need a real operation, not "3 = 3"
                continue
            try:
                out.append((lhs, float(m.group(2))))
            except ValueError:
                continue
        return out

    def _eval_arith(self, expr: str) -> Optional[float]:
        """Evaluate a pure arithmetic expression exactly via the verifiable MathEngine."""
        try:
            from nyxara.mind.math import MathEngine
            return float(MathEngine().evaluate(expr.replace("^", "**")))
        except Exception:  # noqa: BLE001
            return None

    def _synthesise_grounded(self, problem: str, frame: DomainFrame,
                             evidence: List[str]) -> str:
        """Answer strictly from the gathered evidence (no outside facts)."""
        ev = "\n".join(f"- {e[:400]}" for e in evidence if e)[:3000]
        prompt = (f"Answer the question using ONLY the evidence below. If the evidence does not "
                  f"cover it, say so. Cite which evidence supports each claim.\n\n"
                  f"Question: {problem}\n\nEvidence:\n{ev}")
        try:
            out = self.llm.generate(prompt, system=_SOLVE_SYSTEM, temperature=0.2,
                                    max_tokens=600)
            return (out or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _synthesise_offline(self, problem: str, frame: DomainFrame) -> str:
        """A deterministic, structurally-faithful answer when no model is available."""
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(frame.methodology, 1))
        return (f"Approaching as {frame.persona} (no model loaded, so this is the method, "
                f"not a finished answer):\n{steps}")

    def _blank_solution(self, problem: str, frame: DomainFrame) -> DomainSolution:
        return DomainSolution(problem=problem, domain=frame.domain, novel=frame.novel,
                              methodology=list(frame.methodology), engine=frame.engine)

    def _label_for(self, problem: str) -> str:
        """A short label for a novel field, from its most distinctive long token."""
        for tok in _tokens(problem):
            if (len(tok) >= 5 and tok not in _STOPWORDS
                    and not any(tok in sig for sig in _SIGNALS.values())):
                return tok
        for tok in _tokens(problem):
            if tok not in _STOPWORDS:
                return tok
        return ""

    def _llm_real(self) -> bool:
        return _llm_real(self.llm)


# --------------------------------------------------------------------------- #
# Module helpers / constants
# --------------------------------------------------------------------------- #
def _fmt_num(v: float) -> str:
    """Render a number cleanly — an integral float prints without the trailing ``.0``."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


def _llm_real(llm: Any) -> bool:
    """True only when a genuine (non-mock) provider is available."""
    if llm is None:
        return False
    try:
        return llm.chosen_provider().name != "mock"
    except Exception:  # noqa: BLE001
        return False


_CAVEAT: Dict[Domain, str] = {
    Domain.MEDICINE: ("⚕️ This is general information, not a medical diagnosis. "
                      "Consult a qualified clinician for any medical decision."),
    Domain.LAW: ("⚖️ This is general information, not legal advice. "
                 "Consult a qualified lawyer for your specific situation."),
}

_STOPWORDS = frozenset({
    "what", "this", "that", "with", "from", "have", "about", "which", "would", "could",
    "should", "there", "their", "where", "when", "will", "into", "over", "under", "tune",
    "make", "give", "tell", "help", "please", "thing", "things", "some", "many", "much",
    "the", "and", "for", "are", "how", "why", "who", "your", "you",
})

_CLASSIFY_SYSTEM = ("You are NYXARA's domain classifier. Reply with exactly one lowercase "
                    "domain word and nothing else.")
_CODE_SYSTEM = ("You are NYXARA's coding faculty. Write correct, self-contained Python. "
                "Return only code, in a single block.")
_SOLVE_SYSTEM = ("You are NYXARA, reasoning as a domain expert. Optimise for truth and "
                 "usefulness over comfort. Be concise and high-signal. Never fabricate "
                 "facts or sources; if you are unsure, say so.")


# --------------------------------------------------------------------------- #
# Hermetic self-test — no LLM, no network, no external deps.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    gi = GeneralIntelligence()  # fully offline

    checks = {
        "write a python function to reverse a list": Domain.CODING,
        "solve the equation 2*x + 3 = 7 for x": Domain.MATH,
        "design a hypothesis to test photosynthesis rate": Domain.SCIENCE,
        "what pricing strategy maximises startup revenue": Domain.BUSINESS,
        "plan a robot arm trajectory with a servo motor": Domain.ROBOTICS,
        "patient has fever and cough, what is the differential diagnosis": Domain.MEDICINE,
        "design a clean ui layout with good typography": Domain.DESIGN,
        "is this contract clause a breach of liability": Domain.LAW,
    }
    for text, want in checks.items():
        fr = gi.frame(text)
        assert fr.domain is want, f"{text!r} -> {fr.domain} (wanted {want})"
        assert fr.methodology, "every frame must carry a methodology"
    # knowledge-heavy domains carry the caveat constraint
    assert any("caveat" in c for c in gi.frame("dosage of a drug").constraints)

    # a real verifiable math solve
    out = gi.solve("what is 2 + 3 * 4")
    assert out["domain"] == "math" and out["verified"], out

    # novel field → general + learned, recognised next time
    novel = gi.frame("optimise the flux capacitor resonance in a chrono-drive")
    assert novel.novel, novel.to_dict()
    gi.solve("tune the chrono-drive resonance")  # learns "chrono"/"resonance"-ish label
    assert gi.classifier.learned, "a novel field must be learned"

    # coding solve actually executes
    code_out = gi.solve("```python\nresult = sum(range(5))\nprint(result)\n```")
    assert code_out["domain"] == "coding" and code_out["verified"], code_out

    print("general_intelligence.py self-test passed.")
