"""Tests for nyxara.mind.strategic — the Strategic Intelligence faculty.

The faculty reasons any problem through a fixed six-part framework (direct answer →
reality check → weaknesses → root cause → optimised solution → execution steps). The whole
analysis runs offline (no LLM, no network), with deterministic fallbacks, so these tests
are hermetic and reproducible.
"""

from __future__ import annotations

from nyxara.mind.strategic import (
    PremiseCheck,
    Soundness,
    StrategicAnalysis,
    StrategicIntelligence,
    Weakness,
    WeaknessKind,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _si(**kw) -> StrategicIntelligence:
    return StrategicIntelligence(**kw)


_VALID_KINDS = {k.value for k in WeaknessKind}


class _StubConclusion:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.confidence = 0.9
        self.reasoning = f"verdict is {verdict}."


class _StubReport:
    def __init__(self, verdict: str) -> None:
        self.conclusion = _StubConclusion(verdict)


class _StubScientist:
    """A minimal Scientist stand-in: echoes a fixed verdict for any claim."""

    def __init__(self, verdict: str = "refuted") -> None:
        self._verdict = verdict
        self.seen: list[str] = []

    def investigate(self, question: str) -> _StubReport:
        self.seen.append(question)
        return _StubReport(self._verdict)


# --------------------------------------------------------------------------- #
# the framework: every section is produced, fully offline
# --------------------------------------------------------------------------- #
def test_analyze_produces_all_six_sections():
    a = _si().analyze("Should we rewrite the kernel in Rust?")
    assert isinstance(a, StrategicAnalysis)
    assert a.direct_answer.strip()
    assert a.reality_check is not None
    assert a.key_weaknesses, "weaknesses must be non-empty even offline"
    assert a.root_cause.strip()
    assert a.optimized_solution.strip()
    assert a.execution_steps, "execution steps must be non-empty"


def test_to_dict_round_trips_and_is_serialisable():
    import json

    a = _si().analyze("Expand to three new markets next quarter.")
    d = a.to_dict()
    for key in ("problem", "direct_answer", "reality_check", "key_weaknesses",
                "root_cause", "optimized_solution", "execution_steps",
                "hidden_assumptions", "worst_case", "confidence"):
        assert key in d
    # fully JSON-serialisable (drives the console / server surfaces)
    json.loads(json.dumps(d))


def test_weaknesses_are_categorised():
    a = _si().analyze("Cut the test suite to ship faster.")
    for w in a.key_weaknesses:
        assert w.kind.value in _VALID_KINDS
        assert w.issue.strip()
        assert 0.0 <= w.severity <= 1.0


# --------------------------------------------------------------------------- #
# Reality Check reuses the Scientist to stress-test the premise
# --------------------------------------------------------------------------- #
def test_false_premise_is_flagged_via_scientist():
    sci = _StubScientist(verdict="refuted")
    a = _si(scientist=sci).analyze("Given that 2 + 2 = 5, double the target.")
    assert a.reality_check is not None
    assert a.reality_check.verdict is Soundness.FALSE
    # the embedded arithmetic claim was actually routed to the scientist
    assert any("2 + 2 = 5" in q for q in sci.seen)
    # a false premise produces a decisively negative direct answer
    assert ("no" in a.direct_answer.lower()) or ("false" in a.direct_answer.lower())


def test_supported_premise_reads_as_sound():
    sci = _StubScientist(verdict="supported")
    a = _si(scientist=sci).analyze("Given that 2 + 2 = 4, proceed.")
    assert a.reality_check.verdict is Soundness.SOUND


def test_reality_check_heuristic_without_scientist():
    # no scientist available — an unproven marker still flags a questionable premise
    a = _si().analyze("Since everyone knows growth is infinite, raise spend.")
    assert a.reality_check.verdict in (Soundness.QUESTIONABLE, Soundness.FALSE)


# --------------------------------------------------------------------------- #
# graceful degradation: complete analysis with no dependencies at all
# --------------------------------------------------------------------------- #
def test_fully_offline_no_dependencies():
    a = StrategicIntelligence(
        reasoner=None, council=None, scientist=None,
        world_model=None, self_model=None,
    ).analyze("Build a moon base.")
    assert a.direct_answer and a.root_cause and a.optimized_solution
    assert a.key_weaknesses and a.execution_steps
    assert a.worst_case.strip()


def test_empty_problem_does_not_crash():
    a = _si().analyze("")
    assert isinstance(a, StrategicAnalysis)
    assert 0.0 <= a.confidence <= 0.95


# --------------------------------------------------------------------------- #
# honesty: confidence is calibrated and never certain
# --------------------------------------------------------------------------- #
def test_confidence_is_bounded_and_never_certain():
    a = _si().analyze("Predict the exact stock price in ten years.")
    assert 0.0 <= a.confidence <= 0.95


def test_hallucination_risk_lowers_confidence():
    class _SelfModel:
        def hallucination_risk(self, text):
            return 0.9, ["unknowable-future"]

    high = _si(self_model=_SelfModel()).analyze("Name the 100th prime after a googol.")
    low = _si().analyze("Name the 100th prime after a googol.")
    assert high.confidence < low.confidence


# --------------------------------------------------------------------------- #
# strategize() alias and history
# --------------------------------------------------------------------------- #
def test_strategize_alias_and_history():
    si = _si()
    a = si.strategize("Adopt microservices everywhere.")
    assert isinstance(a, StrategicAnalysis)
    assert si.all_analyses() and si.all_analyses()[-1] is a


# --------------------------------------------------------------------------- #
# domain objects
# --------------------------------------------------------------------------- #
def test_dataclass_to_dict_shapes():
    pc = PremiseCheck(premise="x", verdict=Soundness.FALSE, correction="y", confidence=0.8)
    assert pc.to_dict() == {"premise": "x", "verdict": "false",
                            "correction": "y", "confidence": 0.8}
    w = Weakness(issue="z", kind=WeaknessKind.SCALABILITY, severity=0.5)
    assert w.to_dict() == {"issue": "z", "kind": "scalability", "severity": 0.5}


# --------------------------------------------------------------------------- #
# integration through NyxaraCore.strategize
# --------------------------------------------------------------------------- #
def test_core_strategize_integration():
    from nyxara import NyxaraCore

    core = NyxaraCore()
    out = core.strategize("Should we centralise all services into one monolith?")
    assert "error" not in out, out
    for key in ("direct_answer", "reality_check", "key_weaknesses",
                "root_cause", "optimized_solution", "execution_steps"):
        assert key in out
