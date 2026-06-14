"""Tests for nyxara.growth.autonomous_scientist — the self-driven discovery loop.

Observe → Hypothesis → Experiment → Result → Update model. The whole loop runs offline
(no LLM, no network), driven by a deterministic proposition generator, so these tests are
hermetic and reproducible.
"""

from __future__ import annotations

from nyxara.growth.autonomous_scientist import (
    AutonomousScientist,
    Belief,
    BeliefModel,
    DiscoveryCycle,
    DiscoveryReport,
    QuestionOrigin,
)
from nyxara.growth.scientist import Scientist, Verdict
from nyxara.mind.world_model import WorldModel


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _auto(**kw) -> AutonomousScientist:
    return AutonomousScientist(**kw)


# --------------------------------------------------------------------------- #
# the loop runs and creates information, fully offline
# --------------------------------------------------------------------------- #
def test_discover_runs_requested_cycles():
    rep = _auto().discover(cycles=5)
    assert isinstance(rep, DiscoveryReport)
    assert len(rep.cycles) == 5
    assert all(isinstance(c, DiscoveryCycle) for c in rep.cycles)
    assert rep.elapsed_ms >= 0.0


def test_each_cycle_observes_a_question_and_concludes():
    rep = _auto().discover(cycles=4)
    for c in rep.cycles:
        assert c.question                      # she posed a question
        assert isinstance(c.origin, QuestionOrigin)
        assert c.report is not None            # the Scientist produced a report
        assert c.report.conclusion is not None


def test_seed_propositions_get_real_verdicts():
    """The self-generated propositions are genuinely testable, not decorative."""
    rep = _auto().discover(cycles=6)
    verdicts = {
        c.report.hypotheses[0].variable: c.report.conclusion.verdict
        for c in rep.cycles if c.report and c.report.hypotheses
    }
    # a known prime is SUPPORTED, parity-closure is SUPPORTED
    assert verdicts.get("is_prime(11)") is Verdict.SUPPORTED
    assert verdicts.get("parity_closure(even+even)") is Verdict.SUPPORTED
    assert verdicts.get("parity_closure(odd+odd)") is Verdict.SUPPORTED


def test_composite_seed_is_refuted():
    """The integer stream marches 11,12,13,… — composites must be REFUTED, honestly."""
    auto = _auto()
    auto.discover(cycles=16)   # enough cycles to reach a composite candidate (e.g. 12, 14, 15)
    primes_tested = {
        v: b.verdict for v, b in auto.belief_model().beliefs.items()
        if v.startswith("is_prime(")
    }
    assert any(verdict == Verdict.REFUTED.value for verdict in primes_tested.values())


# --------------------------------------------------------------------------- #
# the model actually updates (Observe → … → Update model)
# --------------------------------------------------------------------------- #
def test_belief_model_grows_with_new_information():
    auto = _auto()
    rep = auto.discover(cycles=6)
    assert len(auto.belief_model()) > 0
    assert rep.new_beliefs > 0
    assert rep.model_size == len(auto.belief_model())


def test_repeats_revise_not_duplicate():
    """Re-posing the same propositions revises beliefs in place, never piling up duplicates."""
    auto = _auto()
    auto.discover(cycles=4)
    size = len(auto.belief_model())
    # rewind the generator and the de-dup memory so the same questions are observed again
    auto._seed_n = 0
    auto._seen.clear()
    auto._frontier.clear()
    rep2 = auto.discover(cycles=4)
    assert len(auto.belief_model()) == size       # no new distinct beliefs
    assert rep2.revised_beliefs > 0               # but the evidence was folded in
    some = next(iter(auto.belief_model().beliefs.values()))
    assert some.evidence_count >= 2


def test_belief_model_update_records_verdict_and_confidence():
    model = BeliefModel()
    rep = Scientist().investigate("Is 17 a prime number?")
    delta = model.update(rep)
    assert delta["new"] is True
    b = model.get("is_prime(17)")
    assert isinstance(b, Belief)
    assert b.verdict == Verdict.SUPPORTED.value
    assert 0.0 <= b.confidence <= 1.0


# --------------------------------------------------------------------------- #
# the loop closes: a conclusion spawns the next question
# --------------------------------------------------------------------------- #
def test_follow_up_is_enqueued_and_consumed():
    auto = _auto()
    # first cycle produces a conclusion whose next_hypothesis is pushed onto the frontier
    first = auto.step()
    assert first.report.conclusion.next_hypothesis  # the Scientist always suggests a follow-up
    assert len(auto._frontier) >= 1
    # the very next observe consumes that follow-up
    second = auto.step()
    assert second.origin is QuestionOrigin.FOLLOW_UP


# --------------------------------------------------------------------------- #
# the world model is updated every cycle when wired
# --------------------------------------------------------------------------- #
def test_world_model_updates_each_cycle():
    wm = WorldModel()
    auto = _auto(world_model=wm)
    rep = auto.discover(cycles=5)
    assert len(wm) == 5
    assert rep.world_transitions_added == 5
    assert wm.actions() == ["investigate"]


def test_no_world_model_is_fine():
    rep = _auto().discover(cycles=3)
    assert rep.world_transitions_added == 0
    assert len(rep.cycles) == 3


# --------------------------------------------------------------------------- #
# Observe can mine self-knowledge gaps
# --------------------------------------------------------------------------- #
def test_gap_source_is_mined_for_questions():
    gaps = {"dark matter": "unknown", "consciousness": "unknown"}
    auto = _auto(gap_source=lambda: gaps)
    auto._seed_n = 10_000  # push seeds out of the way so a gap is observed first
    cycle = auto.step()
    assert cycle.origin is QuestionOrigin.GAP
    assert "dark matter" in cycle.question or "consciousness" in cycle.question


def test_gap_source_failure_is_survivable():
    def _boom():
        raise RuntimeError("no self-model")

    auto = _auto(gap_source=_boom)
    cycle = auto.step()             # must fall back to a seed, never crash
    assert cycle.origin is QuestionOrigin.SEED
    assert cycle.report is not None


# --------------------------------------------------------------------------- #
# composition + graceful degradation
# --------------------------------------------------------------------------- #
def test_uses_injected_scientist():
    calls = []

    class _StubScientist:
        def investigate(self, q):
            calls.append(q)
            return Scientist().investigate(q)

    auto = _auto(scientist=_StubScientist())
    auto.discover(cycles=2)
    assert len(calls) == 2


def test_degrades_when_everything_is_none():
    rep = _auto().discover(cycles=3)
    assert isinstance(rep, DiscoveryReport)
    assert len(rep.cycles) == 3
    assert all(c.report is not None for c in rep.cycles)


def test_report_to_dict_roundtrip():
    d = _auto().discover(cycles=2).to_dict()
    assert d["cycles_run"] == 2
    assert "new_beliefs" in d and "model_size" in d
    assert len(d["cycles"]) == 2
    assert d["cycles"][0]["origin"] in {o.value for o in QuestionOrigin}


# --------------------------------------------------------------------------- #
# core integration
# --------------------------------------------------------------------------- #
def test_core_discover_is_wired():
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore(enable_tools=False, enable_skills=False)
    out = core.discover(3)
    assert "error" not in out
    assert out["cycles_run"] == 3
    # the status report surfaces the discovery counts
    rep = core.report()
    assert rep.get("discoveries", 0) >= 3
    assert rep.get("beliefs_held", 0) >= 1
