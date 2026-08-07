"""NYX V.01 · superposition — many answers held at once, and honest about being undecided."""

from __future__ import annotations

from nyxara.nyx.modules import Proposal
from nyxara.nyx.superpose import SolutionSuperposition


def _p(source: str, confidence: float, **kw) -> Proposal:
    return Proposal(source=source, content=f"{source} says so", confidence=confidence, **kw)


def _sp(proposals, **kw) -> SolutionSuperposition:
    return SolutionSuperposition.from_proposals(proposals, **kw)


# -------------------- the state -------------------- #
def test_every_candidate_survives_into_the_state():
    """The point of superposition: the runners-up do not stop existing."""
    sp = _sp([_p("a", 0.9), _p("b", 0.5), _p("c", 0.1)])
    assert len(sp) == 3
    assert set(sp.probabilities()) == {"a", "b", "c"}


def test_probabilities_normalise():
    sp = _sp([_p("a", 0.9), _p("b", 0.3)])
    assert abs(sum(sp.probabilities().values()) - 1.0) < 1e-6


def test_a_stronger_bid_carries_more_probability():
    probs = _sp([_p("a", 0.9), _p("b", 0.2)]).probabilities()
    assert probs["a"] > probs["b"]


def test_candidates_are_capped():
    sp = _sp([_p(f"s{i}", 0.5) for i in range(50)], max_candidates=5)
    assert len(sp) == 5


def test_an_empty_state_collapses_to_nothing_not_a_crash():
    got = _sp([]).collapse()
    assert got.winner is None and got.decided is False and got.considered == 0


# -------------------- evidence -------------------- #
def test_evidence_shifts_the_posterior():
    sp = _sp([_p("a", 0.9), _p("b", 0.2)])
    before = sp.probabilities()["b"]
    sp.observe({"b": 9.0})
    assert sp.probabilities()["b"] > before


def test_evidence_for_an_unknown_label_is_ignored():
    sp = _sp([_p("a", 0.9)])
    before = sp.probabilities()
    sp.observe({"nobody": 5.0})
    assert sp.probabilities() == before


def test_strong_evidence_can_overturn_the_leader():
    sp = _sp([_p("a", 0.9), _p("b", 0.2)])
    assert sp.collapse().label == "a"
    sp.observe({"b": 50.0})
    assert sp.collapse().label == "b"


# -------------------- measurement -------------------- #
def test_a_dominant_candidate_is_decided():
    got = _sp([_p("a", 0.99), _p("b", 0.05)], collapse_threshold=0.5).collapse()
    assert got.decided is True and got.label == "a"


def test_a_close_race_stays_undecided_rather_than_bluffing():
    """Staying superposed and saying so beats committing to a leader that never earned it."""
    got = _sp([_p("a", 0.5), _p("b", 0.5), _p("c", 0.5)], collapse_threshold=0.6).collapse()
    assert got.decided is False
    assert got.label                              # the leader is still reported, just not trusted


def test_force_decides_even_a_close_race():
    got = _sp([_p("a", 0.5), _p("b", 0.5)], collapse_threshold=0.9).collapse(force=True)
    assert got.decided is True


def test_entropy_is_higher_when_belief_is_spread():
    spread = _sp([_p("a", 0.5), _p("b", 0.5), _p("c", 0.5)]).collapse().entropy
    peaked = _sp([_p("a", 0.99), _p("b", 0.01)]).collapse().entropy
    assert spread > peaked


def test_the_collapse_carries_the_original_proposal():
    got = _sp([_p("a", 0.9, verifiable=True), _p("b", 0.1)]).collapse()
    assert got.winner is not None and got.winner.source == "a"
    assert got.winner.verifiable is True


def test_runners_up_are_reported_with_real_probabilities():
    got = _sp([_p("a", 0.9), _p("b", 0.5), _p("c", 0.2)]).collapse()
    assert got.runners_up and all(0.0 <= p <= 1.0 for _, p in got.runners_up)
    assert got.label not in [lab for lab, _ in got.runners_up]


def test_margin_reflects_how_clear_the_win_was():
    clear = _sp([_p("a", 0.99), _p("b", 0.02)]).collapse().margin
    close = _sp([_p("a", 0.51), _p("b", 0.5)]).collapse().margin
    assert clear > close


# -------------------- robustness -------------------- #
def test_none_proposals_are_skipped():
    sp = _sp([_p("a", 0.9), None, _p("b", 0.2)])
    assert len(sp) == 2


def test_zero_confidence_candidates_still_exist_but_lose():
    got = _sp([_p("a", 0.0), _p("b", 0.9)]).collapse()
    assert got.label == "b" and len(got.runners_up) >= 1


def test_stats_reports_the_distribution_and_what_was_verifiable():
    sp = _sp([_p("a", 0.9, verifiable=True), _p("b", 0.2)])
    stats = sp.stats()
    assert stats["candidates"] == 2 and stats["verifiable"] == ["a"]
    assert set(stats["probabilities"]) == {"a", "b"}
