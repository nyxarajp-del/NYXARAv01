"""Tests for nyxara.growth.causal_validation — the graph checked against laws she cannot bend."""

from __future__ import annotations

import pytest

from nyxara.growth.causal_validation import CASES, GroundTruthCase, validate_all, validate_case


def _case(name: str) -> GroundTruthCase:
    return next(c for c in CASES if c.name == name)


# -------------------- the worlds are usable ground truth -------------------- #
def test_sim_exports_the_law_worlds():
    """__init__.py was zero bytes, so nothing advertised these and the causal stack never met
    them. They are the only place a learned graph can be checked against something true by
    construction."""
    import nyxara.sim as sim

    for name in ("RCCircuit", "GasBox", "WaveString", "ElectrostaticWorld", "EpidemicWorld"):
        assert getattr(sim, name) is not None


def test_heavy_environments_load_lazily():
    import nyxara.sim as sim

    assert sim.PhysicsWorld.__name__ == "PhysicsWorld"
    with pytest.raises(AttributeError):
        sim.DefinitelyNotAThing


def test_sampling_is_independent_so_input_pairs_are_negative_controls():
    case = _case("rc_circuit")
    rows = case.sample(60, seed=0)
    assert len(rows) >= 50
    assert set(rows[0]) == set(case.columns)
    # the two inputs are drawn independently, so any edge between them is invention by definition
    assert ("resistance", "capacitance") not in case.true_edges
    assert ("capacitance", "resistance") not in case.true_edges


def test_true_edges_are_every_non_inert_input():
    case = _case("wave_string")
    assert case.true_edges == {("tension", "speed"), ("mu", "speed")}


# -------------------- with a declared ordering she is exact -------------------- #
@pytest.mark.parametrize("name", [c.name for c in CASES])
def test_every_world_is_recovered_exactly_with_a_declared_order(name):
    """The real-use path: the Master supplies the ordering (--order), which is the one thing a
    static table genuinely cannot know, and the numbers supply everything else."""
    report = validate_case(_case(name), n=120, seed=0, declare_order=True)
    assert report.rows >= 50
    assert report.recovered, report.summary()
    assert report.spurious == set()


def test_validate_all_covers_every_case():
    reports = validate_all(n=60)
    assert len(reports) == len(CASES)
    assert all(r.invented_nothing for r in reports), [r.summary() for r in reports]


# -------------------- unaided, the honest picture -------------------- #
def test_notears_abstains_rather_than_guessing_on_the_rc_circuit():
    """When the gradient fit comes back cyclic there is no acyclic ordering to be had, and
    abstaining is the correct answer — the alternative is a confident arbitrary arrow."""
    report = validate_case(_case("rc_circuit"), n=120, seed=0, declare_order=False)
    assert report.orientation == "unknown"
    assert report.spurious == set()


def test_notears_can_invert_a_nonlinear_law_and_the_report_says_so():
    """This is what ground-truth validation is FOR. NOTEARS minimises a LINEAR reconstruction, so
    on v = sqrt(T/mu) the reverse direction fits better and the arrows come back inverted with
    full confidence. Self-validation would never have caught it; a world with a known answer
    catches it immediately. Pinned as a test so the limitation cannot quietly regress into a
    claim of correctness."""
    report = validate_case(_case("wave_string"), n=120, seed=0, declare_order=False)
    assert report.orientation == "notears"
    assert report.spurious, "the known nonlinear failure did not reproduce"
    assert not report.recovered


def test_a_notears_orientation_carries_its_own_warning():
    from nyxara.growth.dataset_causal import learn_causal_graph

    case = _case("wave_string")
    rows = case.sample(120, seed=0)
    learned = learn_causal_graph(case.columns, rows)
    if learned.orientation == "notears":
        assert any("can invert on a nonlinear law" in n for n in learned.notes)


# -------------------- reporting -------------------- #
def test_missed_and_invented_are_counted_separately():
    """A graph that finds everything and invents nothing is a very different object from one that
    scores the same by doing both, so they never collapse into a single accuracy number."""
    report = validate_case(_case("rc_circuit"), n=120, seed=0)
    blob = report.to_dict()
    assert "missed" in blob and "spurious" in blob
    assert blob["recovered"] is True
    assert "invented_nothing" in blob


def test_a_world_that_yields_too_little_data_says_so():
    report = validate_case(_case("rc_circuit"), n=3, seed=0)
    assert not report.recovered
    assert any("too few usable samples" in n for n in report.notes)
