"""What she is actually good at, measured — and what follows from knowing it.

A self-model that lists capabilities is a brochure. This one carries numbers she did not choose.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.selfmodel import MetaLearner, SelfModel


@pytest.fixture()
def model() -> SelfModel:
    m = SelfModel()
    for _ in range(12):
        m.observe("grounding", 1.0)
        m.observe("planning", 0.1)
    return m


# ---- the capability map ------------------------------------------------------ #
def test_a_faculty_that_works_reads_as_reliable(model: SelfModel):
    assert model.capabilities["grounding"].reliable
    assert "grounding" in model.report()["reliable"]


def test_a_faculty_that_fails_reads_as_weak(model: SelfModel):
    assert model.capabilities["planning"].weak
    assert "planning" in model.report()["weak"]


def test_an_untested_faculty_is_neither(model: SelfModel):
    """An untested faculty is not a weak one. Conflating them makes a cold start permanently timid."""
    tool_use = model.capabilities["tool_use"]
    assert not tool_use.weak and not tool_use.reliable
    assert "tool_use" in model.report()["untested"]


def test_certainty_grows_with_evidence():
    """Two successes out of two is a mean of 1.0 and means almost nothing."""
    thin, thick = SelfModel(), SelfModel()
    for _ in range(2):
        thin.observe("recall", 1.0)
    for _ in range(50):
        thick.observe("recall", 1.0)
    assert thick.capabilities["recall"].certainty > thin.capabilities["recall"].certainty


def test_partial_credit_is_allowed(model: SelfModel):
    """Most outcomes are not binary, and a half-success must land between the two."""
    good, middling = SelfModel(), SelfModel()
    for _ in range(10):
        good.observe("reasoning", 1.0)
        middling.observe("reasoning", 0.6)
    assert middling.level("reasoning") < good.level("reasoning")
    assert 0.5 < middling.level("reasoning") < 0.75


def test_repeated_evidence_at_the_prior_moves_the_level_nowhere(model: SelfModel):
    """0.5 against a uniform prior is genuinely no information about the direction — but it is
    information about how much she has seen, and only `certainty` should move."""
    for _ in range(10):
        model.observe("reasoning", 0.5)
    assert model.level("reasoning") == pytest.approx(0.5)
    assert model.capabilities["reasoning"].certainty > 0.0


# ---- confidence that knows its own worth -------------------------------------- #
def test_a_good_faculty_keeps_most_of_its_confidence(model: SelfModel):
    assert model.trust("grounding", 0.9) > 0.7


def test_a_poor_faculty_cannot_hand_up_a_high_confidence(model: SelfModel):
    """Wrong eight times in ten should not be able to claim 0.9 and be believed."""
    assert model.trust("planning", 0.9) < 0.3


def test_an_untested_faculty_is_not_discounted(model: SelfModel):
    assert model.trust("tool_use", 0.9) == 0.9


def test_trust_only_ever_moves_confidence_down(model: SelfModel):
    for capability in model.capabilities:
        assert model.trust(capability, 0.5) <= 0.5


def test_she_can_say_which_faculties_this_task_will_strain(model: SelfModel):
    """The reading behind "is task me meri reliability low hai" — counts, not a mood."""
    assert model.unreliable_for("command") == ["planning"]
    assert model.unreliable_for("question") == []


def test_the_weakest_faculty_is_named(model: SelfModel):
    assert model.weakest().name == "planning"


def test_nothing_measured_means_no_weakest():
    assert SelfModel().weakest() is None


# ---- the join with the error taxonomy ------------------------------------------ #
def test_a_diagnosis_counts_against_the_faculty_it_blamed(model: SelfModel):
    from nyxara.njp.predict import Diagnosis, ErrorKind

    before = model.level("recall")
    for _ in range(10):
        model.observe_diagnosis(Diagnosis(kind=ErrorKind.MEMORY))
    assert model.level("recall") < before


def test_an_unattributed_error_blames_nobody(model: SelfModel):
    """Spreading it across all faculties would slowly convince her she is bad at everything."""
    from nyxara.njp.predict import Diagnosis, ErrorKind

    before = {n: c.observations for n, c in model.capabilities.items()}
    for _ in range(5):
        model.observe_diagnosis(Diagnosis(kind=ErrorKind.UNATTRIBUTED))
    assert {n: c.observations for n, c in model.capabilities.items()} == before


# ---- meta-learning -------------------------------------------------------------- #
def test_the_better_strategy_wins_on_trials():
    """Adopted because trials say so, not because it sounds principled."""
    meta = MetaLearner(SelfModel())
    meta.register("depth", "shallow", 2)
    meta.register("depth", "deep", 8)
    for _ in range(30):
        chosen = meta.choose("depth")
        meta.reward("depth", 0.9 if chosen.name == "deep" else 0.2)
    assert meta.best("depth").name == "deep"


def test_every_strategy_is_tried_before_any_is_preferred():
    meta = MetaLearner()
    meta.register("k", "a", 1)
    meta.register("k", "b", 2)
    for _ in range(meta.min_trials * 2):
        meta.reward("k", 1.0) if meta.choose("k") else None
    assert all(s.trials > 0 for s in meta.strategies["k"].values())


def test_an_unregistered_arm_chooses_nothing():
    assert MetaLearner().choose("nonexistent") is None


def test_exploration_is_deterministic():
    """A run that cannot be replayed cannot be debugged."""
    def _run() -> list:
        meta = MetaLearner()
        meta.register("k", "a", 1)
        meta.register("k", "b", 2)
        picks = []
        for _ in range(40):
            picks.append(meta.choose("k").name)
            meta.reward("k", 0.5)
        return picks

    assert _run() == _run()


# ---- resource allocation --------------------------------------------------------- #
def test_allocation_starts_even_and_says_so():
    """A "learned" allocation derived from four observations would be overclaiming."""
    got = MetaLearner(SelfModel()).allocate()
    assert not got.learned
    assert len(set(round(v, 6) for v in got.weights.values())) == 1


def test_allocation_shifts_toward_what_pays():
    model = SelfModel()
    for _ in range(20):
        model.observe("recall", 0.95)
        model.observe("prediction", 0.05)
    got = MetaLearner(model).allocate()
    assert got.learned
    assert got.weights["memory"] > got.weights["simulation"]


# ---- persistence ------------------------------------------------------------------ #
def test_the_record_survives_a_reload(model: SelfModel):
    fresh = SelfModel()
    fresh.load_dict(model.to_dict())
    assert fresh.weakest().name == "planning"
    assert fresh.capabilities["grounding"].reliable


# ---- wired -------------------------------------------------------------------------- #
def test_the_brain_records_its_own_performance():
    brain = NJPBrain()
    thought = brain.think("my name is Jay")
    brain.resolve(thought, correct=1.0, actual=thought.answer)
    assert brain.self_model.capabilities["grounding"].observations > 0


def test_the_brain_registers_tunable_strategies():
    brain = NJPBrain()
    assert "settle_steps" in brain.meta.strategies
    assert len(brain.meta.strategies["settle_steps"]) == 3


def test_organs_that_are_off_are_absent():
    class _Off:
        self_model_enabled = False
        meta_enabled = False

    brain = NJPBrain(_Off())
    assert brain.self_model is None and brain.meta is None
    assert "self_model" not in brain.stats() and "meta" not in brain.stats()


# --------------------------------------------------------------------------- #
# Which reasoning she is weak in, not whether reasoning is weak
# --------------------------------------------------------------------------- #
def test_reasoning_is_tracked_by_the_route_the_answer_came_by():
    """One flat `reasoning` posterior let a reliable route subsidise a shaky one.

    A stored fact read back is nearly always right; a two-hop composition is a conjecture priced
    at 0.70 before anything else discounts it; a schema transfer is defeasible by construction.
    Averaged together, the number cannot answer the question a caller actually has — *should I
    believe this answer* — and the composed answer is exactly where the warning is needed.
    """
    from nyxara.njp.core import Derivation
    from nyxara.njp.selfmodel import mode_of

    assert mode_of(Derivation(kind="direct")) == "reasoning:direct"
    assert mode_of(Derivation(kind="schema")) == "reasoning:schema"
    short = Derivation(kind="composed", support=[("a", "causes", "b"), ("b", "causes", "c")])
    deep = Derivation(kind="composed", support=[("a", "causes", "b")] * 4)
    assert mode_of(short) == "reasoning:composed:short"
    assert mode_of(deep) == "reasoning:composed:deep"
    # No derivation at all is still reasoning, not a new unnamed faculty.
    assert mode_of(None) == "reasoning"


def test_an_untested_mode_borrows_its_parents_reliability():
    """Otherwise splitting a faculty makes the model *less* informative, not more.

    Every new mode starts at zero observations, falls under `_MIN_OBSERVATIONS`, and is therefore
    not discounted at all — so a brain that had just learned to compose would get a free pass on
    exactly the answers that most deserve doubt.
    """
    model = SelfModel()
    for _ in range(10):
        model.observe("reasoning", 0.6)
    borrowed = model.trust("reasoning:composed:short", 0.72)
    assert borrowed < 0.72, "an untested mode was not discounted at all"
    assert borrowed == pytest.approx(model.trust("reasoning", 0.72))


def test_once_a_mode_has_its_own_evidence_it_stops_borrowing():
    model = SelfModel()
    for _ in range(10):
        model.observe("reasoning", 0.9)
    for _ in range(10):
        model.observe("reasoning:composed:short", 0.2)

    composed = model.trust("reasoning:composed:short", 0.72)
    direct = model.trust("reasoning:direct", 0.72)
    assert composed < direct, "the weak mode was not judged on its own record"
    assert direct == pytest.approx(model.trust("reasoning", 0.72)), "direct still borrows"


def test_trust_is_still_only_ever_downward():
    """A mode that has done well must not be able to inflate a stated confidence."""
    model = SelfModel()
    for _ in range(20):
        model.observe("reasoning:composed:short", 1.0)
    assert model.trust("reasoning:composed:short", 0.5) <= 0.5
