"""What she learned about breaking programs, and that she learned it rather than was told it."""

from __future__ import annotations

import pytest

from nyxara.njp.programming import OPERATIONS, POOLS, RAN, TRAITS, Programmer, Situation, probe


@pytest.fixture(scope="module")
def taught() -> Programmer:
    programmer = Programmer(seed=7)
    programmer.experiment(1200)
    programmer.learn()
    programmer.learn_failure()
    programmer.challenge(tries=250)
    return programmer


# --------------------------------------------------------------------------------------------- #
#  nothing here is told
# --------------------------------------------------------------------------------------------- #
def test_an_outcome_is_what_python_did_and_nothing_else():
    """Every outcome in the module comes from running the operation, so this one must too."""
    programmer = Programmer(seed=1)
    index = next(o for o in OPERATIONS if o.name == "index")
    assert programmer.run(Situation(index, ([1, 2, 3], 5))).outcome == "IndexError"
    assert programmer.run(Situation(index, ([1, 2, 3], 1))).outcome == RAN


def test_no_probe_names_an_error():
    """The readings are senses. If one of them named an outcome the laws would be a lookup."""
    index = next(o for o in OPERATIONS if o.name == "index")
    reading = probe(index, ([1, 2, 3], 5))
    for name, value in reading.items():
        assert "Error" not in name and "error" not in name
        assert not (isinstance(value, str) and value.endswith("Error"))


def test_an_untaught_programmer_has_no_laws():
    assert Programmer(seed=3).learn() == []


def test_switching_induction_off_leaves_the_running_intact():
    """The control separates what watching was worth from what working it out was worth."""
    blind = Programmer(seed=5, learn=False)
    blind.experiment(300)
    assert blind.trials and blind.learn() == []


# --------------------------------------------------------------------------------------------- #
#  what she worked out
# --------------------------------------------------------------------------------------------- #
def test_she_finds_the_index_law(taught):
    laws = [law for law in taught.laws if law.outcome == "IndexError"]
    assert laws, "nothing learned about IndexError"
    assert any("<len(" in name for law in laws for name, _ in law.terms), \
        "the index law does not mention the index against the length"


def test_she_finds_the_division_law(taught):
    laws = [law for law in taught.laws if law.outcome == "ZeroDivisionError"]
    assert laws
    assert any(name.endswith(".zero") and value is True
               for law in laws for name, value in law.terms)


def test_she_finds_the_conversion_law(taught):
    laws = [law for law in taught.laws if law.outcome == "ValueError"]
    assert any(name.endswith(".digits") for law in laws for name, _ in law.terms)


def test_one_outcome_may_have_several_causes(taught):
    """A cause with two shapes is two laws. Demanding one produced a law about lengths."""
    counts = {}
    for law in taught.laws:
        counts[law.outcome] = counts.get(law.outcome, 0) + 1
    assert max(counts.values()) > 1


def test_no_law_is_kept_with_a_counterexample(taught):
    for law in taught.laws:
        assert law.counterexamples == 0 and law.exact


def test_most_laws_are_about_a_kind_of_operation_not_a_named_one(taught):
    """A law about the `index` operation is not a law about indices."""
    assert sum(1 for law in taught.laws if law.about_a_kind) > len(taught.laws) / 2


def test_a_law_that_cannot_survive_being_shot_at_is_not_kept():
    """Exact on everything she ran, and wrong about the world. Only a hunt finds that out."""
    programmer = Programmer(seed=11)
    programmer.experiment(500)
    before = programmer.learn()
    survivors = programmer.challenge(tries=300)
    assert len(survivors) <= len(before)
    for law in survivors:
        assert law.counterexamples == 0


# --------------------------------------------------------------------------------------------- #
#  using it
# --------------------------------------------------------------------------------------------- #
def test_she_predicts_before_running(taught):
    index = next(o for o in OPERATIONS if o.name == "index")
    guess, why = taught.predict(Situation(index, ([1, 2, 3], 9)))
    assert guess == "IndexError" and why


def test_she_says_it_runs_when_no_law_objects(taught):
    index = next(o for o in OPERATIONS if o.name == "index")
    guess, _why = taught.predict(Situation(index, ([1, 2, 3], 1)))
    assert guess == RAN


def test_a_repair_is_verified_by_running_it_again(taught):
    index = next(o for o in OPERATIONS if o.name == "index")
    broken = Situation(index, ([1, 2, 3], 9))
    repair = taught.repair(broken)
    assert repair is not None and repair.outcome == "IndexError"
    fixed = list(broken.args)
    fixed[repair.argument] = eval(repair.became)  # noqa: S307 - a repr of a literal from the pool
    assert not taught.run(Situation(index, tuple(fixed))).failed


def test_nothing_that_already_works_is_repaired(taught):
    index = next(o for o in OPERATIONS if o.name == "index")
    assert taught.repair(Situation(index, ([1, 2, 3], 0))) is None


def test_the_coarse_question_is_answered_separately(taught):
    """Whether it breaks and what it is called are different questions and generalise apart."""
    assert taught.failure_laws
    index = next(o for o in OPERATIONS if o.name == "index")
    assert taught.will_fail(Situation(index, ([1, 2, 3], 9)))[0] is True
    assert taught.will_fail(Situation(index, ([1, 2, 3], 0)))[0] is False


# --------------------------------------------------------------------------------------------- #
#  the world she acts in
# --------------------------------------------------------------------------------------------- #
def test_most_of_what_she_does_is_meant_to_work(taught):
    """A learner in a world where every act fails learns nothing about failing."""
    seen = taught.seen()
    assert seen.get(RAN, 0) > max(v for k, v in seen.items() if k != RAN)


def test_she_sees_more_than_one_kind_of_failure(taught):
    assert len({k for k in taught.seen() if k != RAN}) >= 5


def test_every_operation_declares_what_it_is_written_for():
    for operation in OPERATIONS:
        assert len(operation.kinds) == operation.arity, operation.name
        assert operation.traits, operation.name
        for kind in operation.kinds:
            assert kind in POOLS, kind
        for trait in operation.traits:
            assert trait in TRAITS


def test_the_attribute_pool_contains_names_that_exist_and_names_that_do_not():
    """Shown only failures she concluded getattr always fails — and she was exact and wrong."""
    from nyxara.njp.programming import _Thing

    names = set(dir(_Thing()))
    pool = set(POOLS["text"])
    assert pool & names and pool - names
