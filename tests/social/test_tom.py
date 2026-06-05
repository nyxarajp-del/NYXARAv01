"""Tests for nyxara.social.tom."""

from __future__ import annotations

from nyxara.social.tom import MentalState, TheoryOfMind


def _sally_anne():
    tom = TheoryOfMind()
    tom.add_agent("Sally")
    tom.add_agent("Anne")
    tom.set_world("marble", "basket")
    tom.observe("Sally", "marble")
    tom.observe("Anne", "marble")
    # Sally leaves; Anne moves marble; only Anne witnesses
    tom.set_world("marble", "box")
    tom.observe("Anne", "marble")
    return tom


# -------------------- MentalState -------------------- #
def test_mental_state_model_of_creates_nested():
    ms = MentalState()
    nested = ms.model_of("Bob")
    assert isinstance(nested, MentalState)
    assert ms.model_of("Bob") is nested  # same instance


def test_mental_state_to_dict():
    ms = MentalState(beliefs={"x": 1}, desires={"g": 0.5})
    ms.model_of("B").beliefs["y"] = 2
    d = ms.to_dict(depth=1)
    assert d["beliefs"] == {"x": 1}
    assert "models_of" in d


# -------------------- world & agents -------------------- #
def test_set_get_world():
    tom = TheoryOfMind()
    tom.set_world("x", 5)
    assert tom.get_world("x") == 5


def test_add_agent():
    tom = TheoryOfMind()
    tom.add_agent("A")
    assert "A" in tom.agents()


# -------------------- observe & belief -------------------- #
def test_observe_forms_true_belief():
    tom = TheoryOfMind()
    tom.set_world("loc", "a")
    tom.observe("A", "loc")
    assert tom.belief("A", "loc") == "a"


def test_belief_unknown_is_none():
    tom = TheoryOfMind()
    assert tom.belief("Nobody", "x") is None


def test_set_belief():
    tom = TheoryOfMind()
    tom.set_belief("A", "x", 42)
    assert tom.belief("A", "x") == 42


# -------------------- false belief (Sally-Anne) -------------------- #
def test_sally_retains_false_belief():
    tom = _sally_anne()
    assert tom.get_world("marble") == "box"
    assert tom.belief("Sally", "marble") == "basket"
    assert tom.has_false_belief("Sally", "marble")


def test_anne_has_true_belief():
    tom = _sally_anne()
    assert tom.belief("Anne", "marble") == "box"
    assert not tom.has_false_belief("Anne", "marble")


def test_predict_acts_on_belief_not_truth():
    tom = _sally_anne()
    # Sally searches where she BELIEVES, not where it truly is
    assert tom.predict_belief_based_action("Sally", "marble") == "basket"
    assert tom.predict_belief_based_action("Anne", "marble") == "box"


def test_world_change_doesnt_auto_update_unobserved():
    tom = TheoryOfMind()
    tom.set_world("x", "a")
    tom.observe("A", "x")
    tom.set_world("x", "b")  # changes truth; A doesn't observe
    assert tom.belief("A", "x") == "a"  # stale


def test_predict_unknown_belief_none():
    tom = TheoryOfMind()
    tom.add_agent("A")
    assert tom.predict_belief_based_action("A", "unknown") is None


# -------------------- nested / recursive ToM -------------------- #
def test_nested_belief_set_get():
    tom = TheoryOfMind()
    tom.set_belief(["A", "B"], "x", "value")
    assert tom.belief(["A", "B"], "x") == "value"


def test_attribute_belief():
    tom = TheoryOfMind()
    tom.set_world("marble", "box")
    tom.attribute_belief("Anne", "Sally", "marble", "basket")
    assert tom.belief(["Anne", "Sally"], "marble") == "basket"


def test_second_order_false_belief_attribution():
    tom = _sally_anne()
    # Anne knows Sally didn't see the move
    tom.attribute_belief("Anne", "Sally", "marble", "basket")
    assert tom.attributes_false_belief("Anne", "Sally", "marble")


def test_attributes_false_belief_negative():
    tom = TheoryOfMind()
    tom.set_world("marble", "box")
    tom.attribute_belief("Anne", "Sally", "marble", "box")  # Anne thinks Sally knows
    assert not tom.attributes_false_belief("Anne", "Sally", "marble")


def test_three_level_nesting():
    tom = TheoryOfMind()
    tom.set_belief(["A", "B", "C"], "x", 1)
    assert tom.belief(["A", "B", "C"], "x") == 1


def test_resolve_no_create_returns_none():
    tom = TheoryOfMind()
    assert tom.belief(["A", "B"], "x") is None  # nothing created yet


# -------------------- desires / intentions -------------------- #
def test_desires():
    tom = TheoryOfMind()
    tom.set_desire("A", "goal1", 0.5)
    tom.set_desire("A", "goal2", 0.9)
    assert tom.strongest_desire("A") == "goal2"


def test_strongest_desire_none():
    tom = TheoryOfMind()
    tom.add_agent("A")
    assert tom.strongest_desire("A") is None


def test_intentions():
    tom = TheoryOfMind()
    tom.set_intention("A", "do x")
    tom.set_intention("A", "do y")
    assert tom.predict_intention("A") == "do y"


def test_predict_intention_falls_back_to_desire():
    tom = TheoryOfMind()
    tom.set_desire("A", "want_thing", 1.0)
    assert tom.predict_intention("A") == "want_thing"


def test_predict_intention_unknown():
    tom = TheoryOfMind()
    assert tom.predict_intention("Nobody") is None


# -------------------- state -------------------- #
def test_state():
    tom = _sally_anne()
    s = tom.state("Sally")
    assert s["beliefs"]["marble"] == "basket"


def test_state_unknown_none():
    assert TheoryOfMind().state("Nobody") is None
