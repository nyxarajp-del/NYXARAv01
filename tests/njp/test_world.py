"""Events, causes, consequences — and the difference between a cause and a coincidence.

Grounding gives her facts that hold. This gives her facts that change: what happened, why, what
follows, and what would have happened otherwise.
"""

from __future__ import annotations


from nyxara.njp.brain import NJPBrain
from nyxara.njp.world import Event, WorldView


def _rain_world() -> WorldView:
    """Rain reliably wets the road and often causes a skid. Washing a car also wets the road.

    So "wet" has two routes and "skids" has one — which is what makes the counterfactuals here
    have different right answers instead of all reducing to the same test.
    """
    world = WorldView()
    for i in range(12):
        world.observe(Event(actor="sky", action="rains"))
        world.observe(Event(actor="road", action="wet"))
        if i % 2 == 0:
            world.observe(Event(actor="car", action="skids"))
    for _ in range(4):
        world.observe(Event(actor="man", action="washes", object="car"))
        world.observe(Event(actor="road", action="wet"))
    return world


# ---- the arithmetic has to be arithmetic ----------------------------------- #
def test_a_conditional_probability_never_exceeds_one():
    """The regression that revealed the counting bug: P(wet|rains) came back as 200%.

    A single "rains" was credited once for the "wet" that followed it and again for the next
    cycle's "wet" still inside its window, so `together` outgrew `cause_total`.
    """
    world = _rain_world()
    for link in world.links(causal_only=False):
        assert 0.0 <= link.conditional <= 1.0, f"{link.cause}->{link.effect} = {link.conditional}"


def test_a_counterfactual_never_reports_negative_occurrences():
    world = _rain_world()
    got = world.counterfactual("rains", "wet")
    assert got.support >= 0
    assert "-" not in got.reason.split("of those")[0].split(";")[-1]


def test_expectation_confidence_is_a_probability():
    world = _rain_world()
    assert 0.0 <= world.expect("rains").confidence <= 1.0


# ---- causes, not coincidences ---------------------------------------------- #
def test_a_repeated_cause_with_lift_is_causal():
    world = _rain_world()
    got = world.why("wet")
    assert got.explained
    assert got.causes[0].cause == "rains"
    assert got.causes[0].lift > 0.0


def test_a_cause_must_beat_the_base_rate():
    """Something that precedes an effect no more often than chance explains nothing."""
    world = WorldView()
    for _ in range(20):
        world.observe(Event(action="noise"))
        world.observe(Event(action="common"))
        world.observe(Event(action="common"))
    link = world.link("noise", "common")
    assert link.conditional > 0.0
    assert link.lift < 0.5, "an effect that happens constantly must not look caused"


def test_one_observation_is_never_a_cause():
    """The rooster does not cause the sunrise, however reliably it precedes it once."""
    world = WorldView()
    world.observe(Event(action="rooster"))
    world.observe(Event(action="sunrise"))
    assert not world.link("rooster", "sunrise").causal
    assert not world.why("sunrise").explained


def test_a_correlate_is_reported_as_a_correlate_not_a_cause():
    world = WorldView()
    world.observe(Event(action="rooster"))
    world.observe(Event(action="sunrise"))
    got = world.why("sunrise")
    assert not got.causes and got.correlates
    assert "not often enough" in got.text


def test_an_effect_with_nothing_before_it_says_so():
    world = WorldView()
    world.observe(Event(action="alone"))
    assert "nothing that precedes" in world.why("alone").text


# ---- forward --------------------------------------------------------------- #
def test_she_expects_what_usually_follows():
    world = _rain_world()
    got = world.expect("rains")
    assert got.events and got.events[0][0] == "wet"


def test_an_unseen_antecedent_expects_nothing():
    got = WorldView().expect("never_happened")
    assert not got.events and not got.trusted
    assert "not seen" in got.reason


def test_a_single_sighting_is_not_trusted_foresight():
    """One observation gives a confident-looking 1.0. Trust needs repetition."""
    world = WorldView()
    world.observe(Event(action="a"))
    world.observe(Event(action="b"))
    got = world.expect("a")
    assert got.events and not got.trusted


# ---- backward: counterfactuals --------------------------------------------- #
def test_removing_the_only_cause_stops_the_effect():
    world = _rain_world()
    got = world.counterfactual("rains", "skids")
    assert got.answerable and got.still_happens is False
    assert got.confidence > 0.5


def test_removing_one_of_several_causes_leaves_the_effect():
    """The road still gets wet without rain, because washing the car also wets it."""
    world = _rain_world()
    got = world.counterfactual("washes:car", "wet")
    assert got.answerable and got.still_happens is True


def test_an_unrelated_removal_changes_nothing():
    world = _rain_world()
    got = world.counterfactual("never_happened", "wet")
    assert got.still_happens is True
    assert "never preceded" in got.reason


def test_too_little_evidence_refuses_to_answer():
    """A confident answer about a world that never happened is the easiest kind of lie."""
    world = WorldView()
    world.observe(Event(action="x"))
    world.observe(Event(action="y"))
    got = world.counterfactual("x", "y")
    assert not got.answerable and got.still_happens is None
    assert "too few" in got.reason


# ---- events vs facts -------------------------------------------------------- #
def test_only_things_that_happened_go_on_the_timeline():
    """"the Master's name is Jay" is a fact, not an event. It must not get a timestamp."""
    from nyxara.njp.grounding import Grounder

    world = WorldView()
    grounder = Grounder()
    world.from_grounding(grounder.ground("my name is Jay"))
    assert not world.events


def test_ordering_is_real_not_incidental():
    world = WorldView()
    for name in ("first", "second", "third"):
        world.observe(Event(action=name))
    assert [e.action for e in world.history()] == ["first", "second", "third"]
    assert [e.seq for e in world.history()] == [1, 2, 3]


# ---- persistence ------------------------------------------------------------ #
def test_causal_structure_survives_a_reload():
    world = _rain_world()
    fresh = WorldView()
    fresh.load_dict(world.to_dict())
    assert fresh.why("wet").causes[0].cause == "rains"
    assert fresh.counterfactual("rains", "skids").still_happens is False


# ---- wired ------------------------------------------------------------------ #
def test_the_brain_builds_a_world():
    brain = NJPBrain()
    assert brain.world is not None
    assert "world" in brain.stats()


def test_an_organ_that_is_off_is_absent():
    class _Off:
        world_enabled = False

    brain = NJPBrain(_Off())
    assert brain.world is None
    assert "world" not in brain.stats()
