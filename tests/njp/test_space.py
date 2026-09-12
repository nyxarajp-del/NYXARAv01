"""The hypothesis space as a map, and the detector that says which corner is empty.

The thing under test is not "does it find gaps". Any function that returns every unvaried knob
finds gaps. What is tested here is that it **stays quiet** on a map that was fully examined, that a
region entered by an experiment which broke its own promise is still counted empty, and that an
intervention is judged against the invariants it declared rather than against a hardcoded one.
"""

from __future__ import annotations

import pytest

from nyxara.njp.space import (
    MOVED, REGIONS, SLACK, VERBS, Held, Intervention, Knob, Map, carry_out, gaps, propose, survey,
)
from nyxara.njp.spaceschool import KNOBS, KNOWN, examine, retrodict


# --------------------------------------------------------------------------------------------- #
#  an intervention is checked against its own promises
# --------------------------------------------------------------------------------------------- #
def _world(score: float, keep: float = 1.0):
    return {"score": score, "keep": keep}


HOLD = Held(name="keep", read=lambda w: w["keep"])


def _one(after, *, holds=(HOLD,), deployable=True):
    return Intervention(verb="remove", on="a knob", holds=holds, deployable=deployable,
                        run=lambda: after)


def test_an_intervention_that_keeps_its_promise_is_evidence():
    got = carry_out(_one(_world(0.60)), _world(0.20), score=lambda w: w["score"])
    assert got.verdict == "moved" and got.informative
    assert got.moved == pytest.approx(0.40, abs=1e-6)


def test_an_intervention_that_breaks_its_promise_is_evidence_about_nothing():
    """V.83's ceiling check, generalised: the quantity is declared, not hardcoded."""
    got = carry_out(_one(_world(0.60, keep=0.10)), _world(0.20), score=lambda w: w["score"])
    assert got.verdict == "spoiled"
    assert not got.informative, "a large number from a broken experiment is still not evidence"
    assert got.broke == ["keep"]
    assert "changed more than it meant to" in got.says


def test_the_dangerous_direction_is_caught_too():
    """The score went up **and** the promise broke. This is the one that names the wrong cause."""
    got = carry_out(_one(_world(0.90, keep=0.20)), _world(0.20), score=lambda w: w["score"])
    assert got.verdict == "spoiled" and got.moved >= MOVED


def test_a_promise_nobody_can_read_is_not_a_promise_kept():
    """A check that could not run is reported as not checked — never as passed, never as failed."""
    blind = Held(name="unmeasurable", read=None)
    got = carry_out(_one(_world(0.60), holds=(blind,)), _world(0.20),
                    score=lambda w: w["score"])
    assert got.verdict == "unvouched"
    assert got.unread == ["unmeasurable"] and not got.broke
    assert not got.informative


def test_an_intervention_that_cannot_run_says_so():
    broken = Intervention(verb="remove", on="a knob", run=lambda: 1 / 0)
    got = carry_out(broken, _world(0.20), score=lambda w: w["score"])
    assert not got.ran and got.verdict == "not run"
    assert "ZeroDivisionError" in got.says


def test_an_oracle_intervention_is_marked_as_one():
    """It answers where the cause is. It says nothing about how to remove it, and must not claim to."""
    got = carry_out(_one(_world(0.90), deployable=False), _world(0.20),
                    score=lambda w: w["score"])
    assert got.intervention is not None and not got.intervention.deployable
    assert "[oracle]" in got.render()
    assert got.informative, "an oracle is still a valid experiment"


# --------------------------------------------------------------------------------------------- #
#  the map
# --------------------------------------------------------------------------------------------- #
def _map(entered):
    knobs = [Knob("a", "given"), Knob("b", "method"), Knob("c", "built"), Knob("d", "built")]
    what = [Intervention(verb="add", on=name, run=lambda: _world(0.20)) for name in entered]
    return survey("m", knobs, what, _world(0.20), score=lambda w: w["score"])


def test_a_region_with_a_foot_in_it_is_not_a_blind_spot():
    """Reporting every unvaried knob would fire on every map ever drawn."""
    assert [g.region for g in gaps(_map(["a", "b", "c"]))] == []
    assert [g.region for g in gaps(_map(["a", "b"]))] == ["built"]


def test_a_region_entered_by_a_spoiled_experiment_is_still_empty():
    """A naive coverage count marks it done. It measured nothing, so it is exactly as unexamined."""
    knobs = [Knob("a", "given"), Knob("c", "built")]
    tried = Intervention(verb="remove", on="c", holds=(HOLD,),
                         run=lambda: _world(0.60, keep=0.05))
    drawn = survey("m", knobs, [Intervention(verb="add", on="a", run=lambda: _world(0.20)), tried],
                   _world(0.20), score=lambda w: w["score"])
    found = gaps(drawn)
    assert [g.region for g in found] == ["built"]
    assert found[0].spoiled == ["remove(c) hold(keep)"]
    assert "built" not in drawn.covered


def test_regions_are_not_ranked_by_size():
    """Deleted, not tuned. A region's size says nothing about whether the cause is in it.

    Two empty regions come back in a fixed order with **both** reported, because there is no
    evidence on the map that separates them and inventing a preference would be a guess.
    """
    knobs = [Knob("a", "given"), Knob("b", "scored"),
             Knob("c", "built"), Knob("d", "built"), Knob("e", "built")]
    drawn = survey("m", knobs, [Intervention(verb="add", on="a", run=lambda: _world(0.20))],
                   _world(0.20), score=lambda w: w["score"])
    assert [g.region for g in gaps(drawn)] == ["built", "scored"]


def test_a_proposal_carries_what_the_last_attempt_broke():
    """The map remembers. Re-entering a region arrives holding the promise that failed there."""
    knobs = [Knob("a", "given"), Knob("c", "built")]
    tried = Intervention(verb="remove", on="c", holds=(HOLD,),
                         run=lambda: _world(0.60, keep=0.05))
    drawn = survey("m", knobs, [tried], _world(0.20), score=lambda w: w["score"])
    asked = propose(drawn)
    assert asked, "a region nobody examined should produce something to try"
    assert all("keep" in [h.name for h in one.holds] for one in asked)
    assert all(one.verb != "hold" for one in asked), "`hold` is a promise, not the change"


def test_an_empty_map_claims_nothing():
    blank = Map()
    assert blank.touched == [] and blank.covered == [] and gaps(blank) == []
    assert "not what is missing" in blank.render()
    assert set(REGIONS) and set(VERBS) and 0.0 < MOVED < 0.5 and 0.0 < SLACK < 0.5


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
def test_the_detector_passes_its_retrodiction():
    got = examine()
    assert got["found"] == got["with_a_gap"] == 3
    assert got["missed"] == 0
    assert got["quiet"] == got["no_gap"] == 3, "a covered map must name nothing"
    assert got["false_alarms"] == 0
    assert got["passes"]


def test_the_exam_contains_cases_where_silence_is_the_right_answer():
    """Half of them. Without these the exam is passed by a function that returns every region."""
    assert sum(1 for c in KNOWN if not c.region) == 3


def test_shouting_blind_spot_at_everything_fails_the_exam():
    """The check on the check: a detector with no discrimination must not score well here."""
    import nyxara.njp.space as space

    real = space.Map.touched
    try:
        space.Map.touched = property(lambda self: [])      # nothing is ever covered
        got = retrodict()
        assert got["false_alarms"] > 0, "an indiscriminate detector must be caught"
    finally:
        space.Map.touched = real
    assert examine()["passes"], "and the real one must still pass afterwards"
