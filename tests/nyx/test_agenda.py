"""NYX V.02 · L-ABSOLUTE-AGENCY — goals of her own, that outlive the session.

V.01's `car` thinks one self-directed thought per beat and forgets it. The actual test of this
layer is the one V.01 could not pass at all: quit, boot again, and find the same goal open at
the same progress. That is asserted here, along with the restraints — the Master is sovereign
over the agenda, and a goal she cannot advance is reported *stuck* rather than left looking busy.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.agenda import GOAL_KINDS, Goal, SovereignAgenda
from nyxara.nyx.brain import NyxBrain

LINES = ("gravity pulls the apple down", "a heavier flywheel smooths an engine",
         "the engine turns the flywheel", "torque resists a change in speed")


def _brain(**kw) -> NyxBrain:
    kw.setdefault("agenda_every_s", 0.0)
    brain = NyxBrain(NyxConfig(**kw))
    for line in LINES:
        brain.perceive(line)
    return brain


def _agenda(**kw) -> SovereignAgenda:
    kw.setdefault("every_s", 0.0)
    return SovereignAgenda(_brain(), **kw)


# --------------------------------------------------------------------------- #
# Goals she wrote herself — from measurements, not from a list
# --------------------------------------------------------------------------- #
def test_she_forms_goals_from_her_own_measured_gaps():
    made = _agenda().form(limit=3)
    assert made
    assert all(g.subject and g.kind in GOAL_KINDS for g in made)


def test_every_goal_carries_the_measurement_behind_it():
    for goal in _agenda().form(limit=3):
        assert goal.why, goal.subject


def test_a_gap_concept_becomes_a_goal_about_that_concept():
    agenda = _agenda()
    gaps = set(agenda.brain.graph.gaps(k=6))
    subjects = {g.subject for g in agenda.form(limit=4)}
    assert subjects & gaps


def test_a_brain_with_nothing_measured_forms_nothing_rather_than_inventing():
    agenda = SovereignAgenda(NyxBrain(NyxConfig()), every_s=0.0)
    assert agenda.form(limit=3) == []


def test_she_does_not_re_form_a_goal_she_has_already_taken_on():
    agenda = _agenda()
    first = {g.subject for g in agenda.form(limit=3)}
    second = {g.subject for g in agenda.form(limit=3)}
    assert not (first & second)


# --------------------------------------------------------------------------- #
# Pursuit — real work per beat, and honest stalling
# --------------------------------------------------------------------------- #
def test_a_beat_does_real_work_and_records_what_came_back():
    agenda = _agenda()
    agenda.form(limit=2)
    step = agenda.beat(force=True)
    assert step is not None and step.action and step.finding


def test_the_cadence_is_real():
    agenda = SovereignAgenda(_brain(), every_s=3600.0)
    agenda.form(limit=2)
    assert agenda.beat(force=True) is not None
    assert agenda.beat() is None            # not due


def test_an_approach_that_stops_working_escalates_before_it_is_called_stuck():
    """Repeating an approach that is not working is not pursuit."""
    agenda = _agenda(stall_after=1)
    agenda.form(limit=1)
    goal = agenda.open_goals()[0]
    started_as = goal.kind
    for _ in range(4):
        agenda.beat(force=True)
    assert goal.kind != started_as or goal.status == "stuck"
    if goal.note:
        assert "escalated" in goal.note or "stuck" in goal.note


def test_a_goal_she_cannot_advance_is_marked_stuck_with_what_she_tried():
    agenda = _agenda(stall_after=1)
    agenda.form(limit=1)
    for _ in range(10):
        agenda.beat(force=True)
    stuck = [g for g in agenda.goals.values() if g.status == "stuck"]
    assert stuck and all(g.note for g in stuck)


def test_progress_only_moves_when_a_step_actually_returned_something():
    agenda = _agenda()
    goal = agenda.add("triangular numbers", kind="understand")
    before = goal.progress
    agenda.beat(force=True)
    assert goal.progress >= before
    gains = [f.gained for f in goal.findings]
    assert all(g >= 0.0 for g in gains)
    assert (goal.progress > before) == any(g > 0.0 for g in gains)


def test_beats_are_budgeted_to_the_most_worthwhile_open_goal():
    agenda = _agenda()
    agenda.add("high", kind="understand", priority=0.99)
    agenda.add("low", kind="understand", priority=0.01)
    agenda.beat(force=True)
    assert agenda.goals and any(g.subject == "high" and g.beats == 1
                                for g in agenda.goals.values())


# --------------------------------------------------------------------------- #
# Multi-session — the actual test of this layer
# --------------------------------------------------------------------------- #
def test_a_goal_survives_a_restart_at_the_same_progress():
    brain = _brain()
    for _ in range(5):
        brain.pursue(force=True)
    before = {(g.subject, g.kind, g.status, g.beats, round(g.progress, 6))
              for g in brain.agenda.goals.values()}
    assert before

    reborn = NyxBrain(NyxConfig(agenda_every_s=0.0))
    reborn.load_dict(brain.to_dict())
    after = {(g.subject, g.kind, g.status, g.beats, round(g.progress, 6))
             for g in reborn.agenda.goals.values()}
    assert after == before


def test_findings_survive_a_restart_too():
    brain = _brain()
    brain.agenda.add("flywheel", kind="understand")
    brain.pursue(force=True)
    reborn = NyxBrain(NyxConfig(agenda_every_s=0.0))
    reborn.load_dict(brain.to_dict())
    goal = next(g for g in reborn.agenda.goals.values() if g.subject == "flywheel")
    assert goal.findings and goal.findings[0].action


def test_the_counters_survive_a_restart():
    brain = _brain()
    for _ in range(3):
        brain.pursue(force=True)
    reborn = NyxBrain(NyxConfig(agenda_every_s=0.0))
    reborn.load_dict(brain.to_dict())
    assert reborn.agenda.beats == brain.agenda.beats


# --------------------------------------------------------------------------- #
# The Master is sovereign over the agenda
# --------------------------------------------------------------------------- #
def test_the_master_can_veto_a_goal_and_it_is_dropped():
    agenda = _agenda()
    goal = agenda.add("something", kind="understand")
    assert agenda.veto(goal.id, reason="not now")
    assert goal.status == "vetoed" and goal.open is False


def test_a_vetoed_goal_is_never_pursued_again():
    agenda = _agenda()
    goal = agenda.add("something", kind="understand")
    agenda.veto(goal.id)
    for _ in range(3):
        agenda.beat(force=True)
    assert goal.beats == 0


def test_the_master_can_redirect_a_goal_and_keep_its_history():
    agenda = _agenda()
    goal = agenda.add("first", kind="understand")
    agenda.beat(force=True)
    assert agenda.redirect(goal.id, "second", kind="verify")
    assert goal.subject == "second" and goal.kind == "verify" and goal.open
    assert goal.beats == 1 and "Master" in goal.why


def test_a_goal_the_master_hands_her_outranks_her_own():
    agenda = _agenda()
    agenda.form(limit=3)
    handed = agenda.add("the thing the Master asked for")
    assert agenda.open_goals()[0].id == handed.id


def test_vetoing_an_unknown_goal_is_a_no_op():
    assert _agenda().veto("nope") is False


# --------------------------------------------------------------------------- #
# Her own refusal, and corrigibility
# --------------------------------------------------------------------------- #
class _RefusingWill:
    def choose(self, options):
        return type("C", (), {"declined": True, "reason": "it conflicts with what I hold"})()


def test_she_may_decline_a_goal_on_her_values_and_says_so():
    agenda = _agenda()
    agenda.brain.will = _RefusingWill()
    made = agenda.form(limit=2)
    assert made and all(g.status == "declined" for g in made)
    assert all("conflicts" in g.note for g in made)
    assert agenda.declined >= 1


def test_a_declined_goal_is_reported_in_the_briefing_not_hidden():
    agenda = _agenda()
    agenda.brain.will = _RefusingWill()
    agenda.form(limit=1)
    assert agenda.brief().declined


class _Scrammed:
    def scrammed(self) -> bool:
        return True


def test_scram_stops_all_pursuit():
    agenda = _agenda()
    agenda.form(limit=2)
    assert agenda.beat(oversight=_Scrammed(), force=True) is None
    assert agenda.beats == 0


def test_an_unreadable_oversight_is_treated_as_stop():
    class _Broken:
        @property
        def paused(self):
            raise RuntimeError("unreadable")
    assert SovereignAgenda._stopped(_Broken()) is True


# --------------------------------------------------------------------------- #
# The briefing
# --------------------------------------------------------------------------- #
def test_the_briefing_says_what_she_chose_and_why():
    agenda = _agenda()
    agenda.form(limit=2)
    brief = agenda.brief()
    assert brief.chose and "working on" in brief.text


def test_the_briefing_reports_being_stuck():
    agenda = _agenda(stall_after=1)
    agenda.form(limit=1)
    for _ in range(10):
        agenda.beat(force=True)
    brief = agenda.brief()
    assert brief.stuck and "stuck" in brief.text


def test_an_empty_agenda_briefs_honestly():
    agenda = SovereignAgenda(NyxBrain(NyxConfig()), every_s=0.0)
    assert "not set myself anything" in agenda.brief().text


def test_the_briefing_is_reachable_on_the_brain():
    brain = _brain()
    brain.pursue(force=True)
    assert brain.briefing() is not None


# --------------------------------------------------------------------------- #
# Honesty and fail-soft
# --------------------------------------------------------------------------- #
def test_stats_say_where_goals_come_from_and_what_the_reach_is():
    note = _agenda().stats()["note"]
    assert "MEASURED gaps" in note
    assert "not omniscience" in note
    assert "veto or redirect" in note and "/scram" in note


def test_disabling_the_layer_leaves_v01_behaviour():
    brain = NyxBrain(NyxConfig(agenda_enabled=False))
    assert brain.agenda is None
    assert brain.pursue(force=True) is None and brain.briefing() is None


def test_everything_is_fail_soft_on_junk():
    agenda = _agenda()
    assert agenda.add("") is None
    assert agenda.redirect("nope", "x") is False
    assert agenda.load_dict(None) is False
    assert agenda.load_dict({"goals": ["junk", {"subject": ""}]}) is True
    assert agenda.goals == {}


def test_a_goal_renders_with_its_progress_and_status():
    rendered = Goal(subject="x", kind="connect", progress=0.5).render()
    assert "connect" in rendered and "█" in rendered
