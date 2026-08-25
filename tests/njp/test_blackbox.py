"""Phase 6: the flight recorder, and the half of the milestone that did not exist.

The milestone is *"she identifies her own weakness and creates targeted training"*, and it was
half true. Three organs could already name a weakness — :mod:`nyxara.njp.selfmodel` names the
capability, :mod:`nyxara.njp.curriculum` names the blocked rung, and (as of this phase)
:mod:`nyxara.njp.blackbox` names the *condition* under which one of her strategies fails. Nothing
did anything with any of them. A weakness she can name and does not act on is a diagnosis, not a
curriculum.

Three bugs of mine are pinned here as regressions, because each of them produced a recorder that
looked like it worked:

* **Grading from ``loop.correct``** charged the manifold's next-state anticipation — a claim about
  the substrate — to the strategy that wrote the sentence. It read 38 failures in 44 rows on a
  session where the answers were mostly right.
* **Grading on the turn the correction arrives** attributes it to a *statement*, which chose no
  strategy, so the join that makes a failure mode possible never forms. Measured: six real
  failures, zero pairs.
* **``report.current`` is a ``StageResult``, not a ``Stage``**, so every rung keyed as ``stage:?``
  — one key for all of them, and a training task that could never close by moving rung.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp import NJPBrain
from nyxara.njp.blackbox import BlackBox, Episode
from nyxara.njp.curriculum import Curriculum, Report, Stage, StageResult
from nyxara.njp.metareason import Solution


def _thought(*, stimulus: str = "why did it rain?", strategy: str = "derive",
             kind: str = "causal", answer: str = "because the air cooled",
             verdict: str = "", grounded: bool = True, familiar: bool = True,
             act: str = "ask", correct: int = 0, corrections: int = 0):
    """One turn shaped the way the brain actually shapes it.

    ``correct``/``corrections`` are on the *loop* report, where the substrate's numbers live. They
    are passed here precisely so the tests can prove the recorder does not read them.
    """
    grounding = SimpleNamespace(triples=[("a", "causes", "b")] if grounded else [])
    percept = SimpleNamespace(grounding=grounding,
                              anticipated=SimpleNamespace(trusted=familiar))
    return SimpleNamespace(
        stimulus=stimulus, answer=answer, epistemic="believed", epistemic_confidence=0.6,
        solution=Solution(problem=stimulus, answer=answer, kind=kind, strategy=strategy),
        judgement=SimpleNamespace(verdict=verdict),
        act=SimpleNamespace(kind=act),
        percept=percept,
        loop=SimpleNamespace(turn=1, repaired=0, correct=correct, corrections=corrections))


# --------------------------------------------------------------------------- #
# one row per act of thinking
# --------------------------------------------------------------------------- #

def test_a_turn_that_chose_and_answered_is_a_row():
    box = BlackBox()
    episode = box.record(_thought())
    assert isinstance(episode, Episode)
    assert episode.strategy == "derive"
    assert episode.problem_kind == "causal"
    assert box.stats()["recorded"] == 1


def test_a_turn_that_neither_chose_nor_answered_is_not_a_row():
    """Greetings are not acts of thinking she can be held to, and they would fill the buffer."""
    box = BlackBox()
    assert box.record(_thought(strategy="", answer="")) is None
    assert box.stats()["recorded"] == 0


def test_the_situation_is_recorded_beside_the_choice():
    """The conditions are the axes a failure mode lives on; without them there is only an average."""
    box = BlackBox()
    episode = box.record(_thought(grounded=False, familiar=False))
    assert set(episode.conditions) == {"problem=causal", "act=ask", "ungrounded", "unfamiliar"}


def test_a_broken_thought_costs_a_row_and_never_a_turn():
    box = BlackBox()
    assert box.record(object()) is None
    assert box.grade(object(), correct=True) is None


# --------------------------------------------------------------------------- #
# what counts as failure — the two grading bugs
# --------------------------------------------------------------------------- #

def test_the_substrate_is_not_charged_to_the_strategy():
    """Regression: 38 failures in 44 rows on a session where the answers were mostly right.

    ``loop.correct`` counts the manifold anticipating the next state. Reading it here grades the
    reasoning that produced the sentence on whether the fabric recognised the ground.
    """
    box = BlackBox()
    episode = box.record(_thought(correct=0, corrections=3))
    assert episode.corrected is None
    assert not episode.scored
    assert box.stats()["failed"] == 0


def test_an_abstention_is_not_a_failure():
    """Declining to answer is the right outcome for a question she cannot settle."""
    box = BlackBox()
    episode = box.record(_thought(answer="", strategy="derive", verdict=""))
    assert not episode.failed
    assert not episode.scored


def test_the_grade_lands_on_the_answering_turn():
    """Regression: six real failures, zero pairs.

    She answers on one turn; the Master states the fact three turns later. The correcting turn
    chose no strategy — it is a statement — so a grade read off *it* forms no join at all.
    """
    box = BlackBox()
    answered = box.record(_thought(stimulus="why did it rain?"))
    box.record(_thought(stimulus="the air cooled", strategy="recall", answer="noted"))
    box.grade(Solution(problem="why did it rain?", strategy="derive"), correct=False)
    assert answered.corrected is True
    assert answered.failed
    assert ("problem=causal", "derive") in box._pairs
    assert box.stats()["pairs"] > 0


def test_reality_does_not_vote_twice():
    box = BlackBox()
    box.record(_thought())
    solution = Solution(problem="why did it rain?", strategy="derive")
    box.grade(solution, correct=False)
    box.grade(solution, correct=True)
    assert box._strategies["derive"] == [1, 1]


def test_a_grade_for_an_answer_never_given_finds_nothing():
    box = BlackBox()
    box.record(_thought())
    assert box.grade(Solution(problem="something else", strategy="derive"), correct=False) is None


# --------------------------------------------------------------------------- #
# a failure mode is a claim about a join
# --------------------------------------------------------------------------- #

def _conditional(box: BlackBox, *, fails: int, succeeds: int) -> None:
    """`derive` under two grounding conditions, failing only under one of them."""
    for i in range(fails):
        box.record(_thought(stimulus=f"ungrounded {i}", grounded=False, verdict="refuted"))
    for i in range(succeeds):
        box.record(_thought(stimulus=f"grounded {i}", grounded=True, verdict="established"))


def test_a_conditional_failure_is_reported_as_a_mode():
    box = BlackBox()
    _conditional(box, fails=4, succeeds=4)
    modes = box.failure_modes()
    worst = box.weakest_condition()
    assert worst is not None and worst in modes
    assert worst.strategy == "derive"
    assert worst.condition == "ungrounded"
    assert worst.rate == 1.0
    assert worst.baseline == 0.5
    assert "derive fails 100% of the time when ungrounded, against 50% overall (4/4)" == worst.why()


def test_a_strategy_that_fails_everywhere_has_no_mode():
    """Weak is not the same as conditionally broken, and `selfmodel` already reports weak."""
    box = BlackBox()
    _conditional(box, fails=6, succeeds=0)
    assert box._strategies["derive"] == [6, 6]
    assert box.failure_modes() == []


def test_too_few_rows_is_not_a_finding():
    box = BlackBox()
    _conditional(box, fails=3, succeeds=4)
    assert box.failure_modes() == []
    _conditional(box, fails=1, succeeds=0)
    assert box.failure_modes()


def test_the_counts_outlive_the_buffer():
    """A long session degrades to statistics, not to memory pressure — or to silence."""
    box = BlackBox(capacity=16)
    _conditional(box, fails=12, succeeds=12)
    assert len(box.episodes) == 16
    assert box.recorded == 24
    assert box.weakest_condition() is not None


# --------------------------------------------------------------------------- #
# it records inside a real turn
# --------------------------------------------------------------------------- #

def test_the_brain_records_what_it_thinks():
    brain = NJPBrain()
    for line in ("birds need water", "a sparrow is a bird", "what does a sparrow need?",
                 "why do birds need water?", "what is a sparrow?"):
        brain.think(line)
    stats = brain.blackbox.stats()
    assert stats["recorded"] > 0
    assert stats["episodes"] == stats["recorded"]
    assert any(e.strategy for e in brain.blackbox.episodes)


def test_the_loop_routes_reality_back_to_the_recorder():
    """The same join `metareason` gets. If only one of them is told, the two disagree by design."""
    brain = NJPBrain()
    session = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
               "sparrows need water", "a crow is a bird", "what does a crow need?",
               "crows need water", "a robin is a bird", "what does a robin need?",
               "robins need water"]
    for turn in range(40):
        brain.think(session[turn % len(session)])
    assert brain.loop.totals["strategies_graded"] > 0
    assert any(e.corrected is not None for e in brain.blackbox.episodes)


# --------------------------------------------------------------------------- #
# a named weakness becomes work — the half that did not exist
# --------------------------------------------------------------------------- #

def _report(letter: str = "B", name: str = "representation", blocked: str = "") -> Report:
    """A curriculum report standing on one rung, shaped the way `assess` shapes it."""
    stage = Stage(letter, name, "observations → concepts", "concepts", "concepts", 3,
                  "concepts", "observations", 12)
    result = StageResult(stage=stage, value=1.0, samples=10.0, mastered=False)
    return Report(results=[result])


def test_a_named_weakness_becomes_a_task_under_her_own_mission():
    brain = NJPBrain()
    for line in ("birds need water", "what does a sparrow need?", "a sparrow is a bird"):
        brain.think(line)
    report = _report()
    added = brain.loop._train_on_weakness(report)
    assert added > 0
    names = [n.name for n in brain.goals.nodes.values()]
    assert any(name.startswith("unblock representation:") for name in names)


def test_one_weakness_is_one_piece_of_work():
    """Assessment runs on a cadence; a new task per assessment is a tree that grows forever."""
    brain = NJPBrain()
    brain.think("birds need water")
    report = _report()
    first = brain.loop._train_on_weakness(report)
    again = brain.loop._train_on_weakness(report)
    assert first > 0
    assert again == 0


def test_a_stage_key_carries_its_letter():
    """Regression: `report.current` is a StageResult, so `getattr(current, 'letter')` was always
    absent and every rung keyed as `stage:?`. One key for nine rungs means moving up the ladder
    reads as the same weakness persisting, and the task never closes."""
    brain = NJPBrain()
    brain.think("birds need water")
    keys = {key for key, _name in brain.loop._weaknesses(_report("B", "representation"))}
    assert "stage:B" in keys
    assert "stage:?" not in keys
    later = {key for key, _name in brain.loop._weaknesses(_report("D", "causality"))}
    assert "stage:D" in later


def test_a_training_task_closes_only_when_the_weakness_stops_being_named():
    """Not when she has practised. "I worked on it" is what a system says when it cannot tell
    whether it improved, and the weakness was named with a measurement precisely so the same
    measurement could close it."""
    brain = NJPBrain()
    brain.think("birds need water")
    brain.loop._train_on_weakness(_report("B", "representation"))
    node = brain.goals.nodes[brain.loop._trained["stage:B"]]
    assert str(node.state) != "done"
    before = brain.loop._closed_weaknesses

    mastered = Report(results=[StageResult(stage=Stage("B", "representation"), mastered=True)])
    brain.loop._train_on_weakness(mastered)
    assert "stage:B" not in brain.loop._trained
    assert brain.loop._closed_weaknesses > before
    assert brain.goals.nodes[node.nid].own_progress == 1.0


def test_all_three_organs_that_name_a_weakness_feed_the_curriculum():
    """They fail differently on purpose. A capability score is an average over everything, a
    blocked stage is a threshold on one metric, a failure mode is a join — and a brain watching
    only the average keeps practising what it is mediocre at and never notices the one condition
    it is reliably broken under."""
    brain = NJPBrain()
    session = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
               "sparrows need water", "why do birds need water?"]
    for turn in range(30):
        brain.think(session[turn % len(session)])
    _conditional(brain.blackbox, fails=4, succeeds=4)

    kinds = {key.split(":", 1)[0] for key, _ in brain.loop._weaknesses(_report())}
    assert "mode" in kinds
    assert "stage" in kinds
    weakest = brain.self_model.weakest()
    if weakest is not None and weakest.weak:
        assert "capability" in kinds


def test_the_loop_reports_what_it_committed_to():
    brain = NJPBrain()
    brain.think("birds need water")
    brain.loop._train_on_weakness(_report())
    assert "weaknesses_trained" in brain.loop.totals
    assert "weaknesses_closed" in brain.loop.totals


def test_a_live_curriculum_report_is_the_shape_the_trainer_reads():
    """The trainer is fed by `Curriculum.assess`, not by the fixture above, and a change to one
    that is not a change to the other is exactly how `stage:?` survived."""
    brain = NJPBrain()
    brain.think("birds need water")
    report = Curriculum().assess(brain)
    keys = {key for key, _name in brain.loop._weaknesses(report)}
    assert any(key.startswith("stage:") for key in keys)
    assert "stage:?" not in keys


# --------------------------------------------------------------------------- #
# and it is readable from outside
# --------------------------------------------------------------------------- #

def test_the_recorder_appears_in_the_brains_own_report():
    """A recorder nothing can read is a log file. `stats()` is where the Master looks."""
    brain = NJPBrain()
    for line in ("birds need water", "what does a sparrow need?", "a sparrow is a bird"):
        brain.think(line)
    block = brain.stats().get("blackbox")
    assert isinstance(block, dict)
    assert block["recorded"] == brain.blackbox.recorded


def test_both_phase_six_arrows_are_named_in_the_pipeline_report():
    """That report exists to name what *cannot* happen, and both halves of this milestone are
    exactly the kind of thing it is for: a grade that never reaches the turn that chose, and a
    weakness that never becomes work."""
    fresh = NJPBrain().pipeline_report()
    assert fresh["outcome→failure mode"]["state"] == "open"
    assert fresh["weakness→training"]["state"] == "open"
    assert fresh["outcome→failure mode"]["why"]
    assert fresh["weakness→training"]["why"]

    brain = NJPBrain()
    session = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
               "sparrows need water", "why do birds need water?"]
    for turn in range(45):
        brain.think(session[turn % len(session)])
    report = brain.pipeline_report()
    assert report["outcome→failure mode"]["state"] == "closed"
    assert report["weakness→training"]["state"] == "closed"


def test_a_brain_with_the_recorder_gated_off_is_absent_not_broken():
    brain = NJPBrain(config=SimpleNamespace(blackbox_enabled=False))
    brain.think("birds need water")
    assert brain.blackbox is None
    assert "blackbox" not in brain.stats()
    assert "outcome→failure mode" not in brain.pipeline_report()
