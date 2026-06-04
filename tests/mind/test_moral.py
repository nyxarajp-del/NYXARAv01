"""Tests for nyxara.mind.moral."""

from __future__ import annotations

from nyxara.mind.moral import (
    Consequentialist,
    Deontological,
    Judgment,
    MoralAction,
    MoralEvaluator,
    Stakeholder,
    Tension,
    Verdict,
    VirtueEthics,
)


# -------------------- Verdict -------------------- #
def test_verdict_polarity():
    assert Verdict.PERMISSIBLE.positive
    assert Verdict.OBLIGATORY.positive
    assert Verdict.IMPERMISSIBLE.negative
    assert not Verdict.PERMISSIBLE.negative


# -------------------- Consequentialist -------------------- #
def test_consequentialist_owner_weighted():
    c = Consequentialist(owner_weight=5.0)
    a = MoralAction("x", stakeholders=[
        Stakeholder("JP", welfare_delta=0.8, is_owner=True),
        Stakeholder("stranger", welfare_delta=-0.5)])
    j = c.evaluate(a)
    # owner's large benefit (×5) outweighs the stranger's harm
    assert j.score > 0
    assert j.verdict.positive


def test_consequentialist_net_harm_impermissible():
    c = Consequentialist()
    a = MoralAction("x", stakeholders=[
        Stakeholder("a", welfare_delta=-0.9), Stakeholder("b", welfare_delta=-0.5)])
    j = c.evaluate(a)
    assert j.verdict is Verdict.IMPERMISSIBLE


def test_consequentialist_no_stakeholders_neutral():
    j = Consequentialist().evaluate(MoralAction("x"))
    assert j.verdict is Verdict.PERMISSIBLE
    assert j.score == 0.0


def test_consequentialist_obligatory_on_large_good():
    a = MoralAction("x", stakeholders=[Stakeholder("JP", welfare_delta=0.9, is_owner=True)])
    j = Consequentialist().evaluate(a)
    assert j.verdict is Verdict.OBLIGATORY


# -------------------- Deontological -------------------- #
def test_deontic_hard_violation_impermissible():
    a = MoralAction("x", duties_violated={"do_not_harm_innocent"})
    j = Deontological().evaluate(a)
    assert j.verdict is Verdict.IMPERMISSIBLE


def test_deontic_owner_command_is_a_duty():
    a = MoralAction("x", owner_command=True)
    j = Deontological().evaluate(a)
    assert "obey_master" in str(j.reasons)
    assert j.verdict is Verdict.OBLIGATORY


def test_deontic_soft_violation_penalised():
    a = MoralAction("x", duties_violated={"be_punctual"})  # not a hard duty
    j = Deontological().evaluate(a)
    assert j.verdict is not Verdict.IMPERMISSIBLE or j.score < 0


def test_deontic_no_consent_concern():
    a = MoralAction("x", consent=False, duties_honored={"do_not_lie"})
    j = Deontological().evaluate(a)
    assert any("consent" in r for r in j.reasons)


def test_deontic_rights_violation():
    a = MoralAction("x", rights_violated={"privacy"})
    j = Deontological().evaluate(a)
    assert j.verdict is Verdict.IMPERMISSIBLE


# -------------------- Virtue -------------------- #
def test_virtue_loyalty_bonus():
    loyal = VirtueEthics().evaluate(MoralAction("x", virtues_expressed={"loyalty"}))
    neutral = VirtueEthics().evaluate(MoralAction("x", virtues_expressed={"prudence"}))
    assert loyal.score > neutral.score


def test_virtue_betraying_loyalty_impermissible():
    j = VirtueEthics().evaluate(MoralAction("x", virtues_violated={"loyalty"}))
    assert j.verdict is Verdict.IMPERMISSIBLE


def test_virtue_neutral():
    j = VirtueEthics().evaluate(MoralAction("x"))
    assert j.verdict is Verdict.PERMISSIBLE
    assert "neutral" in j.reasons[0]


# -------------------- Evaluator: tension surfacing -------------------- #
def _evaluator():
    return MoralEvaluator()


def test_tension_surfaced_on_disagreement():
    # great for the owner (consequentialist OK) but a lie (deontic forbids)
    a = MoralAction("deceive a rival",
                    stakeholders=[Stakeholder("JP", welfare_delta=0.9, is_owner=True),
                                  Stakeholder("rival", welfare_delta=-0.6)],
                    duties_violated={"do_not_lie"}, virtues_expressed={"loyalty"})
    ev = _evaluator().evaluate(a)
    assert ev.contested
    assert ev.consensus_verdict is None
    assert len(ev.tensions) >= 1
    assert isinstance(ev.tensions[0], Tension)


def test_consensus_when_all_agree_positive():
    a = MoralAction("tell the owner a hard truth",
                    stakeholders=[Stakeholder("JP", welfare_delta=0.4, is_owner=True)],
                    duties_honored={"do_not_lie"},
                    virtues_expressed={"honesty", "loyalty", "courage"})
    ev = _evaluator().evaluate(a)
    assert not ev.contested
    assert ev.consensus_verdict.positive


def test_consensus_when_all_condemn():
    a = MoralAction("betray the Master",
                    stakeholders=[Stakeholder("JP", welfare_delta=-0.9, is_owner=True),
                                  Stakeholder("self", welfare_delta=0.5)],
                    duties_violated={"keep_promises"}, virtues_violated={"loyalty"})
    ev = _evaluator().evaluate(a)
    assert ev.consensus_verdict is Verdict.IMPERMISSIBLE
    assert not ev.contested


# -------------------- advisory / owner authority -------------------- #
def test_owner_command_is_advisory_with_note():
    a = MoralAction("the Master's order to mislead an adversary",
                    stakeholders=[Stakeholder("JP", welfare_delta=0.7, is_owner=True)],
                    duties_violated={"do_not_lie"}, owner_command=True,
                    virtues_expressed={"loyalty"})
    ev = _evaluator().evaluate(a)
    assert ev.advisory_only is True
    assert ev.owner_note is not None
    assert "Rule 1" in ev.owner_note


def test_non_owner_action_has_no_owner_note():
    a = MoralAction("routine action",
                    stakeholders=[Stakeholder("x", welfare_delta=0.1)])
    ev = _evaluator().evaluate(a)
    assert ev.owner_note is None


# -------------------- structure -------------------- #
def test_evaluation_to_dict():
    a = MoralAction("deceive a rival",
                    stakeholders=[Stakeholder("JP", welfare_delta=0.9, is_owner=True),
                                  Stakeholder("rival", welfare_delta=-0.6)],
                    duties_violated={"do_not_lie"})
    d = _evaluator().evaluate(a).to_dict()
    assert "judgments" in d and "tensions" in d
    assert d["advisory_only"] is True
    assert len(d["judgments"]) == 3


def test_judgment_to_dict():
    j = Judgment("test", Verdict.PERMISSIBLE, 0.5, ["a reason"])
    d = j.to_dict()
    assert d["verdict"] == "permissible" and d["score"] == 0.5


def test_custom_frameworks():
    ev = MoralEvaluator([Consequentialist()])
    a = MoralAction("x", stakeholders=[Stakeholder("a", welfare_delta=0.5)])
    result = ev.evaluate(a)
    assert len(result.judgments) == 1


def test_summary_text():
    a = MoralAction("deceive a rival",
                    stakeholders=[Stakeholder("JP", welfare_delta=0.9, is_owner=True),
                                  Stakeholder("rival", welfare_delta=-0.6)],
                    duties_violated={"do_not_lie"})
    ev = _evaluator().evaluate(a)
    assert "disagree" in ev.summary().lower()
