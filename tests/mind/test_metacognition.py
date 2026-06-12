"""Tests for nyxara.mind.metacognition — the 'do I know this?' gate (Phase 4)."""

from __future__ import annotations

import pytest

from nyxara.mind.metacognition import MetaCognition, MetaDecision


def _mc():
    return MetaCognition(answer_threshold=0.6, abstain_below=0.15)


def test_faculty_wins_when_available():
    v = _mc().assess("2+2?", own_answer="four", own_conf=0.9, faculty_available=True)
    assert v.decision is MetaDecision.USE_FACULTY


def test_confident_own_answers():
    v = _mc().assess("hi", own_answer="Hello, Master.", own_conf=0.8)
    assert v.decision is MetaDecision.ANSWER_SELF


def test_low_confidence_consults_teacher():
    v = _mc().assess("hard", own_answer="a guess", own_conf=0.3, teacher_available=True)
    assert v.decision is MetaDecision.CONSULT_TEACHER


def test_abstains_when_nothing_trustworthy_and_no_teacher():
    assert _mc().assess("hard", own_answer="", own_conf=0.0).decision is MetaDecision.ABSTAIN
    assert _mc().assess("hard", own_answer="weak", own_conf=0.05).decision \
        is MetaDecision.ABSTAIN


def test_best_effort_own_when_no_teacher_but_passable():
    v = _mc().assess("hard", own_answer="a passable answer", own_conf=0.4)
    assert v.decision is MetaDecision.ANSWER_SELF


def test_calibrator_can_lower_confidence_into_abstention():
    class _Pessimist:
        def calibrate(self, c):
            return 0.05            # the model is overconfident; calibration corrects it
    mc = MetaCognition(answer_threshold=0.6, abstain_below=0.15, calibrator=_Pessimist())
    v = mc.assess("hard", own_answer="overconfident guess", own_conf=0.95)
    assert v.decision is MetaDecision.ABSTAIN
