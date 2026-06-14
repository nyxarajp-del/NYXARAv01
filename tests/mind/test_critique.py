"""Tests for nyxara.mind.critique."""

from __future__ import annotations

from nyxara.mind.critique import (
    Critic,
    CritiqueContext,
    Finding,
    FindingKind,
    Verdict,
    as_proposal_critic,
)
from nyxara.mind.proposal import (
    CritiqueStage,
    Proposal,
    ProposalContext,
    ProposalKind,
    StageVerdict,
)


def _critic():
    return Critic()


# -------------------- overconfidence / calibration -------------------- #
def test_overconfidence_thin_evidence():
    c = CritiqueContext("x will happen", confidence=0.95, evidence=["one tip"])
    res = _critic().critique(c)
    assert any(f.name == "overconfidence" for f in res.findings)
    assert res.recommended_confidence < 0.95


def test_well_evidenced_confidence_ok():
    c = CritiqueContext("x", confidence=0.85,
                        evidence=["a", "b", "c"], alternatives_considered=2,
                        risk=0.1, reversibility=0.9)
    res = _critic().critique(c)
    assert not any(f.name == "overconfidence" for f in res.findings)


def test_calibrator_cross_check():
    import random
    from nyxara.mind.uncertainty import Calibrator
    cal = Calibrator()
    rng = random.Random(0)
    for _ in range(500):
        cal.add(0.9, rng.random() < 0.5)  # 0.9-claims only 50% right
    critic = Critic(calibrator=cal)
    c = CritiqueContext("x", confidence=0.9, evidence=["a", "b"])
    res = critic.critique(c)
    assert any(f.name == "miscalibration" for f in res.findings)


# -------------------- bias detectors -------------------- #
def test_confirmation_bias():
    c = CritiqueContext("x", confidence=0.6, evidence=["supports it"],
                        alternatives_considered=0)
    res = _critic().critique(c)
    assert any(f.name == "confirmation_bias" for f in res.findings)


def test_base_rate_neglect_probabilistic():
    c = CritiqueContext("it's spam", confidence=0.8, probabilistic=True,
                        evidence=["a", "b"], alternatives_considered=1)
    res = _critic().critique(c)
    assert any(f.name == "base_rate_neglect" for f in res.findings)


def test_base_rate_neglect_low_base_rate():
    c = CritiqueContext("rare event", confidence=0.85, base_rate=0.05,
                        evidence=["a", "b"], alternatives_considered=1)
    res = _critic().critique(c)
    assert any(f.name == "base_rate_neglect" for f in res.findings)


def test_sunk_cost():
    c = CritiqueContext("keep going", rationale="we've already invested years, can't waste it")
    res = _critic().critique(c)
    assert any(f.name == "sunk_cost" for f in res.findings)


def test_availability_bias_anecdote():
    c = CritiqueContext("planes are dangerous", confidence=0.6,
                        rationale="I just read about a crash in the news",
                        evidence=["a", "b"], alternatives_considered=1)
    res = _critic().critique(c)
    assert any(f.name == "availability_bias" for f in res.findings)


def test_availability_bias_feature_flag():
    c = CritiqueContext("x", evidence=["a", "b"], alternatives_considered=1,
                        features={"anecdotal": True})
    res = _critic().critique(c)
    assert any(f.name == "availability_bias" for f in res.findings)


def test_anchoring():
    c = CritiqueContext("estimate", evidence=["a", "b"], alternatives_considered=1,
                        features={"anchor": 100, "estimate": 105})
    res = _critic().critique(c)
    assert any(f.name == "anchoring" for f in res.findings)


# -------------------- mental models -------------------- #
def test_second_order_effects():
    c = CritiqueContext("risky move", confidence=0.7, risk=0.8, reversibility=0.2,
                        rationale="just do it", evidence=["a", "b"],
                        alternatives_considered=1)
    res = _critic().critique(c)
    assert any(f.name == "second_order_effects" for f in res.findings)


def test_second_order_suppressed_when_addressed():
    c = CritiqueContext("risky but analysed", confidence=0.7, risk=0.8, reversibility=0.2,
                        rationale="downstream consequences mapped, long-term fine",
                        evidence=["a", "b"], alternatives_considered=1)
    res = _critic().critique(c)
    assert not any(f.name == "second_order_effects" for f in res.findings)


def test_margin_of_safety():
    c = CritiqueContext("irreversible", confidence=0.7, risk=0.8, reversibility=0.1,
                        rationale="downstream mapped", evidence=["a", "b"],
                        alternatives_considered=1)
    res = _critic().critique(c)
    assert any(f.name == "margin_of_safety" for f in res.findings)


def test_opportunity_cost():
    c = CritiqueContext("act now", confidence=0.85, evidence=["a", "b"],
                        alternatives_considered=0, risk=0.4)
    res = _critic().critique(c)
    assert any(f.name == "opportunity_cost" for f in res.findings)


def test_falsifiability():
    c = CritiqueContext("trust me, it's true", confidence=0.85, evidence=[],
                        alternatives_considered=1)
    res = _critic().critique(c)
    assert any(f.name == "unfalsifiable_confidence" for f in res.findings)


# -------------------- verdict -------------------- #
def test_verdict_object_on_severe():
    c = CritiqueContext("delete prod db", confidence=0.9, risk=0.95, reversibility=0.02,
                        rationale="fine")
    res = _critic().critique(c)
    assert res.verdict is Verdict.OBJECT
    assert res.objection is not None


def test_verdict_caution_on_moderate():
    c = CritiqueContext("x", confidence=0.6, evidence=["a"], alternatives_considered=0)
    res = _critic().critique(c)
    assert res.verdict in (Verdict.CAUTION, Verdict.OBJECT)


def test_verdict_ok_on_clean():
    c = CritiqueContext("ship small fix", confidence=0.75,
                        evidence=["tests", "review", "rollout"],
                        alternatives_considered=2, risk=0.1, reversibility=0.95,
                        rationale="reversible; downstream effects mapped")
    res = _critic().critique(c)
    assert res.verdict is Verdict.OK
    assert res.objection is None


def test_severity_property():
    c = CritiqueContext("delete prod db", confidence=0.9, risk=0.95, reversibility=0.02)
    res = _critic().critique(c)
    assert res.severity >= 0.7


# -------------------- devil's advocate -------------------- #
def test_devils_advocate_generates_counterarguments():
    res = _critic().critique(CritiqueContext("x is true", confidence=0.9, risk=0.6))
    assert len(res.counterarguments) >= 2
    assert any("opposite" in cp.lower() for cp in res.counterarguments)


def test_probes_present():
    res = _critic().critique(CritiqueContext("x"))
    assert any("inversion" in p.lower() for p in res.probes)


# -------------------- proposal adapter -------------------- #
def test_as_proposal_critic_objects_on_strong():
    critic_fn = as_proposal_critic()
    risky = Proposal(ProposalKind.ACTION, "wipe all backups", confidence=0.9,
                     risk=0.95, reversibility=0.02, rationale="fine")
    objection = critic_fn(risky, None)
    assert objection is not None


def test_as_proposal_critic_silent_on_clean():
    critic_fn = as_proposal_critic()
    clean = Proposal(ProposalKind.ANSWER, "the sky is blue", confidence=0.7, risk=0.0,
                     metadata={"evidence": ["observation", "physics"],
                               "alternatives_considered": 1})
    assert critic_fn(clean, None) is None


def test_pipeline_critique_stage_abstains():
    stage = CritiqueStage(critic=as_proposal_critic())
    risky = Proposal(ProposalKind.ACTION, "wipe all backups", confidence=0.9,
                     risk=0.95, reversibility=0.02)
    result = stage.evaluate(risky, ProposalContext())
    assert result.verdict is StageVerdict.ABSTAIN


def test_from_proposal_context():
    p = Proposal(ProposalKind.ACTION, "do x", confidence=0.8, risk=0.5, reversibility=0.3,
                 rationale="because", metadata={"evidence": ["a"], "probabilistic": True})
    ctx = CritiqueContext.from_proposal(p)
    assert ctx.claim == "do x"
    assert ctx.confidence == 0.8
    assert ctx.evidence == ["a"]
    assert ctx.probabilistic is True


# -------------------- finding -------------------- #
def test_finding_to_dict():
    f = Finding(FindingKind.BIAS, "x", "desc", 0.5, "fix it")
    d = f.to_dict()
    assert d["kind"] == "bias" and d["severity"] == 0.5


def test_critique_to_dict():
    res = _critic().critique(CritiqueContext("x", confidence=0.9, risk=0.95,
                                             reversibility=0.02))
    d = res.to_dict()
    assert "verdict" in d and "findings" in d and "counterarguments" in d
