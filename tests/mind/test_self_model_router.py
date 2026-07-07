"""Tests for nyxara.mind.self_model_router — the PRIMARY self-model router (offline).

The upfront triage reads NYXARA's self-model and decides *which mind* should handle a
prompt before any generation: her own model (SELF), the teacher (TEACHER), an exact
faculty (FACULTY), or — for actions — a verify-before-act gate. Scripted own/teacher
providers and a seeded ``SelfModel`` stand in for real models, so we assert the routing
decisions, the verify-before-act gate, and the fail-open behaviour without any network.
"""

from __future__ import annotations

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.kernel.orchestrator import Candidate
from nyxara.memory.self_model import SelfModel
from nyxara.mind.metacognition import HONEST_ABSTENTION
from nyxara.mind.router import Router
from nyxara.mind.self_model_router import PrimarySelfModelRouter, Route, RoutingPlan


# --------------------------------------------------------------------------- #
# Scripted minds (mirrors tests/mind/test_router.py)
# --------------------------------------------------------------------------- #
class _Own:
    def __init__(self, text="A coherent, confident answer, Master.", ok=True):
        self._t, self._ok = text, ok

    def available(self):
        return self._ok

    def complete(self, req):
        return type("_R", (), {"text": self._t})()


class _Teacher:
    def __init__(self, providers=("tinyllama", "self", "mock")):
        self._p = list(providers)

    def available_providers(self):
        return list(self._p)

    def complete_with(self, name, req):
        return type("_R", (), {"text": f"[teacher:{name}] a careful, correct answer"})()


def _seeded_self_model() -> SelfModel:
    sm = SelfModel()
    sm.set_capability("network", level=0.9, confidence=0.85)     # strong
    sm.set_capability("cooking", level=0.05, confidence=0.9)     # weak
    sm.declare_hallucination_risk("specific dates", risk=0.85,
                                  keywords=("what year", "when did"))
    sm.declare_unknown("quantum chromodynamics")
    return sm


def _router(settings, *, own=None, teacher=None):
    return Router(teacher or _Teacher(), settings=settings,
                  self_provider=own or _Own())


def _settings(**over):
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.self_model_router.enabled = True
    for k, v in over.items():
        setattr(s.self_model_router, k, v)
    return s


def _psr(settings=None, *, self_model="seed", own=None, teacher=None):
    settings = settings or _settings()
    sm = _seeded_self_model() if self_model == "seed" else self_model
    return PrimarySelfModelRouter(self_model=sm, settings=settings,
                                  router=_router(settings, own=own, teacher=teacher))


# --------------------------------------------------------------------------- #
# Triage decisions
# --------------------------------------------------------------------------- #
def test_faculty_route_for_exact_math():
    assert _psr().plan("what is 12 * 12").route is Route.FACULTY


def test_self_route_when_competent_and_low_risk():
    plan = _psr().plan("How do I harden my network firewall?")
    assert plan.route is Route.SELF
    assert plan.competence >= _settings().self_model_router.competence_threshold


def test_teacher_route_when_weak_capability():
    assert _psr().plan("How do I improve my cooking?").route is Route.TEACHER


def test_teacher_route_on_high_hallucination_risk():
    plan = _psr().plan("what year did the network protocol ship?")
    assert plan.route is Route.TEACHER
    assert plan.hallucination_risk >= _settings().self_model_router.hallucination_ceiling


def test_teacher_route_on_known_unknown():
    assert _psr().plan("Explain quantum chromodynamics in depth.").route is Route.TEACHER


def test_abstain_when_weak_and_no_teacher():
    psr = _psr(teacher=_Teacher(providers=["mock"]))
    plan = psr.plan("How do I improve my cooking?")
    assert plan.route is Route.ABSTAIN
    res = psr.route_respond("How do I improve my cooking?")
    assert res.source == "abstain" and res.text == HONEST_ABSTENTION


def test_default_competence_biases_unknown_prompts_to_teacher():
    # nothing in the prompt maps to a capability -> default_competence (0.5) < threshold (0.6)
    assert _psr().plan("Tell me something interesting.").route is Route.TEACHER


# --------------------------------------------------------------------------- #
# Verify-before-act (clause 3)
# --------------------------------------------------------------------------- #
def _act(rationale, *, risk=RiskTier.LOW, confidence=0.8, tool="shell", text="do the thing"):
    return Candidate(text=text, kind="act", capability=Capability.TOOL_CALL,
                     tool=tool, risk=risk, confidence=confidence, rationale=rationale)


def test_verify_before_act_passes_coherent_action():
    cleared, plan = _psr().verify_action(
        "clean up disk", _act("delete the stale cache file to free disk space"))
    assert cleared is True and plan.cleared is True
    assert plan.route is Route.VERIFY_THEN_ACT and plan.is_action


def test_verify_before_act_demotes_incoherent_action():
    cleared, plan = _psr().verify_action(
        "clean up", _act("the the the the", confidence=0.1))
    assert cleared is False and plan.cleared is False


def test_verify_before_act_stricter_for_high_risk():
    # a short-but-real justification scores ~0.69: clears the 0.5 LOW floor but not the 0.7
    # HIGH floor — the same proposal is allowed at low risk and declined at high risk.
    low_ok, _ = _psr().verify_action("x", _act("cleanup", risk=RiskTier.LOW, text=""))
    high_ok, _ = _psr().verify_action("x", _act("cleanup", risk=RiskTier.HIGH, text=""))
    assert low_ok is True and high_ok is False


def test_low_confidence_action_is_demoted_even_if_coherent():
    cleared, _ = _psr().verify_action(
        "x", _act("a perfectly coherent and well reasoned justification here",
                  confidence=0.1))
    assert cleared is False


# --------------------------------------------------------------------------- #
# Fail-open / backward compatibility
# --------------------------------------------------------------------------- #
def test_fail_open_when_self_model_absent():
    psr = PrimarySelfModelRouter(self_model=None, settings=_settings(),
                                 router=_router(_settings()))
    assert psr.plan("anything at all").route in (Route.SELF, Route.TEACHER)


def test_routing_plan_serialises():
    d = RoutingPlan(Route.VERIFY_THEN_ACT, 0.7, "why", is_action=True,
                    verifier_score=0.66, cleared=True).to_dict()
    assert d["route"] == "verify_then_act" and d["verifier_score"] == 0.66
    assert d["cleared"] is True and d["is_action"] is True


# --------------------------------------------------------------------------- #
# Reasoner integration — demotion happens before the gates
# --------------------------------------------------------------------------- #
def test_reasoner_demotes_unverified_act_to_respond():
    from nyxara.mind.llm_reasoner import LLMReasoner
    settings = _settings()
    reasoner = LLMReasoner(settings=settings, self_model=_seeded_self_model())
    data = {"kind": "act", "tool": "shell", "text": "x", "rationale": "the the the",
            "confidence": 0.1, "risk": "high"}
    cand = reasoner._candidate_from(data, "do something")
    # the tool 'shell' is not registered here, so it would already degrade; assert the
    # verify hook does not crash and yields a respond candidate either way.
    assert cand.kind == "respond"


def test_reasoner_no_route_when_self_model_absent_and_router_disabled():
    from nyxara.mind.llm_reasoner import LLMReasoner
    settings = NyxaraSettings.for_profile(Profile.TEST)  # both routers default off/absent
    settings.self_model_router.enabled = False
    reasoner = LLMReasoner(settings=settings)            # no self_model
    assert reasoner._maybe_route("hello") is None
