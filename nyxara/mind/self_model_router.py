"""NYXARA · mind/self_model_router.py — the PRIMARY self-model router (🧭 → 🧠/👑/✅).

Where ``mind/router.py`` is *reactive* — it always drafts with NYXARA's own model first,
then scores and falls back — this router is the **upfront triage**: before a single token
is generated it reads NYXARA's introspectable self-knowledge and decides *which mind should
handle this prompt at all*. The Master asked for exactly three dispositions:

  • prompts her **own self-model** should handle  → :data:`Route.SELF`
  • prompts that should go to the **teacher LLM** → :data:`Route.TEACHER`
  • action prompts that must **pass a verifier before acting** → :data:`Route.VERIFY_THEN_ACT`

plus the two honest extremes a real triage needs: an exact :data:`Route.FACULTY` (math/logic
beats any guess) and an :data:`Route.ABSTAIN` when nothing trustworthy can be said.

It is a *planner*, not a new mind: SELF/TEACHER/FACULTY generation is delegated to the
existing :class:`~nyxara.mind.router.Router` (which re-verifies on the answer's own merits,
so a wrong SELF triage can never force a bad answer through); the verify-before-act gate
reuses the same intrinsic verifier. Like everything upstream of the kernel it only chooses
*which mind drafts* / *whether an action is coherent enough to propose* — it reaches around
no gate and changes no protected attribute. The kernel still disposes every reply.

Advisory and fail-open: any failure, or the absence of a self-model / forged model / teacher,
degrades to today's behaviour rather than crashing a turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from nyxara.agency.permissions import RiskTier
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.mind.metacognition import HONEST_ABSTENTION, MetaCognition, MetaDecision
from nyxara.mind.router import RouterResult, default_verifier

__all__ = ["Route", "RoutingPlan", "PrimarySelfModelRouter"]


def _grounded_or_default() -> Any:
    """Verifier grounded in truth where decidable, else the intrinsic verifier. Never raises."""
    try:
        from nyxara.mind.grounded_verifier import grounded_verifier
        return grounded_verifier()
    except Exception:  # noqa: BLE001 — grounding is additive
        return default_verifier()


_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _risk_from_label(label: str, default: RiskTier = RiskTier.MODERATE) -> RiskTier:
    try:
        return RiskTier[str(label).strip().upper()]
    except (KeyError, AttributeError):
        return default


# --------------------------------------------------------------------------- #
# Routes & the triage plan
# --------------------------------------------------------------------------- #
class Route(str, Enum):
    SELF = "self"                       # competent + low hallucination risk -> own model
    TEACHER = "teacher"                 # weak / unknown / risky / knowledge-heavy -> teacher
    VERIFY_THEN_ACT = "verify_then_act"  # an action — must clear the verifier before acting
    FACULTY = "faculty"                 # exact math / logic fits -> compute it
    TRANSFER = "transfer"               # a new domain whose structure maps onto a known one
    GENERALIZE = "generalize"           # a novel / from-examples task her own generalizers solve
    ABSTAIN = "abstain"                 # nothing trustworthy and no one to consult


@dataclass
class RoutingPlan:
    """The upfront triage decision — auditable: *which mind*, *how sure*, and *why*."""

    route: Route
    confidence: float = 0.0
    reason: str = ""
    competence: float = 0.0
    hallucination_risk: float = 0.0
    domains: List[str] = field(default_factory=list)
    is_action: bool = False
    verifier_score: Optional[float] = None
    cleared: Optional[bool] = None
    transfer: Any = None                 # a mind.transfer.TransferResult on Route.TRANSFER
    generalization: Any = None           # a mind.generalization.GeneralizationResult on Route.GENERALIZE

    def to_dict(self) -> dict:
        return {
            "route": self.route.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "competence": round(self.competence, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "domains": list(self.domains),
            "is_action": self.is_action,
            "verifier_score": (None if self.verifier_score is None
                               else round(self.verifier_score, 4)),
            "cleared": self.cleared,
            "transfer": (self.transfer.to_dict()
                         if self.transfer is not None and hasattr(self.transfer, "to_dict")
                         else None),
            "generalization": (self.generalization.to_dict()
                               if self.generalization is not None
                               and hasattr(self.generalization, "to_dict") else None),
        }


# --------------------------------------------------------------------------- #
# The primary self-model router
# --------------------------------------------------------------------------- #
class PrimarySelfModelRouter:
    """Decide *upfront* which mind handles a prompt — then dispatch to the reactive router."""

    def __init__(self, *, self_model: Any = None, router: Any = None,
                 settings: Optional[NyxaraSettings] = None,
                 verifier: Any = None, llm: Any = None,
                 transfer_engine: Any = None, generalization_engine: Any = None) -> None:
        self.settings = settings or (getattr(router, "settings", None) or get_settings())
        self.cfg = self.settings.self_model_router
        self.self_model = self_model
        self._router = router
        self._llm = llm
        self._transfer = transfer_engine
        self._generalization = generalization_engine
        self.verifier = verifier or _grounded_or_default()
        abstain_below = getattr(self.settings.router, "abstain_below", 0.15)
        self.meta = MetaCognition(answer_threshold=self.cfg.competence_threshold,
                                  abstain_below=abstain_below)

    # ---- the reused reactive router (lazy) ---- #
    def _ensure_router(self) -> Any:
        if self._router is None:
            from nyxara.mind.router import Router  # lazy: only when actually routing
            self._router = Router(self._llm, settings=self.settings)
        return self._router

    def _teacher_available(self) -> bool:
        try:
            return bool(self._ensure_router().teacher_available())
        except Exception:  # noqa: BLE001 — availability probing is advisory
            return False

    # ---- self-knowledge reads (all fail-open to neutral values) ---- #
    def _competence_for(self, prompt: str) -> float:
        """Read NYXARA's own competence for *this* prompt from her self-model.

        Maps the prompt to a named capability (via the configured topic map and any
        capability name surfacing in the text), then weighs mastery by how sure she is of
        it. With nothing to anchor on she assumes ``default_competence`` — kept below the
        ``competence_threshold`` so ambiguous prompts bias toward consulting the teacher
        rather than bluffing."""
        sm = self.self_model
        if sm is None:
            return self.cfg.default_competence
        try:
            text = (prompt or "").lower()
            toks = set(_tokens(prompt))
            names: set = set()
            for keyword, cap_name in (self.cfg.domain_capabilities or {}).items():
                if keyword and keyword.lower() in text:
                    names.add(cap_name)
            for cap_name in getattr(sm, "capabilities", {}):
                cl = cap_name.lower()
                if cl in toks or cl in text:
                    names.add(cap_name)
            scored: List[float] = []
            for cap_name in names:
                cap = sm.capability(cap_name)
                if cap is not None:
                    scored.append(cap.level * (0.5 + 0.5 * cap.confidence))
            return max(scored) if scored else self.cfg.default_competence
        except Exception:  # noqa: BLE001 — self-reading is advisory, never fatal
            return self.cfg.default_competence

    def _knowledge_heavy(self, prompt: str) -> bool:
        """True if the prompt lands on an explicit known-unknown or a self-declared weak skill."""
        sm = self.self_model
        if sm is None:
            return False
        try:
            text = (prompt or "").lower()
            for topic in sm.what_i_dont_know():
                if topic and topic.lower() in text:
                    return True
            for name, _level in sm.where_i_am_weak(threshold=self.cfg.competence_threshold):
                if name and name.lower() in text:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _is_action(self, candidate: Any) -> bool:
        if candidate is None:
            return False
        try:
            if str(getattr(candidate, "kind", "")) == "act":
                return True
            if getattr(candidate, "tool", ""):
                return True
            risk = getattr(candidate, "risk", None)
            floor = _risk_from_label(self.cfg.verify_risk_floor)
            return risk is not None and risk >= floor
        except Exception:  # noqa: BLE001
            return False

    # ---- the triage ---- #
    def plan(self, prompt: str, *, candidate: Any = None,
             difficulty: Optional[float] = None) -> RoutingPlan:
        """Decide, before any generation, which mind should handle ``prompt``.

        ``difficulty`` is the turn's calibrated metacognitive estimate
        (:mod:`~nyxara.mind.metacontrol`), when the kernel computed one. A harder turn
        deterministically tightens the SELF gate — higher competence bar, and the estimate
        discounts effective confidence as internal uncertainty — so one shared judgment
        drives routing and compute alike.
        """
        comp_threshold = self.cfg.competence_threshold
        internal_uncertainty = 0.0
        if difficulty is not None:
            d = max(0.0, min(1.0, float(difficulty)))
            comp_threshold = min(1.0, comp_threshold + 0.15 * d)
            internal_uncertainty = 0.5 * d
        try:
            # 1) a verifiable faculty (exact math / logic) beats any neural guess
            if self.cfg.use_faculties and self._faculty_fits(prompt):
                return RoutingPlan(Route.FACULTY, 1.0,
                                   "a verifiable faculty fits — compute it exactly")

            # 1b) a demonstrated / from-examples task: solve it with her OWN generalizers up front,
            #     before any competence gate could route it to the teacher (or her own model) as
            #     mere prose. This is what lets her learn a genuinely new task from a few examples
            #     and answer a held-out input in the live turn — her faculties, not an LLM.
            if self._generalization_enabled() and self._has_demos(prompt):
                gen = self._try_generalize(prompt)
                if gen is not None:
                    return gen

            # 2) read self-knowledge for this prompt
            risk, domains = 0.0, []
            if self.self_model is not None:
                try:
                    risk, domains = self.self_model.hallucination_risk(prompt)
                except Exception:  # noqa: BLE001
                    risk, domains = 0.0, []
            competence = self._competence_for(prompt)

            # 3) action prompts must clear the verifier before they may act
            if self._is_action(candidate):
                return RoutingPlan(Route.VERIFY_THEN_ACT, competence,
                                   "an action — it must clear the pre-action verifier",
                                   competence=competence, hallucination_risk=risk,
                                   domains=list(domains), is_action=True)

            teacher = self._teacher_available()

            # 4) weak / unknown / hallucination-prone here — this is the *new-domain* case.
            if (risk >= self.cfg.hallucination_ceiling
                    or competence < comp_threshold
                    or self._knowledge_heavy(prompt)):
                # 4a) FIRST try to generalize it herself by relational structure transfer — map
                #     the new domain onto one she already understands (Route.TRANSFER). The
                #     reasoning content is hers, not the base model's.
                transfer = self._try_transfer(prompt)
                transfer_analogical = bool(getattr(transfer, "analogical", False))
                # A *strict* transfer (relation names identical to a known base) is a strong,
                # faithful mapping — take it immediately. A *loose analogical* transfer forces an
                # alien field onto a known base by structure alone; that is weaker than modelling
                # the field from its OWN structure, so defer it and let the unified cascade (which
                # includes her from-scratch domain-genesis faculty) try first — mirroring
                # GeneralizationEngine.generalize. The analogical transfer is used only as a
                # fallback if the cascade also declines.
                if transfer is not None and not transfer_analogical:
                    conf = min(0.9, 0.5 + 0.1 * float(transfer.structural_score))
                    return RoutingPlan(
                        Route.TRANSFER, conf,
                        f"new domain — generalized by structural transfer from "
                        f"'{transfer.base_domain}'",
                        competence=competence, hallucination_risk=risk,
                        domains=list(domains), transfer=transfer)
                # 4a') try the rest of the unified own-faculty cascade — her from-scratch
                #     domain-genesis model, a from-examples task, or a numeric law stated as a
                #     table — before falling back to a loose analogy or the teacher.
                gen = self._try_generalize(prompt)
                if gen is not None:
                    gen.competence = competence
                    gen.hallucination_risk = risk
                    gen.domains = list(domains)
                    return gen
                # 4a'') the cascade declined — use the loose analogical transfer if we have one
                #     (still her own reasoning, just a weaker guarantee than a from-scratch model).
                if transfer is not None:
                    conf = min(0.6, 0.5 + 0.1 * float(transfer.structural_score))
                    return RoutingPlan(
                        Route.TRANSFER, conf,
                        f"new domain — generalized by loose structural analogy from "
                        f"'{transfer.base_domain}'",
                        competence=competence, hallucination_risk=risk,
                        domains=list(domains), transfer=transfer)
                # 4b) otherwise consult the teacher (or best-effort / abstain below)
                if teacher:
                    return RoutingPlan(Route.TEACHER, competence,
                                       "weak / unknown / hallucination-prone — consult the teacher",
                                       competence=competence, hallucination_risk=risk,
                                       domains=list(domains))
                # no teacher: best-effort own answer or an honest abstention
                verdict = self.meta.assess(prompt, own_answer="x", own_conf=competence,
                                           teacher_available=False,
                                           internal_uncertainty=internal_uncertainty)
                if verdict.decision is MetaDecision.ABSTAIN:
                    return RoutingPlan(Route.ABSTAIN, competence,
                                       "nothing trustworthy to say and no one to consult",
                                       competence=competence, hallucination_risk=risk,
                                       domains=list(domains))
                return RoutingPlan(Route.SELF, competence,
                                   "no teacher available — best-effort own answer",
                                   competence=competence, hallucination_risk=risk,
                                   domains=list(domains))

            # 5) competent and low-risk here -> her own model handles it
            return RoutingPlan(Route.SELF, competence,
                               "competent in this domain and low hallucination risk",
                               competence=competence, hallucination_risk=risk,
                               domains=list(domains))
        except Exception:  # noqa: BLE001 — triage is advisory; default to the teacher
            return RoutingPlan(Route.TEACHER, 0.0, "triage failed — defer to the teacher")

    def _faculty_fits(self, prompt: str) -> bool:
        try:
            from nyxara.mind.reasoning_faculties import solve_with_faculties
            if solve_with_faculties(prompt) is not None:
                return True
        except Exception:  # noqa: BLE001 — faculties are advisory; never crash a turn
            pass
        # broadened class: an arbitrary equation/system, identity, calculus or factorisation she
        # can compute-and-*certify* also "fits" — Route.FACULTY means "compute it exactly", and that
        # is exactly what the compute reducer does. Symbolic-only here (no model) so it stays cheap.
        try:
            from nyxara.mind.compute_reasoner import ComputeReasoner
            res = ComputeReasoner().solve(prompt)
            return res is not None and bool(res.verified)
        except Exception:  # noqa: BLE001 — additive; never crash a turn
            return False

    def _ensure_transfer(self) -> Any:
        """Lazily build the relational-transfer engine (her own cross-domain generalizer)."""
        if self._transfer is None:
            from nyxara.mind.transfer import RelationalTransferEngine
            self._transfer = RelationalTransferEngine(
                min_score=float(getattr(self.cfg, "transfer_min_score", 1.0)))
        return self._transfer

    def _try_transfer(self, prompt: str) -> Any:
        """Attempt to generalize a new-domain prompt by structural transfer. Returns a
        ``TransferResult`` or ``None`` (declines honestly when no structure maps)."""
        if not getattr(self.cfg, "use_transfer", True):
            return None
        try:
            return self._ensure_transfer().generalize(prompt)
        except Exception:  # noqa: BLE001 — transfer is advisory; never crash a turn
            return None

    # ---- the unified own-faculty generalization cascade ---- #
    def _generalization_enabled(self) -> bool:
        gcfg = getattr(self.settings, "generalization", None)
        return bool(getattr(self.cfg, "use_generalization", True)
                    and (gcfg is None or getattr(gcfg, "enabled", True)))

    def _has_demos(self, prompt: str) -> bool:
        """True when the prompt carries demonstrations her from-examples inducer could learn."""
        try:
            from nyxara.mind.generalization import has_demos
            return has_demos(prompt)
        except Exception:  # noqa: BLE001
            return False

    def _ensure_generalization(self) -> Any:
        """Lazily build the unified generalization engine (composes her own generalizers)."""
        if self._generalization is None:
            try:
                from nyxara.mind.generalization import GeneralizationEngine
                gcfg = getattr(self.settings, "generalization", None)
                self._generalization = GeneralizationEngine(
                    transfer_engine=self._transfer,
                    min_confidence=float(getattr(gcfg, "min_confidence", 0.4)),
                    parse_demos_enabled=bool(getattr(gcfg, "parse_demos", True)),
                    parse_tables_enabled=bool(getattr(gcfg, "parse_tables", True)),
                    min_demos=int(getattr(gcfg, "min_demos", 2)),
                    use_transfer=bool(getattr(self.cfg, "use_transfer", True)))
            except Exception:  # noqa: BLE001 — generalization is a capability, never required
                self._generalization = None
        return self._generalization

    def _try_generalize(self, prompt: str) -> Optional[RoutingPlan]:
        """Run the unified generalization cascade; return a ``Route.GENERALIZE`` plan or ``None``.

        Declines honestly (returns ``None``) when none of her own generalizers can answer, so the
        normal SELF / TEACHER / abstain path runs unchanged."""
        if not self._generalization_enabled():
            return None
        eng = self._ensure_generalization()
        if eng is None:
            return None
        try:
            res = eng.generalize(prompt)
        except Exception:  # noqa: BLE001 — the cascade is advisory; never crash a turn
            return None
        if res is None:
            return None
        return RoutingPlan(
            Route.GENERALIZE, float(res.confidence),
            f"novel task — solved by her own {res.source} faculty",
            generalization=res)

    # ---- dispatch a conversational reply ---- #
    def route_respond(self, prompt: str, *, system: Optional[str] = None,
                      difficulty: Optional[float] = None) -> RouterResult:
        """Triage, then draft a reply via the reused reactive router (or abstain honestly)."""
        plan = self.plan(prompt, difficulty=difficulty)
        if plan.route is Route.ABSTAIN:
            return RouterResult(HONEST_ABSTENTION, "abstain", plan.confidence, handed_off=False)
        # GENERALIZE: answer from NYXARA's OWN unified generalizers (skill-induction from the
        # examples in the prompt, relational transfer, open-world law-fitting) — the reasoning
        # content is hers, not sampled from any language model. Reported as a "faculty" handoff.
        if plan.route is Route.GENERALIZE and plan.generalization is not None:
            try:
                return RouterResult(plan.generalization.render(), "faculty", plan.confidence,
                                    handed_off=True)
            except Exception:  # noqa: BLE001 — fall through to the reactive router on any error
                pass
        # TRANSFER: answer from NYXARA's OWN structure-mapper — the reasoning content is hers,
        # not sampled from any language model. Reported as a "faculty" handoff (like exact math).
        if plan.route is Route.TRANSFER and plan.transfer is not None:
            try:
                return RouterResult(plan.transfer.render(), "faculty", plan.confidence,
                                    handed_off=True)
            except Exception:  # noqa: BLE001 — fall through to the reactive router on any error
                pass
        # SELF / TEACHER / FACULTY all delegate to the battle-tested confidence router, which
        # generates own-first / teacher-fallback and re-verifies on the answer's own merits.
        return self._ensure_router().draft(prompt, system=system)

    # ---- the verify-before-act gate (clause 3) ---- #
    def verify_action(self, prompt: str, candidate: Any) -> Tuple[bool, RoutingPlan]:
        """Score an action proposal's own coherence; only a clearing proposal may act.

        This is a *pre-gate*: it narrows what reaches the kernel's corrigibility / honesty /
        permission / guardian / oversight gates, never bypasses them. A proposal whose plan
        is incoherent, or whose stated confidence is too low, is declined here so NYXARA does
        not act on a guess."""
        plan = self.plan(prompt, candidate=candidate)
        plan.is_action = True
        plan.route = Route.VERIFY_THEN_ACT
        try:
            rationale = str(getattr(candidate, "rationale", "") or "")
            text = str(getattr(candidate, "text", "") or "")
            subject = f"{rationale} {text}".strip()
            score = float(self.verifier(prompt, subject))
            plan.verifier_score = score
            floor = self.cfg.verify_before_act
            if getattr(candidate, "risk", RiskTier.LOW) >= RiskTier.HIGH:
                floor = max(floor, self.cfg.verify_before_act_high)
            conf = float(getattr(candidate, "confidence", 0.0))
            plan.cleared = bool(score >= floor and conf >= self.cfg.act_min_confidence)
            plan.confidence = score
            plan.reason = ("action cleared the pre-action verifier" if plan.cleared
                           else "action failed the pre-action verifier")
        except Exception:  # noqa: BLE001 — fail-open: if we cannot score it, do not block it
            plan.cleared = True
            plan.reason = "verifier unavailable — action allowed to proceed to the gates"
        return bool(plan.cleared), plan


# --------------------------------------------------------------------------- #
# Self-test / demo (offline, scripted minds)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA primary self-model router self-test (offline, scripted minds)")
    print("=" * 70)

    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.memory.self_model import SelfModel
    from nyxara.mind.router import Router

    s = NyxaraSettings.for_profile(Profile.TEST)
    s.self_model_router.enabled = True

    sm = SelfModel()
    sm.set_capability("network", level=0.9, confidence=0.85)        # strong
    sm.set_capability("cooking", level=0.05, confidence=0.9)        # weak
    sm.declare_hallucination_risk("specific dates", risk=0.85,
                                  keywords=("what year", "when did"))

    class _Own:
        def available(self):
            return True

        def complete(self, req):
            return type("_R", (), {"text": "A firewall blocks the port, Master."})()

    class _Teacher:
        def available_providers(self):
            return ["aicredits", "groq", "airouter", "self", "native"]

        def complete_with(self, name, req):
            return type("_R", (), {"text": f"[teacher:{name}] a careful answer"})()

    router = Router(_Teacher(), settings=s, self_provider=_Own())
    psr = PrimarySelfModelRouter(self_model=sm, router=router, settings=s)

    assert psr.plan("How do I harden my network firewall?").route is Route.SELF
    print("strong capability -> SELF             ✓")
    assert psr.plan("How do I improve my cooking?").route is Route.TEACHER
    print("weak capability   -> TEACHER          ✓")
    assert psr.plan("what year did that happen?").route is Route.TEACHER
    print("hallucination zone-> TEACHER          ✓")
    assert psr.plan("what is 12 * 12").route is Route.FACULTY
    print("exact arithmetic  -> FACULTY          ✓")

    from nyxara.agency.permissions import RiskTier as _RT
    from nyxara.kernel.orchestrator import Candidate

    good = Candidate(text="remove stale cache", kind="act", tool="shell",
                     rationale="delete the stale cache file to free disk space",
                     risk=_RT.LOW, confidence=0.8)
    bad = Candidate(text="x", kind="act", tool="shell", rationale="the the the",
                    risk=_RT.HIGH, confidence=0.1)
    assert psr.verify_action("clean up", good)[0] is True
    assert psr.verify_action("clean up", bad)[0] is False
    print("verify-before-act -> clears good/blocks bad ✓")

    none = PrimarySelfModelRouter(self_model=None, router=router, settings=s)
    assert none.plan("anything at all").route in (Route.SELF, Route.TEACHER)
    print("no self-model     -> fail-open, no crash ✓")

    print("\nALL SELF-TESTS PASSED ✓")
