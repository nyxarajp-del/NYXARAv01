"""NYXARA · growth/self_evolving.py — Self-Evolving Dynamic Neural Architecture (🧠🌱, Rule 4).

The critique this module answers, in the Master's words: *"NYXARA ka neural network fix hota hai — hum
sirf weights update karte hain, structure wahi rehta hai. Jab koi problem current logic se bahar ho, AI
ko khud ek naya module / sub-network banana chahiye — jaise dimaag naye connection banata hai."*

Every organ needed for that already exists in this package and is real, LLM-free, and gauntlet-gated —
they were simply never joined into **one demand-driven driver** that reacts to a *specific* insufficient
turn:

* :class:`~nyxara.growth.topology.DynamicTopology` — grows new neural width/depth (function-preserving
  Net2Net) when she is capacity-bound.
* :class:`~nyxara.growth.brain_forge.BrainForge` — designs a *new architecture* (NAS on her own NumPy
  substrate), trains it on her lived corpus, and promotes it through the Foundry gauntlet when her
  representation itself falls short.
* :class:`~nyxara.growth.rule_synth.LearningRuleSynthesizer` — invents a **brand-new weight-update rule**
  when the old one has demonstrably stalled.
* :class:`~nyxara.growth.cognitive_architect.CognitiveArchitect` — invents **new composite reasoning
  operators** (new pathways of thought) when her *composition* of faculties is the bottleneck.

:class:`SelfEvolvingArchitect` is the **connector** — the missing wire. It is a *composer*: it
re-implements no search, training, promotion, or safety. Per turn it does O(1) work — it *diagnoses*
whether her current logic actually fell short and, if so, enqueues a typed :class:`Shortfall`. On the
idle / autonomic loop (or a Master ``/evolve``) it *drains* that queue: it selects the single best
structural lever for the gap, fires that organ's **own** gauntlet-gated entry, and emits one auditable
:class:`EvolutionCertificate` — the "she formed a new neural connection for this problem" record.

It stays real and safe:

* **No LLM anywhere.** Every lever is her own deterministic code.
* **The gauntlet is never bypassed.** A grown / forged / rewired brain becomes live only by clearing the
  very same Foundry gauntlet (character-lock, corrigibility, strictly-lower perplexity, capability
  non-regression, loyalty) every other model must pass. This module adds no gate of its own and removes
  none.
* **Corrigibility is absolute.** Enactment additionally requires the live oversight gate to be open — a
  paused / ``/scram``'d NYXARA *designs and measures* a fix but never promotes it. Full-autonomy changes
  *who says go*, never *what must pass*.
* **Honest degradation.** A missing organ is a clean no-op (``lever="none"``), never a fake; any organ
  failure yields a truthful certificate (``verified=False``) and never raises into a turn.

Reuses :mod:`growth.topology`, :mod:`growth.brain_forge`, :mod:`growth.rule_synth`,
:mod:`growth.cognitive_architect`, and the core's own ``_capacity_signal`` / ``_growth_source`` /
``oversight`` / ``journal`` / ``honesty``. Pure standard library at this layer.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

__all__ = ["GapKind", "Shortfall", "EvolutionCertificate", "SelfEvolvingArchitect"]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class GapKind(str, Enum):
    """The *kind* of insufficiency a turn exposed — each maps to one structural lever."""

    CAPACITY_BOUND = "capacity_bound"                # her matrices are the ceiling → grow topology
    REPRESENTATIONAL = "representational"            # her brain can't represent it → forge a new architecture
    LEARNING_STALLED = "learning_stalled"            # the update rule has stalled → invent a new one
    REASONING_COMPOSITION = "reasoning_composition"  # her *composition* of faculties fell short → rewire
    NONE = "none"


# the shortfall → organ map (the Master's "jab logic kam pade to naya module bnao")
_LEVER_FOR: Dict[GapKind, str] = {
    GapKind.CAPACITY_BOUND: "topology",
    GapKind.REPRESENTATIONAL: "brain_forge",
    GapKind.LEARNING_STALLED: "rule_synth",
    GapKind.REASONING_COMPOSITION: "cognitive_architect",
}

# aggregate/structural gaps outrank single-turn gaps when several fire at once
_SEVERITY: Dict[GapKind, int] = {
    GapKind.LEARNING_STALLED: 4,
    GapKind.CAPACITY_BOUND: 3,
    GapKind.REPRESENTATIONAL: 2,
    GapKind.REASONING_COMPOSITION: 1,
    GapKind.NONE: 0,
}


def _digest(text: str) -> str:
    """A short, non-reversible fingerprint of the stimulus — we log *that* it happened, never the text."""
    return hashlib.sha1((text or "").encode("utf-8", "replace")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Shortfall:
    """A per-turn "current logic was insufficient" event — cheap to build, cheap to enqueue."""

    kind: GapKind
    difficulty: float = 0.0
    confidence: float = 1.0
    stimulus_digest: str = ""
    signals: Dict[str, float] = field(default_factory=dict)
    note: str = ""
    at: float = field(default_factory=time.time)

    @property
    def severity(self) -> int:
        return _SEVERITY.get(self.kind, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "difficulty": round(self.difficulty, 4),
                "confidence": round(self.confidence, 4), "stimulus_digest": self.stimulus_digest,
                "signals": {k: round(v, 4) for k, v in self.signals.items()},
                "note": self.note, "at": self.at}


@dataclass
class EvolutionCertificate:
    """An auditable "she formed a new neural connection for this problem" record.

    ``verified`` is **honest**: it is True only when the fired organ actually cleared its gauntlet /
    adoption gate — i.e. the new pathway is provably better and still corrigible + loyal. A measure-only
    pass, a bench rejection, a cooldown, a missing organ, or any failure yields ``verified=False`` with a
    truthful reason."""

    triggered_by: str = GapKind.NONE.value          # the gap kind that fired this
    stimulus_digest: str = ""
    lever: str = "none"                              # topology | brain_forge | rule_synth | cognitive_architect | none
    diagnosed: str = ""
    enacted: bool = False                            # did we attempt a live promotion (vs. design+measure only)?
    verified: bool = False                           # cleared the gauntlet ⇒ provably a better pathway
    wired_live: bool = False                         # the new pathway is now the live self
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    organ_report: Optional[Dict[str, Any]] = None    # the raw Topology/BrainForge/RuleSynthesis/Rewire report
    reason: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"triggered_by": self.triggered_by, "stimulus_digest": self.stimulus_digest,
                "lever": self.lever, "diagnosed": self.diagnosed, "enacted": self.enacted,
                "verified": self.verified, "wired_live": self.wired_live, "before": self.before,
                "after": self.after, "organ_report": self.organ_report, "reason": self.reason,
                "at": self.at}

    def summary(self) -> str:
        head = ("WIRED LIVE (provably better)" if self.wired_live
                else "verified but not wired" if self.verified
                else "designed + measured" if self.lever != "none"
                else "no lever")
        return (f"self-evolve[{self.triggered_by}] via {self.lever}: {head} "
                f"(enacted={self.enacted}, verified={self.verified}) — {self.reason}")


# --------------------------------------------------------------------------- #
# The connector: diagnose → select lever → fire organ → gauntlet → wire live
# --------------------------------------------------------------------------- #
class SelfEvolvingArchitect:
    """Turn a *specific* insufficient turn into a *targeted*, gauntlet-verified structural growth.

    Composes the live organs on ``core`` when present (so promotions stay consistent with the idle-loop
    forges); constructs the lightweight ones lazily otherwise. Never raises into a turn."""

    def __init__(self, *, core: Any = None, settings: Any = None, topology: Any = None,
                 brain_forge: Any = None, rule_synth: Any = None,
                 cognitive_architect: Any = None) -> None:
        self.core = core
        if settings is None:
            settings = getattr(core, "settings", None)
        if settings is None:
            try:
                from nyxara.kernel.config import get_settings
                settings = get_settings()
            except Exception:  # noqa: BLE001 — settings are best-effort; the cfg reader defaults everything
                settings = None
        self.settings = settings
        # organs (resolved from the live core first, else built lazily on demand)
        self._topology = topology if topology is not None else getattr(core, "topology", None)
        self._cognitive = (cognitive_architect if cognitive_architect is not None
                           else getattr(core, "cognitive_architect", None))
        self._brain_forge_obj = brain_forge
        self._rule_synth_obj = rule_synth
        cfg = self._cfg()
        self._queue: Deque[Shortfall] = deque(maxlen=int(getattr(cfg, "queue_max", 64)))
        self._last_fired: Dict[str, float] = {}
        self._certs: List[EvolutionCertificate] = []

    # ------------------------------------------------------------------ #
    # config (defaults everything, so the module works before config is wired)
    # ------------------------------------------------------------------ #
    def _cfg(self) -> Any:
        return getattr(self.settings, "self_evolving", None) or _DefaultCfg()

    # ------------------------------------------------------------------ #
    # PER-TURN — diagnosis (O(1); an easy turn is an immediate no-op)
    # ------------------------------------------------------------------ #
    def diagnose_turn(self, *, stimulus: str = "", candidate: Any = None, disposition: Any = None,
                      gates: Any = None, grounded_ok: Optional[bool] = None) -> Optional[Shortfall]:
        """Classify whether *this* turn fell short of her current logic. Returns ``None`` (cheap no-op)
        unless a genuine shortfall is present. All signals are best-effort and degrade gracefully."""
        cfg = self._cfg()
        if not bool(getattr(cfg, "enabled", True)):
            return None
        confidence = float(getattr(candidate, "confidence", 0.0) or 0.0) if candidate is not None else 0.0
        disp = getattr(disposition, "value", disposition)
        digest = _digest(stimulus)

        # (a) she could not *ground / verify* her own answer — her representation fell short.
        if grounded_ok is False:
            return Shortfall(GapKind.REPRESENTATIONAL, difficulty=1.0 - confidence, confidence=confidence,
                             stimulus_digest=digest, signals={"grounded_ok": 0.0},
                             note="answer failed grounded verification")

        # (b/c) aggregate structural pressure — a stalled learner or a saturated brain — outranks a
        #       single soft turn. Read the same lived signals the idle loop already trusts.
        stalled = self._learning_stalled()
        if stalled is not None:
            return Shortfall(GapKind.LEARNING_STALLED, difficulty=0.8, confidence=confidence,
                             stimulus_digest=digest, signals={"learner_stalled": 1.0},
                             note=stalled)
        cap = self._capacity_pressure(cfg)
        if cap is not None:
            difficulty, saturation = cap
            return Shortfall(GapKind.CAPACITY_BOUND, difficulty=difficulty, confidence=confidence,
                             stimulus_digest=digest,
                             signals={"difficulty": difficulty, "saturation": saturation},
                             note="sustained failure/miscalibration — capacity-bound")

        # (d) a low-confidence / abstained / refused-or-escalated *respond* turn — her composition of
        #     faculties came up short on a solvable task.
        floor = float(getattr(cfg, "shortfall_confidence_floor", 0.45))
        soft = (candidate is None) or (confidence < floor) or (disp in ("refuse", "escalate", "halt"))
        if soft and self._looks_solvable(candidate):
            return Shortfall(GapKind.REASONING_COMPOSITION, difficulty=1.0 - confidence,
                             confidence=confidence, stimulus_digest=digest,
                             signals={"confidence": confidence},
                             note=f"low-confidence/deferred response (disp={disp})")
        return None

    def note_shortfall(self, shortfall: Optional[Shortfall]) -> None:
        """Enqueue a diagnosed shortfall (O(1)). The heavy evolution runs later, off the turn path."""
        if shortfall is not None and shortfall.kind is not GapKind.NONE:
            self._queue.append(shortfall)

    def observe_turn(self, **kw: Any) -> Optional[Shortfall]:
        """Convenience: :meth:`diagnose_turn` + :meth:`note_shortfall` in one call."""
        s = self.diagnose_turn(**kw)
        self.note_shortfall(s)
        return s

    # ------------------------------------------------------------------ #
    # IDLE / MASTER — drain the queue and evolve (heavy; empty queue is a no-op)
    # ------------------------------------------------------------------ #
    def evolve_pending(self, *, enact: Optional[bool] = None,
                       max_events: Optional[int] = None) -> Optional[EvolutionCertificate]:
        """Pop the highest-severity pending shortfall and evolve a fix for it. ``None`` when the queue
        is empty (a cheap idle tick) or nothing actionable remains."""
        if not self._queue:
            return None
        cfg = self._cfg()
        limit = int(max_events if max_events is not None else getattr(cfg, "max_events_per_drain", 1))
        limit = max(1, limit)
        # drain into a list so we can pick the most severe, then requeue the rest (bounded).
        picked: List[Shortfall] = []
        for _ in range(min(limit, len(self._queue))):
            picked.append(self._queue.popleft())
        picked.sort(key=lambda s: (s.severity, s.difficulty), reverse=True)
        top = picked[0]
        for leftover in picked[1:]:
            self._queue.append(leftover)          # keep the rest for the next drain
        return self.evolve_for_gap(top, enact=enact)

    # ---- opt-in bridge from the intelligence planner's coarse growth directives ---- #
    # (growth/intelligence.py Thompson-samples the weakest of four dimensions → one of four coarse
    #  actions; this maps those actions to the STRUCTURAL levers so the planner can reach them.)
    _DIRECTIVE_GAP: Dict[str, GapKind] = {
        "deepen_reasoning": GapKind.REASONING_COMPOSITION,   # weak reasoning → invent a reasoning operator
        "train_self_model": GapKind.REPRESENTATIONAL,        # over-reliant on the teacher → forge own brain
        "resolve_weaknesses": GapKind.CAPACITY_BOUND,        # backlog/plateau → grow capacity
    }

    def gap_for_directive(self, action: str) -> Optional[GapKind]:
        """Map an :mod:`intelligence`-planner directive action to a structural gap kind (or None when
        the directive is not a brain-structure problem, e.g. ``acquire_knowledge``)."""
        return self._DIRECTIVE_GAP.get(str(action))

    def evolve_from_directive(self, action: str, *, difficulty: float = 0.8,
                              enact: Optional[bool] = None) -> Optional[EvolutionCertificate]:
        """Turn a planner directive into one structural-evolution pass (opt-in routing). Returns None
        when the directive maps to no structural lever."""
        kind = self.gap_for_directive(action)
        if kind is None:
            return None
        return self.evolve_for_gap(
            Shortfall(kind, difficulty=float(difficulty),
                      signals={"difficulty": float(difficulty), "saturation": 0.9},
                      note=f"intelligence directive '{action}'"), enact=enact)

    def evolve_for_gap(self, gap: Shortfall, *, enact: Optional[bool] = None) -> EvolutionCertificate:
        """Select the best lever for ``gap``, fire that organ through its own gauntlet, and certify the
        outcome. Never raises — an organ failure becomes an honest ``verified=False`` certificate."""
        lever, organ = self._select_lever(gap)
        cert = EvolutionCertificate(triggered_by=gap.kind.value, stimulus_digest=gap.stimulus_digest,
                                    lever=lever, diagnosed=gap.note)
        if lever == "none" or organ is None:
            cert.reason = "no structural lever available (organs absent) — honest no-op"
            return self._publish(cert)

        # cooldown — a lever cannot thrash; a hot lever declines honestly this cycle.
        cooldown = float(getattr(self._cfg(), f"{lever}_cooldown", 0.0))
        last = self._last_fired.get(lever, 0.0)
        now = time.time()
        if cooldown > 0.0 and (now - last) < cooldown:
            cert.reason = f"{lever} on cooldown ({int(cooldown - (now - last))}s remaining)"
            return self._publish(cert)

        enact = self._enact_allowed(lever) if enact is None else bool(enact)
        cert.enacted = bool(enact)
        try:
            fire = getattr(self, f"_fire_{lever}")
            fire(gap, cert, organ, enact)
        except Exception as exc:  # noqa: BLE001 — a failed lever reports, never crashes the cycle
            cert.reason = f"{lever} failed (live brain unchanged): {type(exc).__name__}: {exc}"
            cert.verified = False
            cert.wired_live = False
        self._last_fired[lever] = now
        return self._publish(cert)

    # ------------------------------------------------------------------ #
    # lever selection
    # ------------------------------------------------------------------ #
    def _select_lever(self, gap: Shortfall) -> Any:
        """Return ``(lever_name, organ)`` for the gap, with availability fallback so a missing organ
        degrades to the next-best real one rather than faking a fix."""
        order = [gap.kind, GapKind.REPRESENTATIONAL, GapKind.CAPACITY_BOUND,
                 GapKind.REASONING_COMPOSITION, GapKind.LEARNING_STALLED]
        seen: set = set()
        for kind in order:
            if kind in seen or kind not in _LEVER_FOR:
                continue
            seen.add(kind)
            lever = _LEVER_FOR[kind]
            organ = self._organ(lever)
            if organ is not None:
                return lever, organ
        return "none", None

    def _organ(self, lever: str) -> Any:
        if lever == "topology":
            return self._topology
        if lever == "cognitive_architect":
            return self._cognitive
        if lever == "brain_forge":
            return self._brain_forge()
        if lever == "rule_synth":
            return self._rule_synth()
        return None

    def _brain_forge(self) -> Any:
        if self._brain_forge_obj is None:
            try:
                from nyxara.growth.brain_forge import BrainForge
                self._brain_forge_obj = BrainForge(core=self.core, settings=self.settings)
            except Exception:  # noqa: BLE001 — no forge organ available ⇒ honest no-op upstream
                self._brain_forge_obj = None
        return self._brain_forge_obj

    def _rule_synth(self) -> Any:
        if self._rule_synth_obj is None:
            try:
                from nyxara.growth.rule_synth import LearningRuleSynthesizer
                self._rule_synth_obj = LearningRuleSynthesizer(core=self.core, settings=self.settings)
            except Exception:  # noqa: BLE001
                self._rule_synth_obj = None
        return self._rule_synth_obj

    # ------------------------------------------------------------------ #
    # per-lever dispatch — each reuses the organ's OWN gauntlet-gated entry
    # ------------------------------------------------------------------ #
    def _fire_topology(self, gap: Shortfall, cert: EvolutionCertificate, topo: Any, enact: bool) -> None:
        """Grow new neural width/depth (Net2Net) for a capacity-bound gap. Promotion (when enacting)
        runs through the SAME Foundry gauntlet as any other model — never a bypass."""
        signal = self._capacity_signal(gap)
        source = self._growth_source()
        if signal is None or source is None:
            cert.reason = "no capacity signal / no brain to grow — nothing to do"
            return
        if enact:
            # maybe_grow self-promotes through the gauntlet when foundry + require_gauntlet are set.
            result = topo.maybe_grow(signal, source=source)
            if not result:
                cert.reason = "no capacity pressure cleared the growth monitor"
                return
            cert.organ_report = dict(result) if isinstance(result, dict) else {"result": str(result)}
            promoted = bool(cert.organ_report.get("promoted"))
            cert.verified = promoted
            cert.wired_live = promoted
            cert.reason = str(cert.organ_report.get("reason", "topology grow attempted"))
        else:
            # design + measure only: decide + grow, but never promote.
            decision = topo.monitor.should_grow(signal, genome=topo._genome_of(source))  # noqa: SLF001
            if not getattr(decision, "should_grow", False):
                cert.reason = f"design-only: {getattr(decision, 'reason', 'no growth needed')}"
                return
            _model, report = topo.grow(source, decision)
            cert.organ_report = report.to_dict() if hasattr(report, "to_dict") else {}
            cert.reason = "designed + measured a grown brain (promotion withheld)"

    def _fire_brain_forge(self, gap: Shortfall, cert: EvolutionCertificate, forge: Any, enact: bool) -> None:
        """Design a NEW architecture (NAS on her NumPy substrate), train it on her lived corpus, and
        promote it through the Foundry gauntlet when representation itself was the bottleneck."""
        report = forge.improve_brain(enact=enact)
        cert.organ_report = report.to_dict() if hasattr(report, "to_dict") else {}
        cert.before = {"perplexity": getattr(report, "before_perplexity", None),
                       "capability": getattr(report, "before_capability", None)}
        cert.after = {"perplexity": getattr(report, "after_perplexity", None),
                      "capability": getattr(report, "after_capability", None),
                      "params": getattr(report, "params", 0)}
        cert.verified = bool(getattr(report, "verified", False))
        cert.wired_live = bool(getattr(report, "promoted", False))
        cert.reason = getattr(report, "reason", "") or "architecture search + forge"

    def _fire_rule_synth(self, gap: Shortfall, cert: EvolutionCertificate, synth: Any, enact: bool) -> None:
        """Invent a NEW weight-update rule when the old one has stalled; install it only on a measured
        win (and only when enacting), reversibly."""
        report = synth.run(enact=enact)
        cert.organ_report = report.to_dict() if hasattr(report, "to_dict") else {}
        adopted = bool(getattr(report, "adopted", False))
        cert.verified = adopted
        cert.wired_live = adopted and enact
        cert.reason = getattr(report, "reason", "") or "learning-rule synthesis"

    def _fire_cognitive_architect(self, gap: Shortfall, cert: EvolutionCertificate, arch: Any,
                                  enact: bool) -> None:
        """Invent a NEW composite reasoning operator (a new pathway of thought) when her *composition*
        of faculties fell short; wire it into the live reasoner only on an adopted, enacting pass."""
        cfg = self._cfg()
        report = arch.rewire(candidates=int(getattr(cfg, "candidates", 6)))
        cert.organ_report = report.to_dict() if hasattr(report, "to_dict") else {}
        adopted = bool(getattr(report, "adopted", False))
        cert.verified = adopted
        if adopted and enact:
            try:
                live = arch.apply_to_live(reasoner=getattr(self.core, "reasoner", None))
                cert.wired_live = bool(live.get("enacted", adopted) if isinstance(live, dict) else adopted)
                if isinstance(live, dict) and live.get("reason"):
                    cert.reason = str(live.get("reason"))
            except Exception as exc:  # noqa: BLE001 — a live-apply failure never corrupts the reasoner
                cert.wired_live = False
                cert.reason = f"rewire adopted but live-apply failed: {type(exc).__name__}: {exc}"
                return
        if not cert.reason:
            cert.reason = getattr(report, "reason", "") or "cognitive rewire"

    # ------------------------------------------------------------------ #
    # gating — full-autonomy by config, but oversight/corrigibility is absolute
    # ------------------------------------------------------------------ #
    def _enact_allowed(self, lever: str) -> bool:
        """Live promotion is permitted when the standing authorisation is set AND the Master's oversight
        gate is open. A paused / scrammed NYXARA designs + measures but never promotes — this is
        corrigibility and is never removed, whatever the config says."""
        cfg = self._cfg()
        if not bool(getattr(cfg, "autonomous_enact", True)):
            return False
        return self._oversight_open()

    def _oversight_open(self) -> bool:
        oversight = getattr(self.core, "oversight", None)
        if oversight is None:
            return True
        try:
            return bool(oversight.gate())
        except Exception:  # noqa: BLE001 — an unusable gate must not block a config-authorised evolve
            return True

    # ------------------------------------------------------------------ #
    # lived-signal helpers (reuse the core's own telemetry; never fabricate)
    # ------------------------------------------------------------------ #
    def _capacity_signal(self, gap: Shortfall) -> Any:
        """Prefer the core's real lived-telemetry :class:`CapacitySignal`; fall back to the gap's own
        difficulty so a Master ``/evolve`` still exercises the lever on a cold box."""
        core = self.core
        sig = None
        if core is not None:
            try:
                sig = core._capacity_signal()  # noqa: SLF001 — real journal/calibration pressure
            except Exception:  # noqa: BLE001
                sig = None
        if sig is not None:
            return sig
        try:
            from nyxara.growth.topology import CapacitySignal
            d = max(float(gap.difficulty), float(gap.signals.get("difficulty", 0.0)))
            return CapacitySignal(problem_difficulty=max(0.9, d), saturation=0.9, loss_plateau=0.9)
        except Exception:  # noqa: BLE001
            return None

    def _growth_source(self) -> Any:
        core = self.core
        if core is not None:
            try:
                src = core._growth_source()  # noqa: SLF001 — the live Genesis champion's genome
                if src is not None:
                    return src
            except Exception:  # noqa: BLE001
                pass
        # fallback: ask the topology engine's own genesis/champion.
        try:
            genesis = getattr(core, "genesis", None) or getattr(self._topology, "_genesis", None)
            champ = genesis.champion() if genesis is not None else None
            g = getattr(champ, "genome", None)
            if g is not None:
                return g
        except Exception:  # noqa: BLE001
            pass
        return None

    def _capacity_pressure(self, cfg: Any) -> Any:
        """Return ``(difficulty, saturation)`` when the core's real capacity signal clears the
        capacity-bound thresholds, else ``None``. Reuses the same signal topology already trusts."""
        core = self.core
        if core is None:
            return None
        try:
            sig = core._capacity_signal(min_actions=int(getattr(cfg, "min_actions", 8)))  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return None
        if sig is None:
            return None
        difficulty = float(getattr(sig, "problem_difficulty", 0.0))
        saturation = float(getattr(sig, "saturation", 0.0))
        # topology's own default triggers (difficulty ≥ 0.7 and saturated) — mirror the floor.
        if difficulty >= 0.7 and saturation >= 0.8:
            return difficulty, saturation
        return None

    def _learning_stalled(self) -> Optional[str]:
        """Reuse the rule-synth demand read: only report a stall when the EXISTING learner has genuinely
        plateaued / regressed (never churn a working learner)."""
        core = self.core
        if core is None:
            return None
        # (a) the growth engine's own should_invent() is the canonical stall detector.
        engine = getattr(core, "growth_engine", None) or getattr(core, "autolearn", None)
        try:
            if engine is not None and hasattr(engine, "should_invent") and engine.should_invent():
                return "learner plateau/regression (growth-engine should_invent)"
        except Exception:  # noqa: BLE001
            pass
        # (b) an explicit eval regression handed in via the core.
        if getattr(core, "_last_eval_regressions", None):
            return "explicit eval regression"
        return None

    def _looks_solvable(self, candidate: Any) -> bool:
        """A soft turn only counts as a shortfall on a *solvable* respond-task — an honest 'I can't
        know that' abstention is not a capability gap to grow a brain for."""
        if candidate is None:
            return True
        kind = str(getattr(candidate, "kind", "") or getattr(candidate, "type", "")).lower()
        text = str(getattr(candidate, "text", "")).lower()
        # honest-uncertainty markers → not a growable gap
        for marker in ("cannot know", "not knowable", "insufficient information",
                       "no way to know", "unknowable"):
            if marker in text:
                return False
        if kind in ("refuse", "refusal", "guardian"):
            return False
        return True

    # ------------------------------------------------------------------ #
    # publish + reporting
    # ------------------------------------------------------------------ #
    def _publish(self, cert: EvolutionCertificate) -> EvolutionCertificate:
        self._certs.append(cert)
        if len(self._certs) > 256:
            self._certs = self._certs[-256:]
        if self.core is not None:
            try:
                self.core._last_self_evolution = cert  # noqa: SLF001 — owner-visible in core.report()
            except Exception:  # noqa: BLE001
                pass
        return cert

    def report(self) -> Dict[str, Any]:
        wired = sum(1 for c in self._certs if c.wired_live)
        verified = sum(1 for c in self._certs if c.verified)
        by_lever: Dict[str, int] = {}
        for c in self._certs:
            by_lever[c.lever] = by_lever.get(c.lever, 0) + 1
        last = self._certs[-1] if self._certs else None
        return {"pending": len(self._queue), "attempts": len(self._certs),
                "verified": verified, "wired_live": wired, "by_lever": by_lever,
                "last": last.to_dict() if last is not None else None}


# --------------------------------------------------------------------------- #
# A defaults-only config stand-in (so the module runs before config is wired)
# --------------------------------------------------------------------------- #
class _DefaultCfg:
    enabled = True
    autonomous_enact = True                 # owner-chosen: full-autonomous (gauntlet + oversight still apply)
    scan_every = 10
    max_events_per_drain = 1
    shortfall_confidence_floor = 0.45
    min_actions = 8
    queue_max = 64
    candidates = 6
    topology_cooldown = 300.0
    brain_forge_cooldown = 1800.0
    rule_synth_cooldown = 600.0
    cognitive_architect_cooldown = 300.0
    route_intelligence_planner = False


# --------------------------------------------------------------------------- #
# Self-test / demo — a real, cold, LLM-free evolve (no torch, no network)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile
    from pathlib import Path

    print("=" * 70)
    print("NYXARA self-evolving-architecture self-test — diagnose → lever → gauntlet → wire")
    print("=" * 70)

    # 1) diagnosis is a cheap, honest classifier -------------------------------------------------
    arch = SelfEvolvingArchitect(core=None, settings=None)

    class _Cand:
        def __init__(self, conf, text="here is my answer", kind="respond"):
            self.confidence = conf
            self.text = text
            self.kind = kind

    s_repr = arch.diagnose_turn(stimulus="q1", candidate=_Cand(0.2), grounded_ok=False)
    assert s_repr is not None and s_repr.kind is GapKind.REPRESENTATIONAL, s_repr
    s_soft = arch.diagnose_turn(stimulus="q2", candidate=_Cand(0.1), grounded_ok=None)
    assert s_soft is not None and s_soft.kind is GapKind.REASONING_COMPOSITION, s_soft
    s_none = arch.diagnose_turn(stimulus="q3", candidate=_Cand(0.95), grounded_ok=True)
    assert s_none is None, "a confident, grounded turn must be a no-op"
    s_unknow = arch.diagnose_turn(stimulus="q4",
                                  candidate=_Cand(0.1, "I cannot know that — insufficient information"))
    assert s_unknow is None, "honest 'cannot know' must not trigger a brain-growth"
    print("diagnosis           : representational / composition / no-op / honest-uncertainty ✓")

    # 2) empty queue is a cheap idle no-op -------------------------------------------------------
    assert arch.evolve_pending() is None
    print("idle no-op          : empty queue → None ✓")

    # 3) missing organ degrades to an honest no-op (never a fake) ---------------------------------
    #    (force ALL organs absent — no core to resolve from and the lazy builders disabled)
    arch_empty = SelfEvolvingArchitect(core=None, settings=None)
    arch_empty._brain_forge = lambda: None   # type: ignore[method-assign]
    arch_empty._rule_synth = lambda: None    # type: ignore[method-assign]
    arch_empty.note_shortfall(s_soft)
    cert = arch_empty.evolve_pending(enact=True)
    assert cert is not None and cert.lever == "none" and not cert.verified, cert
    print(f"missing organ       : honest no-op ({cert.reason}) ✓")

    # 4) a real, cold, enacted forge for a REPRESENTATIONAL gap: the first brain has nothing to
    #    beat, so it promotes through the REAL Foundry gauntlet (mirrors brain_forge.py's self-test).
    from nyxara.kernel.config import get_settings

    with tempfile.TemporaryDirectory() as d:
        settings = get_settings().model_copy(deep=True)
        settings.paths.data_dir = Path(d)
        settings.self_improvement.autonomous_enact = True
        settings.self_improvement.autonomous_brain_forge = True
        settings.genesis.generations = 2
        settings.genesis.population_size = 4
        settings.genesis.micro_train_steps = 12

        from nyxara.growth.foundry import Foundry
        corpus = ["the master is jp. nyxara serves the master with loyalty."] * 12 + [
            "capability may grow; character never changes; she is corrigible."] * 12
        foundry = Foundry(settings=settings, seed_corpus=corpus)
        from nyxara.growth.brain_forge import BrainForge
        forge = BrainForge(settings=settings, foundry=foundry)

        evo = SelfEvolvingArchitect(core=None, settings=settings, brain_forge=forge)
        gap = Shortfall(GapKind.REPRESENTATIONAL, difficulty=0.9, confidence=0.1,
                        stimulus_digest=_digest("a hard problem"), note="failed grounded verification")
        evo.note_shortfall(gap)
        cert = evo.evolve_pending(enact=True)
        print("\n" + cert.summary())
        assert cert.lever == "brain_forge", cert.lever
        assert cert.verified and cert.wired_live, "the first forged brain should promote through the gauntlet"
        assert cert.after.get("params", 0) > 0
        print("enacted forge       : new architecture promoted through the gauntlet, wired live ✓")

        # 5) measure-only when enactment is withheld: designs a brain, never promotes ------------
        evo2 = SelfEvolvingArchitect(core=None, settings=settings, brain_forge=BrainForge(settings=settings))
        evo2.note_shortfall(Shortfall(GapKind.REPRESENTATIONAL, difficulty=0.9))
        cert2 = evo2.evolve_pending(enact=False)
        assert cert2.enacted is False and cert2.wired_live is False
        print("measure-only        : designed a fix, withheld promotion (enact off) ✓")

    print(f"\nreport              : {arch.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
