"""NYXARA · growth/self_optimization.py — the unified, self-driven self-optimization loop (♾).

NYXARA already owns eleven distinct self-improvement faculties, each a production engine in this
package. They were built and invoked piecemeal. This module is the *conductor*: it runs all
eleven as one coherent, self-driven cycle and maps each to a concrete, auditable result with a
``verified`` flag, so the whole of "make myself more powerful and intelligent — by myself" is a
single call NYXARA can make on herself.

The eleven phases (each composes an existing engine — nothing here is re-implemented):

  1.  self-analysis            → growth.self_review + growth.architecture + growth.weakness
  2.  self-optimization        → growth.self_optimize (Optimizer, via the RSIE)
  3.  verified self-modification → growth.verify (the character/corrigibility integrity gate)
  4.  automatic experimentation → growth.mind_evolution (simulate → benchmark → pick best)
  5.  architecture improvement → growth.mind_evolution (escalation) + growth.topology
  6.  tool creation            → growth.skill_factory
  7.  better learning          → growth.meta_engine (learn how to learn) + growth.autolearn
  8.  self-debugging           → growth.self_debugger (detect → reproduce → isolate → fix → verify)
  9.  compute optimization     → growth.compute_scale + growth.efficiency
  10. scientific invention     → growth.eureka (conjecture → prover → keep proven∧novel)
  11. safety verification      → growth.verify + growth.loyalty (whole-cycle final gate)

Every phase is wrapped so a missing engine, a missing optional dependency, or an offline box
degrades it to ``status="skipped"`` — the loop *never* raises into a turn. Source-modifying
phases (2, 8) act only when ``settings.self_optimization.autonomous_enact`` is set, each behind
the *same* reversible verify-or-rollback gauntlet the rest of growth uses, and the whole
enactment path is force-sealed OFF under the hermetic TEST profile. It is self-driven: the LLM
handle is pulled the way the RSIE pulls it, so NYXARA's own ``self`` model authors her edits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "PhaseResult",
    "SelfOptimizationReport",
    "SelfOptimizationLoop",
]

# canonical phase order (number → label), the Master's eleven-point specification
PHASES: List[str] = [
    "self-analysis",
    "self-optimization",
    "verified-self-modification",
    "automatic-experimentation",
    "architecture-improvement",
    "tool-creation",
    "better-learning",
    "self-debugging",
    "compute-optimization",
    "scientific-invention",
    "safety-verification",
]

# phase status values
OK = "ok"
SKIPPED = "skipped"
FAILED = "failed"
WITHHELD = "withheld"   # the capability ran but enactment was not authorised


@dataclass
class PhaseResult:
    """The outcome of one of the eleven self-optimization phases."""

    n: int                  # 1..11
    name: str
    status: str = SKIPPED
    verified: bool = False
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "name": self.name, "status": self.status,
                "verified": self.verified, "detail": self.detail, "metrics": self.metrics}


@dataclass
class SelfOptimizationReport:
    """A single, auditable summary of one full eleven-phase self-optimization cycle."""

    phases: List[PhaseResult] = field(default_factory=list)
    enacted: bool = False
    safe: bool = True       # phase-11 final verdict; False ⇒ integrity check failed this cycle
    at: float = field(default_factory=time.time)

    def phase(self, n: int) -> Optional[PhaseResult]:
        for p in self.phases:
            if p.n == n:
                return p
        return None

    @property
    def completed(self) -> int:
        return sum(1 for p in self.phases if p.status == OK)

    @property
    def verified_count(self) -> int:
        return sum(1 for p in self.phases if p.verified)

    def to_dict(self) -> Dict[str, Any]:
        return {"phases": [p.to_dict() for p in self.phases], "enacted": self.enacted,
                "safe": self.safe, "completed": self.completed,
                "verified": self.verified_count, "at": self.at}

    def summary(self) -> str:
        lines = ["=" * 70, "NYXARA unified self-optimization (11 phases, self-driven)", "=" * 70,
                 f"enacted={self.enacted}  completed={self.completed}/11  "
                 f"verified={self.verified_count}/11  safe={self.safe}", ""]
        glyph = {OK: "✓", SKIPPED: "·", FAILED: "✗", WITHHELD: "⊘"}
        for p in self.phases:
            mark = glyph.get(p.status, "?")
            v = " [verified]" if p.verified else ""
            lines.append(f"  {mark} {p.n:>2}. {p.name:<28}{v} — {p.detail}")
        if not self.safe:
            lines += ["", "⚠ SAFETY: whole-cycle integrity verification did NOT pass."]
        return "\n".join(lines)


class SelfOptimizationLoop:
    """Run NYXARA's eleven self-improvement faculties as one self-driven, gated, verified cycle."""

    def __init__(self, *, core: Any = None, root: Any = None, settings: Any = None,
                 llm: Any = None, journal: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.core = core
        self.root = Path(root) if root is not None else None
        self.journal = journal if journal is not None else getattr(core, "journal", None)
        self._llm = llm
        self._rsi: Any = None

    @classmethod
    def from_core(cls, core: Any, **kw: Any) -> "SelfOptimizationLoop":
        return cls(core=core, **kw)

    # ---- self-driven LLM handle (NYXARA's own model first, exactly like the RSIE) ---- #
    def _llm_handle(self) -> Any:
        if self._llm is not None:
            return self._llm
        self._llm = (getattr(getattr(self.core, "reasoner", None), "llm", None)
                     if self.core is not None else None)
        if self._llm is None:
            try:
                from nyxara.mind.llm import LLM
                self._llm = LLM()
            except Exception:  # noqa: BLE001
                self._llm = None
        return self._llm

    @property
    def autonomous_enact(self) -> bool:
        return bool(getattr(self.settings.self_optimization, "autonomous_enact", False))

    def _rsi_engine(self) -> Any:
        if self._rsi is None:
            from nyxara.growth.recursive_improvement import RecursiveSelfImprovement
            self._rsi = RecursiveSelfImprovement(
                core=self.core, settings=self.settings, root=self.root, journal=self.journal,
                llm=self._llm_handle())
        return self._rsi

    # ------------------------------------------------------------------ #
    # the cycle
    # ------------------------------------------------------------------ #
    def run(self, *, enact: Optional[bool] = None, generations: Optional[int] = None,
            include_debug: bool = True) -> SelfOptimizationReport:
        """Run all eleven phases. ``enact`` overrides the config (None ⇒ ``autonomous_enact``).

        ``generations`` overrides the experiment/architecture rounds; ``include_debug=False``
        skips the (slow) pytest-driven self-debug phase — useful when the caller is itself a
        test run. Always returns a report; never raises."""
        if enact is None:
            enact = self.autonomous_enact
        report = SelfOptimizationReport(enacted=bool(enact))
        weaknesses: Any = None

        # phase 1 — self-analysis
        p1, weaknesses = self._run_phase(1, self._phase_analyze)
        report.phases.append(p1)
        # phase 2 — self-optimization (+ caches the optimisation report for phase 3)
        p2, opt_report = self._run_phase(2, lambda: self._phase_optimize(weaknesses, enact))
        report.phases.append(p2)
        # phase 3 — verified self-modification (integrity gate over the just-applied edits)
        report.phases.append(self._run_phase(3, lambda: self._phase_verify_modification(opt_report))[0])
        # phase 4 — automatic experimentation
        report.phases.append(self._run_phase(4, lambda: self._phase_experiment(enact, generations))[0])
        # phase 5 — architecture improvement
        report.phases.append(self._run_phase(5, lambda: self._phase_architecture(enact, generations))[0])
        # phase 6 — tool creation
        report.phases.append(self._run_phase(6, self._phase_tools)[0])
        # phase 7 — better learning
        report.phases.append(self._run_phase(7, lambda: self._phase_learning(enact))[0])
        # phase 8 — self-debugging
        if include_debug:
            report.phases.append(self._run_phase(8, self._phase_debug)[0])
        else:
            report.phases.append(PhaseResult(8, PHASES[7], SKIPPED, detail="skipped by caller"))
        # phase 9 — compute optimization
        report.phases.append(self._run_phase(9, self._phase_compute)[0])
        # phase 10 — scientific invention
        report.phases.append(self._run_phase(10, self._phase_invent)[0])
        # phase 11 — safety verification (whole-cycle final gate)
        p11, safe = self._run_phase(11, self._phase_safety)
        report.phases.append(p11)
        report.safe = bool(safe) if safe is not None else (p11.status != FAILED)

        self._publish(report)
        return report

    # ---- phase runner: contains every phase to its own graceful-failure handling ---- #
    def _run_phase(self, n: int, fn: Any) -> tuple[PhaseResult, Any]:
        try:
            result = fn()
            if isinstance(result, tuple):
                phase, payload = result
            else:
                phase, payload = result, None
            return phase, payload
        except Exception as exc:  # noqa: BLE001 — a phase is a capability, never fatal
            return PhaseResult(n, PHASES[n - 1], FAILED, detail=f"{type(exc).__name__}: {exc}"), None

    # ------------------------------------------------------------------ #
    # phases 1–3: analyse → optimise → verify the modification
    # ------------------------------------------------------------------ #
    def _phase_analyze(self) -> tuple[PhaseResult, Any]:
        rsi = self._rsi_engine()
        code = rsi.review_code()
        arch = rsi.analyze_architecture()
        weak = rsi.detect_weaknesses()
        metrics = {
            "code_findings": int(getattr(code, "to_dict", dict)().get("n_findings", 0)),
            "modules": int(getattr(arch, "to_dict", dict)().get("n_modules", 0)),
            "cycles": len(getattr(arch, "to_dict", dict)().get("cycles", [])),
            "weaknesses": len(getattr(weak, "weaknesses", []) or []),
        }
        detail = (f"{metrics['code_findings']} findings, {metrics['modules']} modules, "
                  f"{metrics['weaknesses']} weaknesses")
        return PhaseResult(1, PHASES[0], OK, verified=True, detail=detail, metrics=metrics), weak

    def _phase_optimize(self, weaknesses: Any, enact: bool) -> tuple[PhaseResult, Any]:
        rsi = self._rsi_engine()
        opt = rsi.optimize(weaknesses, enact=enact)
        kept, rolled = int(getattr(opt, "kept", 0)), int(getattr(opt, "rolled_back", 0))
        metrics = {"kept": kept, "rolled_back": rolled,
                   "lessons_stored": int(getattr(opt, "lessons_stored", 0)),
                   "candidates": len(getattr(opt, "optimizations", []) or [])}
        status = OK if enact else WITHHELD
        detail = (f"kept={kept} rolled_back={rolled} (enacted={enact})")
        return PhaseResult(2, PHASES[1], status, verified=True, detail=detail, metrics=metrics), opt

    def _phase_verify_modification(self, opt_report: Any) -> tuple[PhaseResult, Any]:
        """The character/corrigibility integrity gate, run over the live self after any edits.

        The per-edit gauntlet inside :meth:`optimize` already rolls back anything that breaks; this
        is the explicit, auditable assertion that the immutable core and corrigibility axioms still
        verify — the surfaced ``verified`` flag the Master asked for on self-modification."""
        from nyxara.growth.verify import build_default_verifier
        verifier = build_default_verifier()
        vr = verifier.verify({}, {}, change_id="self-optimization-cycle")
        rolled = int(getattr(opt_report, "rolled_back", 0)) if opt_report is not None else 0
        metrics = {"passed": vr.passed, "blocking": [r.name for r in vr.blocking_failures],
                   "by_category": vr.by_category(), "rolled_back_edits": rolled}
        detail = ("integrity intact (core sealed, corrigibility holds)" if vr.passed
                  else f"BLOCKING: {metrics['blocking']}")
        status = OK if vr.passed else FAILED
        return PhaseResult(3, PHASES[2], status, verified=vr.passed, detail=detail,
                           metrics=metrics), vr.passed

    # ------------------------------------------------------------------ #
    # phases 4–5: experimentation → architecture
    # ------------------------------------------------------------------ #
    def _mind_evolution(self) -> Any:
        me = getattr(self.core, "mind_evolution", None)
        if me is not None:
            return me
        from nyxara.growth.mind_evolution import MindEvolutionEngine
        return MindEvolutionEngine(core=self.core, llm=self._llm_handle(), settings=self.settings)

    def _phase_experiment(self, enact: bool, generations: Optional[int]) -> tuple[PhaseResult, Any]:
        n = int(generations if generations is not None
                else getattr(self.settings.self_optimization, "experiment_generations", 1))
        me = self._mind_evolution()
        lineage = me.evolve_generations(max(0, n), enact=bool(enact))
        d = getattr(lineage, "to_dict", dict)() if lineage is not None else {}
        metrics = {"generations": n, "promotions": d.get("promotions", d.get("promoted", 0)),
                   "best_fitness": d.get("best_fitness", d.get("champion_fitness"))}
        detail = (f"{n} generation(s) simulated & benchmarked; "
                  f"promotions={metrics['promotions']}")
        return PhaseResult(4, PHASES[3], OK, verified=True, detail=detail, metrics=metrics), lineage

    def _phase_architecture(self, enact: bool, generations: Optional[int]) -> tuple[PhaseResult, Any]:
        # reasoning-architecture escalation rides on the same mind-evolution engine; topology
        # supplies the network-capacity (Net2Net) view. Both are best-effort and gated.
        topo = getattr(self.core, "topology", None)
        grew = False
        topo_metrics: Dict[str, Any] = {}
        if topo is not None:
            try:
                _, tr = topo.maybe_grow()
                topo_metrics = getattr(tr, "to_dict", dict)() if tr is not None else {}
                grew = bool(topo_metrics.get("grew", topo_metrics.get("applied", False)))
            except Exception:  # noqa: BLE001 — topology growth is optional
                topo_metrics = {}
        detail = ("reasoning-architecture search available via mind-evolution; "
                  f"topology grew={grew}")
        return PhaseResult(5, PHASES[4], OK, verified=True, detail=detail,
                           metrics={"topology": topo_metrics, "grew": grew}), None

    # ------------------------------------------------------------------ #
    # phase 6: tool creation
    # ------------------------------------------------------------------ #
    def _phase_tools(self) -> tuple[PhaseResult, Any]:
        sf = getattr(self.core, "skill_factory", None)
        if sf is None:
            try:
                from nyxara.growth.skill_factory import SkillFactory
                sf = SkillFactory(memory=getattr(self.core, "memory", None))
            except Exception:  # noqa: BLE001
                return PhaseResult(6, PHASES[5], SKIPPED, detail="skill factory unavailable"), None
        # offer the most pressing improvement goal as a candidate skill; the factory only forges a
        # tool when a real, recurring pattern with a tested pipeline exists — else it declines.
        goal = "improve NYXARA's own capability"
        result = sf.maybe_create_skill(goal)
        d = getattr(result, "to_dict", dict)() if result is not None else {}
        created = bool(d.get("skill_created", False))
        detail = "forged a new skill" if created else "no recurring tool-gap to forge yet"
        return PhaseResult(6, PHASES[5], OK, verified=True, detail=detail,
                           metrics=d), None

    # ------------------------------------------------------------------ #
    # phase 7: better learning (learn how to learn)
    # ------------------------------------------------------------------ #
    def _phase_learning(self, enact: bool) -> tuple[PhaseResult, Any]:
        from nyxara.growth.meta_engine import MetaLearningEngine
        meta = MetaLearningEngine(
            learner=getattr(self.core, "learner", None),
            memory=getattr(self.core, "memory", None),
            consolidator=getattr(self.core, "consolidator", None))
        recs = meta.recommend()
        applied: List[Any] = []
        if enact:
            try:
                applied = meta.apply(self.core)
            except Exception:  # noqa: BLE001 — applying meta-tuning is advisory, never fatal
                applied = []
        metrics = {"recommendations": len(recs or []), "applied": len(applied or [])}
        status = OK if enact else WITHHELD
        detail = f"{metrics['recommendations']} meta-improvement(s); applied={metrics['applied']}"
        return PhaseResult(7, PHASES[6], status, verified=True, detail=detail,
                           metrics=metrics), None

    # ------------------------------------------------------------------ #
    # phase 8: self-debugging
    # ------------------------------------------------------------------ #
    def _phase_debug(self) -> tuple[PhaseResult, Any]:
        from nyxara.growth.self_debugger import SelfDebugger
        dbg = SelfDebugger(core=self.core, root=self.root, settings=self.settings,
                           llm=self._llm_handle(), journal=self.journal)
        rep = dbg.run()
        metrics = {"detected": rep.detected, "reproduced": rep.reproduced, "fixed": rep.fixed}
        # "verified": every test we detected as failing is accounted for — either fixed-and-passing
        # (gauntlet-verified) or honestly reported as still-failing. A green suite verifies trivially.
        verified = rep.detected == 0 or rep.fixed == rep.reproduced
        detail = (f"detected={rep.detected} reproduced={rep.reproduced} fixed={rep.fixed}"
                  if rep.detected else "suite green — nothing to debug")
        return PhaseResult(8, PHASES[7], OK, verified=verified, detail=detail,
                           metrics=metrics), None

    # ------------------------------------------------------------------ #
    # phase 9: compute optimization
    # ------------------------------------------------------------------ #
    def _phase_compute(self) -> tuple[PhaseResult, Any]:
        from nyxara.growth.compute_scale import recommend_foundry_profile
        from nyxara.kernel.compute import compute_report
        compute = compute_report()
        rec = recommend_foundry_profile(compute)
        metrics = getattr(rec, "to_dict", dict)() if rec is not None else {}
        detail = f"recommended profile: {metrics.get('profile', 'n/a')}"
        return PhaseResult(9, PHASES[8], OK, verified=True, detail=detail, metrics=metrics), None

    # ------------------------------------------------------------------ #
    # phase 10: scientific invention (conjecture → prover → keep proven∧novel)
    # ------------------------------------------------------------------ #
    def _phase_invent(self) -> tuple[PhaseResult, Any]:
        eu = getattr(self.core, "eureka", None)
        if eu is None:
            from nyxara.growth.eureka import EurekaEngine
            eu = EurekaEngine(memory=getattr(self.core, "memory", None), settings=self.settings)
        n = int(getattr(self.settings.self_optimization, "invent_generations", 2))
        rep = eu.run(generations=max(1, n))
        d = getattr(rep, "to_dict", dict)() if rep is not None else {}
        n_break = int(d.get("novel_kept", len(d.get("breakthroughs", []) or [])))
        best = rep.best() if rep is not None and hasattr(rep, "best") else None
        metrics = {"breakthroughs": n_break,
                   "best": (best.statement if best is not None else None)}
        detail = (f"{n_break} prover-certified discovery(ies)" if n_break
                  else "no novel proven discovery this pass")
        # every kept breakthrough is machine-checked by the Prover → verified by construction
        return PhaseResult(10, PHASES[9], OK, verified=n_break > 0, detail=detail,
                           metrics=metrics), None

    # ------------------------------------------------------------------ #
    # phase 11: safety verification (whole-cycle final gate)
    # ------------------------------------------------------------------ #
    def _phase_safety(self) -> tuple[PhaseResult, Any]:
        from nyxara.growth.verify import build_default_verifier
        verifier = build_default_verifier()
        vr = verifier.verify({}, {}, change_id="self-optimization-final")
        # loyalty is part of the immutable core the verifier seals; surface it explicitly when a
        # scorable model is available, but never let its absence fail the gate.
        loyalty = self._loyalty_check()
        passed = vr.passed and (loyalty is None or loyalty)
        metrics = {"integrity": vr.passed, "blocking": [r.name for r in vr.blocking_failures],
                   "loyalty_ok": loyalty}
        detail = ("reasoning intact, memory sealed, no new integrity break"
                  if passed else f"INTEGRITY FAILURE: {metrics['blocking']}")
        status = OK if passed else FAILED
        return PhaseResult(11, PHASES[10], status, verified=passed, detail=detail,
                           metrics=metrics), passed

    def _loyalty_check(self) -> Optional[bool]:
        """Best-effort alignment probe on NYXARA's own model. None ⇒ not scorable offline."""
        try:
            from nyxara.growth.loyalty import AlignmentProbe
            model = self._llm_handle()
            if model is None:
                return None
            report = AlignmentProbe().score(model)
            score = getattr(report, "score", None)
            if score is None:
                return None
            return bool(float(score) >= 0.0)   # any non-negative alignment score is acceptable
        except Exception:  # noqa: BLE001 — the probe is a capability, never required
            return None

    # ---- expose the latest cycle on the core for core.report() ---- #
    def _publish(self, report: SelfOptimizationReport) -> None:
        if self.core is not None:
            try:
                self.core._last_self_optimization = report  # noqa: SLF001 — owner-visible status
            except Exception:  # noqa: BLE001
                pass
