"""NYXARA · growth/recursive_improvement.py — Recursive Self-Improvement (♻, the RSI loop).

This is the orchestrator that unifies NYXARA's five self-improvement faculties into one
codebase-level cycle she can run on herself:

    review code → analyse architecture → run benchmarks → detect weaknesses → optimise

1. **Self code review** (:mod:`nyxara.growth.self_review`) — real ``ast`` static analysis.
2. **Self architecture analysis** (:mod:`nyxara.growth.architecture`) — her real import graph.
3. **Self benchmark testing** — reuses the real :mod:`nyxara.eval.benchmark` battery and the
   confidence router (capability + handoff), nothing re-implemented.
4. **Self weakness detection** (:mod:`nyxara.growth.weakness`) — folds the three into a ranked,
   actionable list.
5. **Self optimisation** (:mod:`nyxara.growth.self_optimize`) — *auto-applies* source fixes
   under a verify-or-rollback gauntlet (the Master's authorised, reversible self-modification),
   and enacts the safe non-source set: store durable lessons, tune reasoning depth within
   config bounds, and (optionally) drive the gauntlet-gated foundry.

Everything degrades gracefully offline and is honest about what it did. The whole cycle is
driven in the background by :class:`~nyxara.growth.autolearn.GrowthEngine` on a throttled
cadence — it is not a per-turn faculty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["SelfImprovementReport", "RecursiveSelfImprovement"]


@dataclass
class SelfImprovementReport:
    """A single, auditable summary of one recursive self-improvement cycle."""

    code: Optional[Dict[str, Any]] = None
    architecture: Optional[Dict[str, Any]] = None
    benchmark: Optional[Dict[str, Any]] = None
    # EXTERNAL ground-truth validation: held-out fold + adversarial battery the optimiser never
    # edits against, with a combined ``transfer_score``. None when validation is disabled/unrun.
    validation: Optional[Dict[str, Any]] = None
    weaknesses: Optional[Dict[str, Any]] = None
    optimizations: List[Dict[str, Any]] = field(default_factory=list)
    lessons_stored: int = 0
    tuned: Optional[Dict[str, Any]] = None
    kept: int = 0
    rolled_back: int = 0
    enacted: bool = False
    # --- intelligence index: I_(t+1) = f(I_t, C_available) --- #
    intelligence_index: Optional[float] = None
    intelligence_t: Optional[int] = None
    # the momentum-free raw capability target for this cycle; the meta tower is scored by its
    # change (Δbase), NOT the smoothed index, so evolving the smoothing can never game the score.
    intelligence_base: Optional[float] = None
    # open-ended auto-curriculum frontier score (diagnostic; also folded into transfer_score)
    frontier: Optional[float] = None
    # world-grounded score: prediction graded by REAL execution (diagnostic; folded into transfer)
    grounded: Optional[float] = None
    compute: Optional[Dict[str, Any]] = None
    # substrate self-expansion: the parallelism/cluster NYXARA surveyed this cycle (growth/substrate)
    substrate: Optional[Dict[str, Any]] = None
    effort_budget: Optional[Dict[str, Any]] = None
    directive: Optional[Dict[str, Any]] = None   # index-driven: which growth action to take next
    # --- full-wire: the directive turned into a real cross-system action, + its outcome --- #
    directive_action: Optional[str] = None       # the action she planned + (when enacted) ran
    directive_results: Optional[Dict[str, Any]] = None   # what dispatching it actually did
    ledger: Optional[Dict[str, Any]] = None      # lifelong credit assignment for this cycle
    meta_meta: Optional[Dict[str, Any]] = None   # meta-meta: evolving HOW she improves (the knobs)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "architecture": self.architecture,
                "benchmark": self.benchmark, "validation": self.validation,
                "weaknesses": self.weaknesses,
                "optimizations": self.optimizations, "lessons_stored": self.lessons_stored,
                "tuned": self.tuned, "kept": self.kept, "rolled_back": self.rolled_back,
                "enacted": self.enacted, "intelligence_index": self.intelligence_index,
                "intelligence_t": self.intelligence_t,
                "intelligence_base": self.intelligence_base, "frontier": self.frontier,
                "grounded": self.grounded,
                "compute": self.compute, "substrate": self.substrate,
                "effort_budget": self.effort_budget, "directive": self.directive,
                "directive_action": self.directive_action,
                "directive_results": self.directive_results, "ledger": self.ledger,
                "meta_meta": self.meta_meta, "at": self.at}

    def summary(self) -> str:
        n_weak = (self.weaknesses or {}).get("n_weaknesses", 0)
        acc = (self.benchmark or {}).get("accuracy")
        lines = ["=" * 70, "NYXARA recursive self-improvement", "=" * 70,
                 f"code findings   : {(self.code or {}).get('n_findings', 0)}",
                 f"architecture    : {(self.architecture or {}).get('n_modules', 0)} modules, "
                 f"{len((self.architecture or {}).get('cycles', []))} cycle(s), "
                 f"{len((self.architecture or {}).get('layering_violations', []))} layering",
                 f"benchmark       : accuracy {acc:.0%}" if acc is not None
                 else "benchmark       : (not run)",
                 f"weaknesses      : {n_weak}",
                 f"optimisation    : enacted={self.enacted}, kept={self.kept}, "
                 f"rolled_back={self.rolled_back}, lessons={self.lessons_stored}"]
        if self.validation is not None and "transfer_score" in self.validation:
            v = self.validation
            lines.append(f"validation      : transfer {v['transfer_score']:.0%} "
                         f"(held-out {v.get('holdout_mean_score', 0.0):.0%}, "
                         f"adversarial {v.get('adversarial_mean_score', 0.0):.0%}) "
                         f"— external ground truth")
            fr = v.get("frontier") if isinstance(v, dict) else None
            if isinstance(fr, dict):
                lines.append(f"frontier        : {fr.get('frontier_score', 0.0):.0%} "
                             f"@ tier {fr.get('frontier_tier')} "
                             f"(open-ended curriculum, prover-certified)")
            gr = v.get("grounded") if isinstance(v, dict) else None
            if isinstance(gr, dict):
                lines.append(f"grounded        : {gr.get('grounded_score', 0.0):.0%} "
                             f"(real execution, not a stored answer key)")
        if self.intelligence_index is not None:
            lines.append(f"intelligence    : I_{self.intelligence_t} = "
                         f"{self.intelligence_index:.4f}")
            if self.directive and self.directive.get("goodhart"):
                lines.append("                  ⚠ GOODHART: proxy rose but held-out did not "
                             "— index gain discounted")
        if self.directive is not None:
            lines.append(f"directive       : {self.directive.get('action')} "
                         f"(focus={self.directive.get('focus')}, "
                         f"escalate={self.directive.get('escalate')})")
        if self.directive_results is not None:
            dispatched = self.directive_results.get("dispatched")
            lines.append(f"directive action: {self.directive_action} "
                         f"(dispatched={dispatched})")
        if self.ledger is not None:
            lines.append(f"ledger          : Δindex={self.ledger.get('delta')}, "
                         f"arms credited={len(self.ledger.get('arms', []))}")
        if self.substrate is not None:
            s = self.substrate
            lines.append(f"substrate       : {s.get('parallel_units')} unit(s)/"
                         f"{s.get('nodes')} node(s) ×{s.get('expansion_factor')} ceiling "
                         f"[{', '.join(s.get('providers', []))}]")
        if self.effort_budget is not None:
            lines.append(f"effort budget   : {self.effort_budget}")
        if self.tuned:
            lines.append(f"tuned           : {self.tuned}")
        return "\n".join(lines)


class RecursiveSelfImprovement:
    """The codebase-level self-improvement loop, wired to NYXARA's real faculties."""

    def __init__(self, *, core: Any = None, memory: Any = None, settings: Any = None,
                 llm: Any = None, root: Any = None, growth_engine: Any = None,
                 journal: Any = None, intelligence: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.core = core
        self.memory = memory if memory is not None else getattr(core, "memory", None)
        self.journal = journal if journal is not None else getattr(core, "journal", None)
        self.growth_engine = growth_engine
        self.root = root
        self._llm = llm
        self._intelligence = intelligence
        # lifelong credit assignment + (optional, heavy-ML) learned payoff forecaster
        self._ledger: Any = None
        self._bandit: Any = None
        self._forecaster: Any = None
        # per-cycle caches
        self._code = None
        self._arch = None
        self._bench = None
        self._validation: Optional[Dict[str, Any]] = None   # external held-out + adversarial scores
        self._compute = None
        self._effort: Optional[Dict[str, Any]] = None
        self._directive: Optional[Dict[str, Any]] = None
        self._ledger_state: Any = None
        self._touched_arms: List[str] = []     # arms credited at the end of this cycle
        self._prior_index: Optional[float] = None
        self._prior_base: Optional[float] = None    # momentum-free raw target before this cycle
        self._prior_transfer: Optional[float] = None  # external held-out score before this cycle
        self._last_signals: Optional[Dict[str, float]] = None
        self._meta: Any = None                 # meta-meta controller (lazy; gated)
        self._meta_built = False
        self._substrate: Any = None            # substrate manager (lazy; raises the effort ceiling)
        self._substrate_built = False

    # ---- meta-meta: evolve the improvement engine's OWN knobs, scored by realised index gain ---- #
    def _meta_meta(self) -> Any:
        """The meta-meta controller — built once, only when self-modification is authorised.

        Gated so it never silently rewrites knobs: it acts only when ``autonomous_enact`` and
        ``allow_tuning`` are set and ``meta_meta_enabled`` is on, and never under the TEST profile
        (the hermetic suite must not evolve live knobs). Returns ``None`` when disabled."""
        if self._meta_built:
            return self._meta
        self._meta_built = True
        cfg = self.settings.self_improvement
        try:
            if str(getattr(getattr(self.settings, "profile", None), "value", "")) == "test":
                return None
            if not (bool(cfg.autonomous_enact) and bool(cfg.allow_tuning)
                    and bool(getattr(cfg, "meta_meta_enabled", True))):
                return None
            from nyxara.growth.meta_meta import MetaGenome, RecursiveMetaController
            # seed the engine genome from ALL current config values — the algorithm + measurement
            # knobs, not just the two execution knobs — so the tower starts from the live policy.
            seed = MetaGenome(
                recursion_depth=int(getattr(cfg, "llm_edit_recursion_depth", 3)),
                max_edits_per_cycle=int(getattr(cfg, "max_edits_per_cycle", 3)),
                ledger_reward_scale=float(getattr(cfg, "ledger_reward_scale", 40.0)),
                ledger_prior_strength=float(getattr(cfg, "ledger_prior_strength", 1.0)),
                bandit_severity_blend=float(getattr(cfg, "bandit_severity_blend", 0.7)),
                intelligence_momentum=float(getattr(cfg, "intelligence_momentum", 0.7)))
            path = None
            try:
                from pathlib import Path
                base = Path(self.settings.paths.data_dir) / "foundry"
                path = base / "meta_meta.json"
            except Exception:  # noqa: BLE001
                path = None
            # accelerating returns: let the tower grow its height + execution caps under sustained
            # gains, tied to the substrate it actually holds (more units ⇒ a higher permitted
            # recursion/edit ceiling), so acceleration is real but bounded by the hard caps.
            # The Master's dial (#6): when configured, RAISE the absolute ceilings (bounded by the
            # immovable hard guard in meta_meta.py) so the loop may grow more aggressively. This never
            # removes the bound — it makes it Master-configurable and still hard-capped.
            try:
                from nyxara.growth.meta_meta import MetaGenome
                MetaGenome.set_absolute_ceilings(
                    recursion=getattr(cfg, "recursion_ceiling", None),
                    edits=getattr(cfg, "edits_ceiling", None))
                abs_r, abs_e = MetaGenome._ABS_MAX_RECURSION, MetaGenome._ABS_MAX_EDITS
            except Exception:  # noqa: BLE001 — fall back to the built-in 16 / 24 ceilings
                abs_r, abs_e = 16, 24
            r_ceiling = e_ceiling = None
            try:
                mgr = self._substrate_mgr()
                if mgr is not None:
                    units = int(mgr.survey().parallel_units)
                    # acceleration scales with surveyed substrate, up to the (raisable) absolute ceiling
                    r_ceiling = max(5, min(abs_r, 4 + units))
                    e_ceiling = max(6, min(abs_e, 4 + units * 2))
            except Exception:  # noqa: BLE001
                r_ceiling = e_ceiling = None
            # with no substrate signal but a Master-raised ceiling, still let the loop use it
            if r_ceiling is None and abs_r > 16:
                r_ceiling = abs_r
            if e_ceiling is None and abs_e > 24:
                e_ceiling = abs_e
            self._meta = RecursiveMetaController(
                height=int(getattr(cfg, "meta_levels", 3)), champion=seed, persist_path=path,
                can_grow=bool(getattr(cfg, "meta_tower_can_grow", True)),
                hard_max_height=int(getattr(cfg, "meta_levels_hard_max", 6)),
                recursion_ceiling=r_ceiling, edits_ceiling=e_ceiling)
        except Exception:  # noqa: BLE001 — the meta-meta loop is a capability, never required
            self._meta = None
        return self._meta

    # ---- substrate self-expansion: survey the parallelism/cluster, raise the ceiling ---- #
    def _substrate_mgr(self) -> Any:
        """The substrate manager — built once, surveys local cores + any cluster.

        Returns ``None`` when substrate expansion is disabled, so the budget falls back exactly to
        the prior single-box behaviour. Never raises — a survey failure simply yields no manager."""
        if self._substrate_built:
            return self._substrate
        self._substrate_built = True
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "substrate_expansion", True)):
            return None
        try:
            from nyxara.growth.substrate import SubstrateManager
            self._substrate = SubstrateManager(settings=self.settings)
        except Exception:  # noqa: BLE001 — substrate expansion is a capability, never required
            self._substrate = None
        return self._substrate

    # ---- intelligence index: I_(t+1) = f(I_t, C_available) ---- #
    def _intel(self) -> Any:
        if self._intelligence is None:
            from nyxara.growth.intelligence import IntelligenceIndex
            self._intelligence = IntelligenceIndex(memory=self.memory, settings=self.settings)
        return self._intelligence

    # ---- lifelong credit assignment (ledger) + learned payoff forecaster ---- #
    def _ledger_enabled(self) -> bool:
        s = getattr(self, "settings", None)
        if s is None:
            return False
        return bool(getattr(s.self_improvement, "enable_improvement_ledger", True))

    def _ledger_obj(self) -> Any:
        if self._ledger is None:
            from nyxara.growth.credit import EditStrategyBandit, ImprovementLedger
            self._ledger = ImprovementLedger(memory=self.memory, settings=self.settings)
            self._bandit = EditStrategyBandit(self._ledger, forecaster=self._forecaster_obj())
        return self._ledger

    def _bandit_obj(self) -> Any:
        if self._bandit is None:
            self._ledger_obj()
        return self._bandit

    def _forecaster_obj(self) -> Any:
        if self._forecaster is None:
            if not bool(getattr(self.settings.self_improvement, "use_payoff_forecaster", False)):
                return None
            try:
                from nyxara.growth.forecaster import PayoffForecaster
                self._forecaster = PayoffForecaster(settings=self.settings)
            except Exception:  # noqa: BLE001 — the forecaster is an optional accelerant
                self._forecaster = None
        return self._forecaster

    def _ledger_state_obj(self) -> Any:
        if self._ledger_state is None and self._ledger_enabled():
            try:
                self._ledger_state = self._ledger_obj().load()
            except Exception:  # noqa: BLE001
                self._ledger_state = None
        return self._ledger_state

    def _compute_report(self) -> Any:
        if self._compute is None:
            try:
                from nyxara.kernel.compute import compute_report
                self._compute = compute_report()
            except Exception:  # noqa: BLE001 — compute introspection is best-effort
                self._compute = None
        return self._compute

    @classmethod
    def from_core(cls, core: Any, **kw: Any) -> "RecursiveSelfImprovement":
        return cls(core=core, **kw)

    # ---- shared LLM handle (same source the orchestrator uses) ---- #
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

    # ---- (1) self code review ---- #
    def review_code(self) -> Any:
        from nyxara.growth.self_review import SelfReviewer
        cfg = self.settings.self_improvement
        self._code = SelfReviewer(
            root=self.root, llm=self._llm_handle(),
            max_function_length=cfg.max_function_length,
            max_complexity=cfg.max_complexity, max_args=cfg.max_args,
            enable_llm_enrichment=cfg.enable_llm_enrichment).review()
        return self._code

    # ---- (2) self architecture analysis ---- #
    def analyze_architecture(self) -> Any:
        from nyxara.growth.architecture import ArchitectureAnalyzer
        self._arch = ArchitectureAnalyzer(root=self.root, llm=self._llm_handle()).analyze()
        return self._arch

    # ---- (3) self benchmark testing (reuses eval/benchmark) ---- #
    def run_benchmarks(self, *, category: Optional[str] = None) -> Dict[str, Any]:
        from nyxara.eval.benchmark import (build_default_benchmark, core_solver,
                                           run_router)
        cfg = self.settings.self_improvement
        full = build_default_benchmark()
        # reserve a deterministic held-out fold for EXTERNAL validation: the self-improvement loop
        # detects weaknesses / selects edits ONLY against the train fold, so the held-out fold (and
        # the adversarial battery, run in run_validation) never leaks into edit selection. That
        # isolation is exactly what stops the index Goodharting the very tasks it measures.
        if bool(getattr(cfg, "validation_enabled", True)):
            bench, _holdout = full.split(float(getattr(cfg, "validation_holdout_frac", 0.3)))
        else:
            bench = full
        # the intelligence index drives measurement effort: when the machine cannot afford the
        # full battery (effort budget says benchmark_full is False) and no category was pinned,
        # she probes a single representative category instead — a real, index-governed compute
        # saving, not a cosmetic number.
        scope = category
        self._plan_effort()
        if (scope is None and self._effort is not None
                and not self._effort.get("benchmark_full", True)):
            cats = bench.categories()
            if cats:
                scope = cats[0]
        try:
            report = bench.run(core_solver(), category=scope)
        except Exception as exc:  # noqa: BLE001
            self._bench = {"error": f"benchmark failed: {exc}", "accuracy": 0.0,
                           "failures": [], "by_category": {}, "scope": scope or "full"}
            return self._bench
        handoff: Dict[str, int] = {}
        try:
            _, handoff = run_router(bench, settings=self.settings, category=scope)
        except Exception:  # noqa: BLE001 — router is optional signal
            handoff = {}
        self._bench = {
            "accuracy": report.accuracy, "mean_score": report.mean_score,
            "by_category": report.by_category(),
            "failures": [r.to_dict() for r in report.failures()],
            "handoff": handoff, "report": report.to_dict(), "scope": scope or "full"}
        return self._bench

    # ---- (3b) EXTERNAL validation — held-out fold + adversarial battery (anti-Goodhart) ---- #
    def run_validation(self) -> Optional[Dict[str, Any]]:
        """Measure REAL, transferable capability on tasks the optimiser never edits against.

        Three external rulers, run through the *same* sovereign solver as the proxy benchmark:

        * the **real-world held-out corpus** (``eval/datasets.build_realworld_benchmark``) — a
          curated, externally-true set of real facts and multi-step problems NYXARA never trains on
          (or any standard dataset the Master points ``validation_realworld_path`` at). This is the
          genuinely-external ruler and, by default, the **dominant** term in the score;
        * the **held-out fold** of the training battery (``Benchmark.split``) — a same-distribution
          generalisation check; and
        * the full **adversarial** battery (``eval/hard_benchmark.build_hard_benchmark``: multi-hop
          deduction, sequence induction, code-output prediction, grounded reading and calibration
          traps) — an out-of-distribution transfer check.

        None is ever passed to :meth:`detect_weaknesses` / :meth:`_weakest_categories`, so none can
        steer edit selection. Their weighted ``transfer_score`` is the one signal NYXARA cannot
        game: it rises only when capability genuinely transfers to data she did not author. Honest
        about cost — when the effort budget says the machine cannot afford the full battery it scopes
        the adversarial run to a representative category, and any failure degrades to a clean
        ``None``/error block. The real-world corpus degrades cleanly on its own too: if it is
        unavailable (or disabled) its weight is dropped and the remaining rulers are renormalised.
        """
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "validation_enabled", True)):
            self._validation = None
            return None
        try:
            from nyxara.eval.benchmark import build_default_benchmark, core_solver
            from nyxara.eval.hard_benchmark import build_hard_benchmark
        except Exception as exc:  # noqa: BLE001
            self._validation = {"error": f"validation import failed: {exc}",
                                "transfer_score": 0.0}
            return self._validation
        full = build_default_benchmark()
        _train, holdout = full.split(float(getattr(cfg, "validation_holdout_frac", 0.3)))
        adversarial = build_hard_benchmark()
        self._plan_effort()
        adv_scope: Optional[str] = None
        if self._effort is not None and not self._effort.get("benchmark_full", True):
            cats = adversarial.categories()
            adv_scope = cats[0] if cats else None
        # The real, externally-true held-out corpus — loaded only if enabled and available. Its
        # absence is not an error (the bundled file may be missing in a stripped install); we simply
        # drop its weight below and lean on the other two rulers.
        realworld = None
        rw_error: Optional[str] = None
        if bool(getattr(cfg, "validation_realworld_enabled", True)):
            try:
                from nyxara.eval.datasets import build_realworld_benchmark
                realworld = build_realworld_benchmark(self.settings)
                # anti-memorisation: when configured, score a deterministic ROTATING subset (seeded by
                # the number of validated cycles so far) so the optimiser can never overfit a fixed
                # list. sample_n == 0 keeps the whole corpus (prior behaviour) — rotation is opt-in.
                n = int(getattr(cfg, "validation_sample_n", 0) or 0)
                if realworld is not None and n > 0 and bool(getattr(cfg, "validation_rotate", True)):
                    try:
                        seed = len(self._intel().load().validation_history)
                    except Exception:  # noqa: BLE001 — no index yet ⇒ a stable default seed
                        seed = 0
                    realworld = realworld.sample(n, seed=seed)
            except Exception as exc:  # noqa: BLE001 — a missing corpus must never break the cycle
                rw_error = str(exc)
        try:
            solver = core_solver()
            ho = holdout.run(solver)
            adv = adversarial.run(solver, category=adv_scope)
            rw = realworld.run(solver) if realworld is not None else None
        except Exception as exc:  # noqa: BLE001 — validation is a measurement, never fatal
            self._validation = {"error": f"validation failed: {exc}", "transfer_score": 0.0}
            return self._validation
        # mean_score (graded, partial credit) is a smoother transfer signal than pass/fail accuracy.
        # Blend the rulers by configured weight, dropping (and renormalising over) any absent ruler so
        # the score is always a clean convex combination of what was actually measured.
        w_rw = float(getattr(cfg, "transfer_weight_realworld", 0.5))
        w_ho = float(getattr(cfg, "transfer_weight_holdout", 0.2))
        w_adv = float(getattr(cfg, "transfer_weight_adversarial", 0.3))
        terms = [(w_ho, float(ho.mean_score)), (w_adv, float(adv.mean_score))]
        if rw is not None:
            terms.append((w_rw, float(rw.mean_score)))
        # 4th ruler: the open-ended auto-curriculum — fresh, prover-certified problems at the edge of
        # her capability. It cannot saturate (the frontier tracks her) and cannot be memorised (every
        # batch is regenerated), so it is the one ruler that keeps moving once a fixed battery is
        # mastered. None when disabled or unavailable, in which case its weight is simply dropped.
        frontier_block = self._run_curriculum(solver)
        if frontier_block is not None:
            w_fr = float(getattr(cfg, "transfer_weight_frontier", 0.25))
            terms.append((w_fr, float(frontier_block["frontier_score"])))
        # 5th ruler: world-grounded experiments — NYXARA predicts a program's output and is graded by
        # REAL execution (no stored key). Like the others, dropped (not zeroed) when unavailable.
        grounded_block = self._run_grounded(solver)
        if grounded_block is not None:
            w_gr = float(getattr(cfg, "transfer_weight_grounded", 0.2))
            terms.append((w_gr, float(grounded_block["grounded_score"])))
            # 6th ruler: agentic tool-task authoring — NYXARA writes code graded by REAL execution.
            # Present only when actually measured; like the others its weight is dropped if absent.
            if "tool_grounded_score" in grounded_block:
                w_tg = float(getattr(cfg, "transfer_weight_tool_grounded", 0.2))
                terms.append((w_tg, float(grounded_block["tool_grounded_score"])))
        wsum = sum(w for w, _ in terms) or 1.0
        transfer = sum(w * s for w, s in terms) / wsum
        self._validation = {
            "holdout_accuracy": round(ho.accuracy, 6),
            "holdout_mean_score": round(ho.mean_score, 6), "holdout_n": len(ho),
            "adversarial_accuracy": round(adv.accuracy, 6),
            "adversarial_mean_score": round(adv.mean_score, 6), "adversarial_n": len(adv),
            "adversarial_by_category": adv.by_category(),
            "adversarial_scope": adv_scope or "full",
            "transfer_score": round(transfer, 6)}
        if rw is not None:
            self._validation.update({
                "realworld_accuracy": round(rw.accuracy, 6),
                "realworld_mean_score": round(rw.mean_score, 6), "realworld_n": len(rw),
                "realworld_by_category": rw.by_category(),
                "realworld_source": getattr(cfg, "validation_realworld_path", None) or "bundled"})
        elif rw_error is not None:
            self._validation["realworld_error"] = rw_error
        if frontier_block is not None:
            self._validation["frontier"] = frontier_block
        if grounded_block is not None:
            self._validation["grounded"] = grounded_block
        return self._validation

    def _run_grounded(self, solver: Any) -> Optional[Dict[str, Any]]:
        """Run one world-grounded experiment pass and return its block, or None.

        NYXARA predicts a program's output and is graded by REAL execution in the isolated sandbox —
        an outcome she did not store in advance. Best-effort: any failure (no sandbox, etc.) degrades
        to None so the other rulers carry the transfer score unchanged."""
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "grounded_experiments", True)):
            return None
        try:
            from nyxara.growth.grounded_experiments import GroundedExperiment
            exp = GroundedExperiment(settings=self.settings)
            rep = exp.evaluate(solver, trials=int(getattr(cfg, "grounded_experiments_budget", 5)))
            return rep.to_dict()
        except Exception:  # noqa: BLE001 — grounded is a ruler; on failure drop it, never zero
            return None

    def _run_curriculum(self, solver: Any) -> Optional[Dict[str, Any]]:
        """Run one open-ended auto-curriculum pass and return its frontier block, or None.

        Generates fresh, prover-certified problems at the edge of NYXARA's capability, grades the
        same sovereign solver the other rulers use, and advances/retreats the frontier. Best-effort:
        any failure degrades to None so the other rulers carry the transfer score unchanged."""
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "auto_curriculum_enabled", True)):
            return None
        try:
            from nyxara.growth.curriculum import AutoCurriculum
            cur = AutoCurriculum(memory=self.memory, settings=self.settings)
            rep = cur.evaluate(solver, per_tier=int(getattr(cfg, "curriculum_per_tier", 4)))
            return rep.to_dict()
        except Exception:  # noqa: BLE001 — the curriculum is a ruler; on failure drop it, never zero
            return None

    # ---- (4) self weakness detection ---- #
    def detect_weaknesses(self) -> Any:
        from nyxara.growth.weakness import WeaknessSynthesizer
        code = self._code if self._code is not None else self.review_code()
        arch = self._arch if self._arch is not None else self.analyze_architecture()
        bench = self._bench
        if bench is None and self.settings.self_improvement.benchmark_in_cycle:
            bench = self.run_benchmarks()
        return WeaknessSynthesizer().synthesize(code=code, arch=arch, bench=bench)

    # ---- (5) self optimisation (auto-apply source + safe set) ---- #
    def optimize(self, weaknesses: Any = None, *, enact: Optional[bool] = None
                 ) -> SelfImprovementReport:
        cfg = self.settings.self_improvement
        if enact is None:
            enact = bool(cfg.autonomous_enact)
        # META-META: bind the current champion knob-set into the engine BEFORE this cycle runs, so
        # the cycle's recursion depth / edit budget are the ones currently on trial (or champion).
        meta = self._meta_meta() if enact else None
        if meta is not None:
            meta.apply(self.settings)
        wreport = weaknesses if weaknesses is not None else self.detect_weaknesses()
        report = SelfImprovementReport(
            code=self._code.to_dict() if self._code is not None else None,
            architecture=self._arch.to_dict() if self._arch is not None else None,
            benchmark=self._bench, validation=self._validation,
            weaknesses=wreport.to_dict(), enacted=bool(enact))
        # surface the open-ended curriculum + grounded scores for the index diagnostics + summary
        if isinstance(self._validation, dict) and isinstance(self._validation.get("frontier"), dict):
            report.frontier = float(self._validation["frontier"].get("frontier_score", 0.0))
        if isinstance(self._validation, dict) and isinstance(self._validation.get("grounded"), dict):
            report.grounded = float(self._validation["grounded"].get("grounded_score", 0.0))

        # --- safe, non-source enactment (lessons + tuning) --- #
        report.lessons_stored = self._store_lessons(wreport, enact=enact)
        report.tuned = self._maybe_tune(wreport, enact=enact)

        # --- compute the effort budget (I_t × compute) BEFORE attempting edits --- #
        self._compute_effort_budget(report)

        # --- auto-apply source edits under the gauntlet --- #
        if enact:
            self._apply_source_edits(wreport, report)

        # --- full wire: turn the index's directive into a real cross-system action --- #
        self._enact_directive(wreport, report, enact=enact)

        # --- update the intelligence index: I_(t+1) = f(I_t, C_available) --- #
        self._update_intelligence(report)

        # --- lifelong credit assignment: did this cycle's interventions raise the index? --- #
        self._credit_outcome(report)
        # --- cross-module bus: publish what the CODE channel learned for the other faculties --- #
        self._publish_signals(report)
        # --- META-META: score the active algorithm by the realised RAW capability gain, and evolve
        # the whole recursive tower. Fitness is Δbase (the momentum-free target), NOT the smoothed
        # index, so the genome may evolve the index smoothing without being able to game its score.
        if meta is not None and report.enacted and report.intelligence_base is not None \
                and self._prior_base is not None:
            try:
                meta.record(float(report.intelligence_base) - float(self._prior_base))
                meta.maybe_evolve()
                report.meta_meta = meta.status()
            except Exception:  # noqa: BLE001 — the meta-meta loop is advisory, never fatal
                pass
        return report

    # ---- the full cycle ---- #
    def run(self, *, enact: Optional[bool] = None,
            category: Optional[str] = None) -> SelfImprovementReport:
        self._code = self._arch = self._bench = None
        self._validation = None
        self._compute = self._effort = None
        self._ledger_state = None
        self._touched_arms = []
        self._prior_index = None
        self._prior_base = None
        self._prior_transfer = None
        self._last_signals = None
        # the index governs effort up front, before any expensive step, so it shapes how deeply
        # she benchmarks and reasons this cycle — not merely how the cycle is later summarised
        self._plan_effort()
        self._observe(category=category)
        weaknesses = self.detect_weaknesses()
        return self.optimize(weaknesses, enact=enact)

    def _observe(self, *, category: Optional[str]) -> None:
        """Run the three independent observation steps (review / architecture / benchmark).

        They write disjoint per-cycle caches, so running them concurrently yields exactly the
        same state as running them in order — just faster, since each is I/O- and
        subprocess-bound. Falls back to sequential when ``parallel_cycle`` is off or the
        thread pool is unavailable; an error in any one branch is contained to that branch's
        own graceful-empty handling."""
        cfg = self.settings.self_improvement
        do_bench = cfg.benchmark_in_cycle
        do_val = bool(do_bench and getattr(cfg, "validation_enabled", True))
        steps = [self.review_code, self.analyze_architecture]
        if do_bench:
            steps.append(lambda: self.run_benchmarks(category=category))
        # external validation writes its OWN disjoint cache (self._validation) and never feeds
        # weakness detection, so it composes with the other observation steps just like benchmark
        if do_val:
            steps.append(self.run_validation)
        if not getattr(self.settings.self_improvement, "parallel_cycle", True) or len(steps) < 2:
            for step in steps:
                step()
            return
        # warm the shared lazy handles single-threaded before fanning out
        self._llm_handle()
        self._plan_effort()
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(steps)) as pool:
                for fut in [pool.submit(s) for s in steps]:
                    fut.result()
        except Exception:  # noqa: BLE001 — never let parallelism break the cycle; run in order
            for step in steps:
                step()

    # ------------------------------------------------------------------ #
    # enactment helpers
    # ------------------------------------------------------------------ #
    def _store_lessons(self, wreport: Any, *, enact: bool) -> int:
        """Write the top weaknesses into semantic memory as durable lessons (always safe)."""
        if self.memory is None:
            return 0
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return 0
        stored = 0
        for w in wreport.top(8):
            text = (f"Self-improvement weakness [{w.source}/{w.severity:.2f}]: {w.title}. "
                    f"Remediation: {w.remediation}")
            try:
                self.memory.remember(
                    text, mem_type=MemoryType.SEMANTIC,
                    provenance=Provenance(SourceType.SELF_REFLECTION,
                                          confidence=min(1.0, 0.5 + 0.4 * w.severity)),
                    importance=min(1.0, 0.5 + 0.4 * w.severity),
                    tags=["self-improvement", "weakness", w.source])
                stored += 1
            except Exception:  # noqa: BLE001
                continue
        return stored

    def _maybe_tune(self, wreport: Any, *, enact: bool) -> Optional[Dict[str, Any]]:
        """Raise reasoning depth when capability is weak — *or* when the index has plateaued and
        its growth directive says to deepen reasoning. The index thus genuinely *drives* a
        decision (escalate on a stall, deepen when reasoning is the bottleneck), not merely caps
        it. Clamped to the config bound [1, 20]."""
        cfg = self.settings.self_improvement
        if not (enact and cfg.allow_tuning):
            return None
        self._plan_effort()
        directive = self._directive or {}
        escalate = bool(directive.get("escalate"))
        deepen = directive.get("action") == "deepen_reasoning"
        acc = (self._bench or {}).get("accuracy")
        weak = acc is not None and acc < 0.8
        # tune when capability is weak, OR the index plateaued (escalate), OR reasoning is the
        # diagnosed bottleneck — the directive turns a passive metric into an active driver
        if not (weak or escalate or deepen):
            return None
        try:
            current = int(self.settings.llm.recursive_improvement_iterations)
            # the intelligence index + compute set the ceiling on reasoning depth: a weaker mind
            # on a weaker machine is not permitted to reason as deeply as it cannot afford to.
            ceiling = 20
            if self._effort is not None and self._effort.get("recursion_depth"):
                ceiling = max(1, int(self._effort["recursion_depth"]))
            step = 3 if (deepen or escalate) else 2      # push harder when the index demands it
            target = max(1, min(ceiling, current + step))
            if target == current:
                return None
            self.settings.llm.recursive_improvement_iterations = target
            reason = (f"directive '{directive.get('action')}'"
                      + (" — plateau, escalating" if escalate else "")
                      + (f"; accuracy {acc:.0%}" if acc is not None else "")
                      + f" (depth ceiling {ceiling})")
            return {"recursive_improvement_iterations": {"from": current, "to": target},
                    "reason": reason, "directive": directive.get("action")}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _plan_effort(self) -> None:
        """Compute this cycle's effort budget from the persisted index + available compute, once,
        before any expensive step. Leaves ``self._effort`` as None when the index is disabled or
        unavailable — in which case every effort lever falls back to its full/unscaled default."""
        if self._effort is not None:
            return
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "intelligence_index_enabled", True)):
            return
        try:
            intel = self._intel()
            prior = intel.load()
            compute = self._compute_report()
        except Exception:  # noqa: BLE001 — effort scaling is advisory, never fatal
            return
        self._prior_index = float(getattr(prior, "index", 0.0) or 0.0)
        # the raw, momentum-free capability target carried on the prior state — the meta tower's
        # fitness anchor (so evolving the index smoothing can never inflate the meta score).
        try:
            self._prior_base = float(getattr(prior, "last_inputs", {}).get(
                "base", self._prior_index))
        except Exception:  # noqa: BLE001
            self._prior_base = self._prior_index
        # the external held-out score carried on the prior state — the anchor the ledger credits
        # against, so an edit is reinforced by the gain it produced on tasks it never trained on
        try:
            self._prior_transfer = float(getattr(prior, "last_inputs", {}).get("transfer", 0.0))
        except Exception:  # noqa: BLE001
            self._prior_transfer = 0.0
        # substrate self-expansion: a survey of the parallelism/cluster NYXARA holds lifts the
        # effort ceiling (never lowers it) inside effort_budget, bounded by the absolute hard cap.
        try:
            self._effort = intel.effort_budget(prior, compute, substrate=self._substrate_mgr())
        except Exception:  # noqa: BLE001
            try:
                self._effort = intel.effort_budget(prior, compute)
            except Exception:  # noqa: BLE001
                self._effort = None
        # the index also DIAGNOSES the bottleneck up front (from the persisted prior signals), so
        # this cycle's effort is steered toward the weakest dimension. When planning is on, the
        # uncertainty-aware Thompson planner chooses under the index's per-dimension posteriors,
        # discounted by affordability and the ledger's learned per-action payoff; otherwise it
        # falls back to the greedy point-estimate directive. Computed independently so a directive
        # failure never clears the (separate) effort budget.
        try:
            if bool(getattr(self.settings.self_improvement, "plan_actions", True)):
                scores = None
                if self._ledger_enabled():
                    try:
                        scores = self._bandit_obj().action_scores(
                            list(("deepen_reasoning", "train_self_model",
                                  "resolve_weaknesses", "acquire_knowledge")),
                            state=self._ledger_state_obj())
                    except Exception:  # noqa: BLE001
                        scores = None
                self._directive = intel.plan_action(prior, compute, action_scores=scores)
            else:
                self._directive = intel.growth_directive(prior, compute)
        except Exception:  # noqa: BLE001
            self._directive = None

    def _compute_effort_budget(self, report: SelfImprovementReport) -> None:
        """Record the planned effort budget (and compute) on the report for audit."""
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "intelligence_index_enabled", True)):
            return
        self._plan_effort()
        if self._effort is None:
            return
        try:
            report.effort_budget = dict(self._effort)
            compute = self._compute_report()
            report.compute = compute.to_dict() if hasattr(compute, "to_dict") else None
            mgr = self._substrate_mgr()
            if mgr is not None:
                report.substrate = mgr.survey().to_dict()
        except Exception:  # noqa: BLE001 — recording is advisory, never fatal
            pass

    def _update_intelligence(self, report: SelfImprovementReport) -> None:
        """Fold this cycle's signals + compute into the index and persist it (best-effort)."""
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "intelligence_index_enabled", True)):
            return
        try:
            intel = self._intel()
            compute = self._compute_report()
            prior = intel.load()
            signals = intel.compute_signals(report)
            self._last_signals = dict(signals)
            self._last_signals["capacity"] = intel.compute_capacity(compute)
            # declare whether an EXTERNAL held-out score was really measured this cycle, so the
            # index tracks transfer and runs the Goodhart guard only on a genuine validation
            validated = bool(self._validation is not None
                             and "transfer_score" in (self._validation or {}))
            state = intel.update(prior, signals, compute, validated=validated)
            intel.save(state)
            report.intelligence_index = round(float(state.index), 6)
            report.intelligence_t = int(state.t)
            # the momentum-free raw target this cycle — the meta tower is scored by its delta
            report.intelligence_base = round(
                float(state.last_inputs.get("base", state.index)), 6)
            # the freshly-updated index diagnoses the next bottleneck — surfaced for the growth
            # engine / observability so the loop knows where to invest next
            report.directive = intel.growth_directive(state, compute)
            if report.compute is None:
                report.compute = compute.to_dict() if hasattr(compute, "to_dict") else None
        except Exception:  # noqa: BLE001 — the index is a measurement, never fatal
            pass

    # ------------------------------------------------------------------ #
    # lifelong credit assignment — learn which interventions raise the index
    # ------------------------------------------------------------------ #
    def _prioritize(self, ranked: List[Any]) -> List[Any]:
        """Reorder weaknesses by the ledger's learned payoff, then by cross-module relevance.

        Two influences compose: the ledger's *learned* payoff (which intervention classes have
        actually raised the index), and the **cross-module signal bus** — what OTHER faculties
        flagged this cycle (a recurring failure reflection noticed, a blind spot the world model
        reported, the weakest benchmark category). So the code channel fixes what reflection and
        the world model are worried about, not only what it independently scored highest."""
        ordered = list(ranked)
        if self._ledger_enabled():
            try:
                feats = self._last_signals or {}
                ordered = self._bandit_obj().prioritize(
                    ordered, state=self._ledger_state_obj(), features=feats)
            except Exception:  # noqa: BLE001 — ranking is an optimisation, never required
                ordered = list(ranked)
        return self._cross_module_boost(ordered)

    def _cross_module_boost(self, ranked: List[Any]) -> List[Any]:
        """Stable-reorder weaknesses so those the OTHER faculties flagged rise to the front.

        Reads the shared signal bus's aggregated focus terms (posted by reflection, the world
        model and the benchmark) and moves matching weaknesses forward, preserving the incoming
        (ledger) order on ties. A no-op when nothing has been posted, so it never hurts."""
        try:
            from nyxara.growth.signal_bus import get_signal_bus
            terms = get_signal_bus().focus_terms()
            if not terms:
                return ranked
            order = {id(w): i for i, w in enumerate(ranked)}

            def relevance(w: Any) -> float:
                text = (f"{getattr(w, 'title', '')} {getattr(w, 'source', '')} "
                        f"{getattr(w, 'remediation', '')}").lower()
                return sum(wt for t, wt in terms.items() if t in text)

            return sorted(ranked, key=lambda w: (-relevance(w), order[id(w)]))
        except Exception:  # noqa: BLE001 — cross-module steering is advisory, never required
            return ranked

    def _publish_signals(self, report: SelfImprovementReport) -> None:
        """Publish the code channel's findings onto the shared bus for the other faculties.

        Closes the feedback loop in the other direction: the weakest benchmark categories become
        focus terms that steer the *next* cycle's editing AND any other reader (research, foundry),
        and the realised index delta is posted as a health signal. Best-effort; never fatal."""
        try:
            from nyxara.growth.signal_bus import get_signal_bus
            bus = get_signal_bus()
            for topic in self._weakest_categories(limit=3):
                bus.post("weak_category", topic, source="benchmark", weight=1.0)
            if report.intelligence_index is not None and self._prior_index is not None:
                bus.post("index_delta",
                         float(report.intelligence_index) - float(self._prior_index),
                         source="code", weight=1.0)
            # surface the external-validation health so other faculties can react to overfitting
            if report.directive is not None:
                bus.post("transfer_gain", float(report.directive.get("transfer_gain", 0.0)),
                         source="validation", weight=1.0)
                if report.directive.get("goodhart"):
                    bus.post("goodhart", 1.0, source="validation", weight=1.0)
        except Exception:  # noqa: BLE001 — publishing is advisory, never fatal
            pass

    def _mark_arm(self, weakness: Any) -> None:
        """Remember that this weakness-class was acted on, to credit it at cycle end."""
        if getattr(self, "_touched_arms", None) is None or not self._ledger_enabled():
            return
        try:
            from nyxara.growth.credit import EditStrategyBandit
            self._touched_arms.append(EditStrategyBandit.weakness_key(weakness))
        except Exception:  # noqa: BLE001
            pass

    def _credit_outcome(self, report: SelfImprovementReport) -> None:
        """Fold the realised index change into the ledger, crediting every arm acted on this cycle.

        The reward is the change in the persisted index from before to after the cycle: an
        intervention that coincided with a rising index is reinforced, one that coincided with a
        fall is discouraged — so future cycles flow effort to what actually works. Best-effort and
        measurement-only; never touches a gate. Only runs after a real (enacted) cycle — a
        read-only dry-run takes no intervention, so there is nothing to credit."""
        if not report.enacted or not self._ledger_enabled() or not self._touched_arms:
            return
        if report.intelligence_index is None or self._prior_index is None:
            return
        try:
            led = self._ledger_obj()
            state = self._ledger_state_obj() or led.load()
            # credit by the EXTERNAL transfer delta (gain on held-out / adversarial tasks) rather
            # than the self-referential index delta — an intervention is reinforced only when it
            # actually transferred, so the ledger can never learn to reward Goodharting the proxy.
            cfg = self.settings.self_improvement
            if (bool(getattr(cfg, "credit_on_transfer", True)) and self._validation
                    and "transfer_score" in self._validation):
                cur_t = float(self._validation.get("transfer_score", 0.0))
                delta = cur_t - float(self._prior_transfer or 0.0)
                credit_basis = "transfer"
            else:
                delta = float(report.intelligence_index) - float(self._prior_index)
                credit_basis = "index"
            arms = list(dict.fromkeys(self._touched_arms))   # dedupe, preserve order
            led.record_delta(state, arms, delta)
            led.save(state)
            # teach the (optional) learned forecaster the same outcome, in context
            fc = self._forecaster_obj()
            if fc is not None and self._last_signals is not None:
                feats = dict(self._last_signals)
                feats["index"] = float(report.intelligence_index)
                reward = led.reward_from_delta(delta)
                for arm in arms:
                    fc.observe(arm, feats, reward)
            report.ledger = {"delta": round(delta, 6), "arms": arms,
                             "basis": credit_basis,
                             "reward": round(led.reward_from_delta(delta), 4),
                             "summary": state.summary()}
        except Exception:  # noqa: BLE001 — credit assignment is advisory, never fatal
            pass

    # ------------------------------------------------------------------ #
    # full wire — the index's directive becomes a real cross-system action
    # ------------------------------------------------------------------ #
    def _enact_directive(self, wreport: Any, report: SelfImprovementReport, *,
                         enact: bool) -> None:
        """Dispatch the planned growth directive into a real action elsewhere in the system.

        ``deepen_reasoning`` and ``resolve_weaknesses`` are already handled by the tuning and
        source-edit paths above; this method wires the two that previously had nowhere to go:
        ``train_self_model`` (drive the gauntlet-gated foundry) and ``acquire_knowledge`` (enqueue
        her weakest benchmark categories onto the research/investigation queues the idle scientist
        already drains). Each dispatch honours its own subsystem's gates and degrades to a clean
        no-op when the handle or permission is absent."""
        directive = self._directive or {}
        action = directive.get("action")
        report.directive_action = action
        if not action:
            return
        cfg = self.settings.self_improvement
        dispatch_on = bool(getattr(cfg, "enable_directive_dispatch", True))
        if not (enact and dispatch_on):
            report.directive_results = {"action": action, "dispatched": False,
                                        "reason": "dry-run or directive dispatch disabled"}
            return
        # this is a real, enacted cycle — credit the action arm for the index change it drives
        if self._ledger_enabled():
            try:
                from nyxara.growth.credit import arm_key
                self._touched_arms.append(arm_key(action=action))
            except Exception:  # noqa: BLE001
                pass
        if action == "train_self_model":
            report.directive_results = self._dispatch_train_self_model()
        elif action == "acquire_knowledge":
            report.directive_results = self._dispatch_acquire_knowledge(wreport)
        else:
            report.directive_results = {"action": action, "dispatched": False,
                                        "reason": "handled by source-edit / tuning path"}

    def _dispatch_train_self_model(self) -> Dict[str, Any]:
        """Drive one gauntlet-gated foundry cycle via the growth engine (forge a better brain)."""
        eng = self.growth_engine
        if eng is None or not hasattr(eng, "improve_self"):
            return {"action": "train_self_model", "dispatched": False,
                    "reason": "no growth engine handle"}
        if not bool(getattr(self.settings.foundry, "enabled", False)):
            return {"action": "train_self_model", "dispatched": False,
                    "reason": "foundry disabled (NYXARA_FOUNDRY__ENABLED)"}
        # respect live oversight if the core exposes it (same gate AutoForge honours)
        oversight = getattr(self.core, "oversight", None)
        if oversight is not None:
            try:
                if not oversight.gate():
                    return {"action": "train_self_model", "dispatched": False,
                            "reason": "oversight gate closed (paused/scrammed)"}
            except Exception:  # noqa: BLE001
                pass
        try:
            results = eng.improve_self(generations=1)
            promoted = sum(1 for r in (results or [])
                           if bool(getattr(r, "promoted", False)))
            return {"action": "train_self_model", "dispatched": True,
                    "cycles": len(results or []), "promoted": promoted}
        except Exception as exc:  # noqa: BLE001 — forging is heavy/optional; never fatal
            return {"action": "train_self_model", "dispatched": False, "reason": str(exc)}

    def _dispatch_acquire_knowledge(self, wreport: Any) -> Dict[str, Any]:
        """Enqueue her weakest benchmark categories as research/investigation topics.

        The idle scientist + researcher already drain ``core._research_queue`` and
        ``core._investigation_queue`` (orchestrator idle maintenance), so this closes the loop:
        a thin-knowledge diagnosis becomes real, grounded inquiry on exactly where she fails."""
        core = self.core
        if core is None:
            return {"action": "acquire_knowledge", "dispatched": False,
                    "reason": "no core handle"}
        topics = self._weakest_categories(limit=3)
        if not topics:
            return {"action": "acquire_knowledge", "dispatched": False,
                    "reason": "no failing categories to research"}
        enq = 0
        for q in topics:
            for attr in ("_investigation_queue", "_research_queue"):
                queue = getattr(core, attr, None)
                if isinstance(queue, list) and q not in queue:
                    queue.append(q)
                    enq += 1
        return {"action": "acquire_knowledge", "dispatched": enq > 0,
                "enqueued": enq, "topics": topics}

    def _weakest_categories(self, *, limit: int = 3) -> List[str]:
        """The lowest-scoring benchmark categories this cycle — her real knowledge gaps."""
        bench = self._bench or {}
        by_cat = bench.get("by_category") or {}
        scored: List[tuple] = []
        for cat, val in by_cat.items():
            try:
                score = float(val.get("accuracy") if isinstance(val, dict) else val)
            except Exception:  # noqa: BLE001
                continue
            scored.append((score, str(cat)))
        scored.sort()
        return [f"improve capability: {cat}" for _, cat in scored[:limit] if cat]

    def _apply_source_edits(self, wreport: Any, report: SelfImprovementReport) -> None:
        from nyxara.growth.self_optimize import EditGenerator, LLMEditGenerator, Optimizer
        cfg = self.settings.self_improvement
        gen = EditGenerator(llm=self._llm_handle())
        # the self-authored generator authors real fixes the transforms cannot express, using
        # NYXARA's OWN model (the `self` provider) — never an external LLM when self_authored_only
        # is set. It self-disables unless allow_llm_edits is set AND a real author is available, so
        # on a bare machine (own model not yet trained) this is simply None-equivalent and the
        # deterministic transform path stands on its own.
        llm_gen: Any = LLMEditGenerator(llm=self._llm_handle(), settings=self.settings)
        if not llm_gen.available():
            llm_gen = None
        optimizer = Optimizer(root=self.root, settings=self.settings,
                              journal=self.journal,
                              permissions=getattr(self.core, "permissions", None))
        try:
            budget = int(getattr(cfg, "max_edits_per_cycle", 3))
            # compute scales the budget: a weaker machine attempts fewer self-edits (the index
            # never raises the ceiling above the config max — only lowers it).
            if self._effort is not None:
                budget = min(budget, int(self._effort.get("max_edits_per_cycle", budget)))
            # lifelong credit: order the edits by learned payoff (Thompson over the ledger),
            # falling back to raw severity when the ledger is empty/disabled.
            ranked = self._prioritize(wreport.ranked())
            self._enact_edits(ranked, gen, optimizer, budget, report, llm_gen=llm_gen)
        finally:
            optimizer.close()

    def _generate_edit(self, weakness: Any, gen: Any, llm_gen: Any) -> Any:
        """Author one edit: deterministic transform first (instant/offline), then a self-authored
        whole-file rewrite (NYXARA's own model) for weaknesses flagged ``edit_strategy == "llm"``
        when an author is available."""
        edit = gen.generate(weakness)               # deterministic, AST-validated, always tried
        if edit is not None:
            return edit
        if llm_gen is not None and getattr(weakness, "edit_strategy", "") == "llm":
            return llm_gen.generate(weakness)        # real, whole-file self-authored fix
        return None

    def _enact_edits(self, ranked: Any, gen: Any, optimizer: Any, budget: int,
                     report: SelfImprovementReport, *, llm_gen: Any = None) -> None:
        """Apply up to ``budget`` source edits in severity order, at most one *kept* edit per
        file per cycle in the outer pass.

        Weakness line numbers come from a single review pass. A kept edit can change a file's
        line count (e.g. removing an import), which would invalidate the line numbers of every
        other weakness in that same file — so a second same-file edit this cycle would target a
        stale line and either miss or mislocate. Once a file has a kept edit, the outer loop
        defers it; instead, when recursion is enabled, :meth:`_recurse_file` re-reviews just that
        file (fresh line numbers) and chains further edits on it — the genuine *recursive* in
        RSI. Edits that roll back leave the file byte-for-byte unchanged, so they never claim
        the file."""
        edited_files: set = set()
        edits_done = 0
        for w in ranked:
            if edits_done >= budget:
                break
            if not getattr(w, "is_source_edit", False):
                continue
            edit = self._generate_edit(w, gen, llm_gen)
            if edit is None:
                continue
            if edit.file in edited_files:        # a kept edit already shifted this file's lines
                continue
            outcome = optimizer.apply(edit)
            report.optimizations.append(outcome.to_dict())
            if outcome.applied:
                edits_done += 1
            if outcome.kept:
                report.kept += 1
                self._mark_arm(w)                # credit this weakness-class at cycle end
                edited_files.add(edit.file)      # claim the file only once it actually changed
                edits_done += self._recurse_file(
                    edit.file, gen, llm_gen, optimizer, budget - edits_done, report)
            if outcome.rolled_back:
                report.rolled_back += 1

    def _recurse_file(self, file_path: str, gen: Any, llm_gen: Any, optimizer: Any,
                      remaining: int, report: SelfImprovementReport) -> int:
        """Re-review one just-improved file and chain further edits on it (depth-bounded).

        This is what makes self-improvement *recursive*: each kept edit re-reviews the changed
        source with fresh line numbers and lets the next fix build on the last, all within the
        same reversible gauntlet. Returns the number of edits *applied* (kept or rolled back) so
        the caller keeps honouring the global per-cycle budget."""
        cfg = getattr(getattr(self, "settings", None), "self_improvement", None)
        depth = int(getattr(cfg, "llm_edit_recursion_depth", 0)) if cfg is not None else 0
        if depth <= 0 or remaining <= 0:
            return 0
        applied_total = 0
        for _ in range(depth):
            if applied_total >= remaining:
                break
            progressed = False
            for w in self._rereview_file(file_path):
                if applied_total >= remaining:
                    break
                if not getattr(w, "is_source_edit", False):
                    continue
                edit = self._generate_edit(w, gen, llm_gen)
                if edit is None:
                    continue
                outcome = optimizer.apply(edit)
                report.optimizations.append(outcome.to_dict())
                if outcome.applied:
                    applied_total += 1
                if outcome.kept:
                    report.kept += 1
                    self._mark_arm(w)            # credit this weakness-class at cycle end
                    progressed = True
                    break                        # re-review with fresh line numbers
                if outcome.rolled_back:
                    report.rolled_back += 1
            if not progressed:                   # nothing more to safely improve on this file
                break
        return applied_total

    def _rereview_file(self, file_path: str) -> List[Any]:
        """Ranked weaknesses for a single file (used by the recursion to re-target fresh lines)."""
        try:
            from pathlib import Path

            from nyxara.growth.self_review import CodeReviewReport, SelfReviewer
            from nyxara.growth.weakness import WeaknessSynthesizer
            cfg = self.settings.self_improvement
            reviewer = SelfReviewer(
                root=self.root, llm=self._llm_handle(),
                max_function_length=cfg.max_function_length,
                max_complexity=cfg.max_complexity, max_args=cfg.max_args,
                enable_llm_enrichment=cfg.enable_llm_enrichment)
            findings = reviewer.review_file(Path(file_path))
            code = CodeReviewReport(findings=findings, files_scanned=1)
            return WeaknessSynthesizer().synthesize(code=code).ranked()
        except Exception:  # noqa: BLE001 — a re-review failure just ends the recursion
            return []


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA recursive-self-improvement self-test")
    print("=" * 70)

    rsi = RecursiveSelfImprovement()
    report = rsi.run(enact=False)              # dry-run: analyse everything, change nothing
    print("\n" + report.summary())

    assert report.code is not None and report.code["files_scanned"] > 20
    assert report.architecture is not None and report.architecture["n_modules"] > 50
    assert report.weaknesses is not None
    acc = report.benchmark["accuracy"]
    assert 0.0 <= acc <= 1.0, "accuracy must be a real fraction"
    assert "handoff" in report.benchmark
    assert report.kept == 0 and report.rolled_back == 0 and not report.enacted
    assert not report.optimizations, "dry-run must apply no source edits"

    # full wire: a directive is planned every cycle, but a dry-run dispatches nothing and
    # credits no arm (credit assignment reflects only real, enacted interventions)
    assert report.directive_action is not None, "the index must plan an action each cycle"
    assert report.directive_results is not None and not report.directive_results["dispatched"]
    assert report.ledger is None, "a read-only dry-run must credit no arms"

    # the intelligence index is measured every cycle: I_(t+1) = f(I_t, C_available)
    assert report.intelligence_index is not None and 0.0 <= report.intelligence_index <= 1.0
    assert report.intelligence_t is not None and report.intelligence_t >= 1
    assert report.effort_budget is not None and "max_edits_per_cycle" in report.effort_budget
    assert report.compute is not None and "recommended_device" in report.compute
    print(f"\nintelligence index  : I_{report.intelligence_t} = {report.intelligence_index:.4f}")

    # external ground-truth validation ran on held-out + adversarial tasks the loop never edits
    assert report.validation is not None and "transfer_score" in report.validation
    v = report.validation
    assert 0.0 <= v["transfer_score"] <= 1.0
    assert v["holdout_n"] >= 1 and v["adversarial_n"] >= 1
    # the index carries the transfer signal and a Goodhart verdict for this cycle
    assert report.directive is not None and "goodhart" in report.directive
    print(f"validation (transfer): {v['transfer_score']:.4f} "
          f"(held-out {v['holdout_mean_score']:.2f}, adversarial {v['adversarial_mean_score']:.2f})")

    print("\nALL SELF-TESTS PASSED ✓")
