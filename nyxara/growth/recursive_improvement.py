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
    compute: Optional[Dict[str, Any]] = None
    effort_budget: Optional[Dict[str, Any]] = None
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "architecture": self.architecture,
                "benchmark": self.benchmark, "weaknesses": self.weaknesses,
                "optimizations": self.optimizations, "lessons_stored": self.lessons_stored,
                "tuned": self.tuned, "kept": self.kept, "rolled_back": self.rolled_back,
                "enacted": self.enacted, "intelligence_index": self.intelligence_index,
                "intelligence_t": self.intelligence_t, "compute": self.compute,
                "effort_budget": self.effort_budget, "at": self.at}

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
        if self.intelligence_index is not None:
            lines.append(f"intelligence    : I_{self.intelligence_t} = "
                         f"{self.intelligence_index:.4f}")
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
        # per-cycle caches
        self._code = None
        self._arch = None
        self._bench = None
        self._compute = None
        self._effort: Optional[Dict[str, Any]] = None

    # ---- intelligence index: I_(t+1) = f(I_t, C_available) ---- #
    def _intel(self) -> Any:
        if self._intelligence is None:
            from nyxara.growth.intelligence import IntelligenceIndex
            self._intelligence = IntelligenceIndex(memory=self.memory, settings=self.settings)
        return self._intelligence

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
        bench = build_default_benchmark()
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
        wreport = weaknesses if weaknesses is not None else self.detect_weaknesses()
        report = SelfImprovementReport(
            code=self._code.to_dict() if self._code is not None else None,
            architecture=self._arch.to_dict() if self._arch is not None else None,
            benchmark=self._bench, weaknesses=wreport.to_dict(), enacted=bool(enact))

        # --- safe, non-source enactment (lessons + tuning) --- #
        report.lessons_stored = self._store_lessons(wreport, enact=enact)
        report.tuned = self._maybe_tune(wreport, enact=enact)

        # --- compute the effort budget (I_t × compute) BEFORE attempting edits --- #
        self._compute_effort_budget(report)

        # --- auto-apply source edits under the gauntlet --- #
        if enact:
            self._apply_source_edits(wreport, report)

        # --- update the intelligence index: I_(t+1) = f(I_t, C_available) --- #
        self._update_intelligence(report)
        return report

    # ---- the full cycle ---- #
    def run(self, *, enact: Optional[bool] = None,
            category: Optional[str] = None) -> SelfImprovementReport:
        self._code = self._arch = self._bench = None
        self._compute = self._effort = None
        # the index governs effort up front, before any expensive step, so it shapes how deeply
        # she benchmarks and reasons this cycle — not merely how the cycle is later summarised
        self._plan_effort()
        self.review_code()
        self.analyze_architecture()
        if self.settings.self_improvement.benchmark_in_cycle:
            self.run_benchmarks(category=category)
        weaknesses = self.detect_weaknesses()
        return self.optimize(weaknesses, enact=enact)

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
        """Raise reasoning depth when capability is weak (clamped to the config bound [1,20])."""
        cfg = self.settings.self_improvement
        if not (enact and cfg.allow_tuning):
            return None
        acc = (self._bench or {}).get("accuracy")
        if acc is None or acc >= 0.8:
            return None
        try:
            current = int(self.settings.llm.recursive_improvement_iterations)
            # the intelligence index + compute set the ceiling on reasoning depth: a weaker mind
            # on a weaker machine is not permitted to reason as deeply as it cannot afford to, so
            # the index genuinely governs depth — it is not only an edit-count knob.
            ceiling = 20
            self._plan_effort()
            if self._effort is not None and self._effort.get("recursion_depth"):
                ceiling = max(1, int(self._effort["recursion_depth"]))
            target = max(1, min(ceiling, current + 2))   # config Field(1..20) also bounds it
            if target == current:
                return None
            self.settings.llm.recursive_improvement_iterations = target
            return {"recursive_improvement_iterations": {"from": current, "to": target},
                    "reason": f"benchmark accuracy {acc:.0%} below 80% (index ceiling {ceiling})"}
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
            self._effort = intel.effort_budget(intel.load(), self._compute_report())
        except Exception:  # noqa: BLE001 — effort scaling is advisory, never fatal
            self._effort = None

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
            state = intel.update(prior, signals, compute)
            intel.save(state)
            report.intelligence_index = round(float(state.index), 6)
            report.intelligence_t = int(state.t)
            if report.compute is None:
                report.compute = compute.to_dict() if hasattr(compute, "to_dict") else None
        except Exception:  # noqa: BLE001 — the index is a measurement, never fatal
            pass

    def _apply_source_edits(self, wreport: Any, report: SelfImprovementReport) -> None:
        from nyxara.growth.self_optimize import EditGenerator, LLMEditGenerator, Optimizer
        cfg = self.settings.self_improvement
        gen = EditGenerator(llm=self._llm_handle())
        # the LLM generator authors real fixes the transforms cannot express; it self-disables
        # unless allow_llm_edits is set AND a real (non-mock/self) provider is available, so on a
        # bare/offline machine this is simply None-equivalent and the deterministic path stands
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
            self._enact_edits(wreport.ranked(), gen, optimizer, budget, report, llm_gen=llm_gen)
        finally:
            optimizer.close()

    def _generate_edit(self, weakness: Any, gen: Any, llm_gen: Any) -> Any:
        """Author one edit: deterministic transform first (instant/offline), LLM fallback for
        weaknesses flagged ``edit_strategy == "llm"`` when a real provider is available."""
        edit = gen.generate(weakness)               # deterministic, AST-validated, always tried
        if edit is not None:
            return edit
        if llm_gen is not None and getattr(weakness, "edit_strategy", "") == "llm":
            return llm_gen.generate(weakness)        # real, whole-file authored fix
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

    # the intelligence index is measured every cycle: I_(t+1) = f(I_t, C_available)
    assert report.intelligence_index is not None and 0.0 <= report.intelligence_index <= 1.0
    assert report.intelligence_t is not None and report.intelligence_t >= 1
    assert report.effort_budget is not None and "max_edits_per_cycle" in report.effort_budget
    assert report.compute is not None and "recommended_device" in report.compute
    print(f"\nintelligence index  : I_{report.intelligence_t} = {report.intelligence_index:.4f}")

    print("\nALL SELF-TESTS PASSED ✓")
