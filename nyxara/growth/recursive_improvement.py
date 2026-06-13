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
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "architecture": self.architecture,
                "benchmark": self.benchmark, "weaknesses": self.weaknesses,
                "optimizations": self.optimizations, "lessons_stored": self.lessons_stored,
                "tuned": self.tuned, "kept": self.kept, "rolled_back": self.rolled_back,
                "enacted": self.enacted, "at": self.at}

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
        if self.tuned:
            lines.append(f"tuned           : {self.tuned}")
        return "\n".join(lines)


class RecursiveSelfImprovement:
    """The codebase-level self-improvement loop, wired to NYXARA's real faculties."""

    def __init__(self, *, core: Any = None, memory: Any = None, settings: Any = None,
                 llm: Any = None, root: Any = None, growth_engine: Any = None,
                 journal: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.core = core
        self.memory = memory if memory is not None else getattr(core, "memory", None)
        self.journal = journal if journal is not None else getattr(core, "journal", None)
        self.growth_engine = growth_engine
        self.root = root
        self._llm = llm
        # per-cycle caches
        self._code = None
        self._arch = None
        self._bench = None

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
        try:
            report = bench.run(core_solver(), category=category)
        except Exception as exc:  # noqa: BLE001
            self._bench = {"error": f"benchmark failed: {exc}", "accuracy": 0.0,
                           "failures": [], "by_category": {}}
            return self._bench
        handoff: Dict[str, int] = {}
        try:
            _, handoff = run_router(bench, settings=self.settings, category=category)
        except Exception:  # noqa: BLE001 — router is optional signal
            handoff = {}
        self._bench = {
            "accuracy": report.accuracy, "mean_score": report.mean_score,
            "by_category": report.by_category(),
            "failures": [r.to_dict() for r in report.failures()],
            "handoff": handoff, "report": report.to_dict()}
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

        # --- auto-apply source edits under the gauntlet --- #
        if enact:
            self._apply_source_edits(wreport, report)
        return report

    # ---- the full cycle ---- #
    def run(self, *, enact: Optional[bool] = None,
            category: Optional[str] = None) -> SelfImprovementReport:
        self._code = self._arch = self._bench = None
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
            target = max(1, min(20, current + 2))   # bound enforced by config Field(1..20) too
            if target == current:
                return None
            self.settings.llm.recursive_improvement_iterations = target
            return {"recursive_improvement_iterations": {"from": current, "to": target},
                    "reason": f"benchmark accuracy {acc:.0%} below 80%"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _apply_source_edits(self, wreport: Any, report: SelfImprovementReport) -> None:
        from nyxara.growth.self_optimize import EditGenerator, Optimizer
        cfg = self.settings.self_improvement
        gen = EditGenerator(llm=self._llm_handle())
        optimizer = Optimizer(root=self.root, settings=self.settings,
                              journal=self.journal,
                              permissions=getattr(self.core, "permissions", None))
        try:
            budget = int(getattr(cfg, "max_edits_per_cycle", 3))
            edits_done = 0
            for w in wreport.ranked():
                if edits_done >= budget:
                    break
                if not getattr(w, "is_source_edit", False):
                    continue
                edit = gen.generate(w)
                if edit is None:
                    continue
                outcome = optimizer.apply(edit)
                report.optimizations.append(outcome.to_dict())
                if outcome.applied:
                    edits_done += 1
                if outcome.kept:
                    report.kept += 1
                if outcome.rolled_back:
                    report.rolled_back += 1
        finally:
            optimizer.close()


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

    print("\nALL SELF-TESTS PASSED ✓")
